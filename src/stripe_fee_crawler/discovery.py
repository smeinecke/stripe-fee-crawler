"""Market discovery and fee-page validation for Stripe."""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import urlparse

from lxml import html

from .exceptions import (
    AccessChallengeError,
    ContentSecurityError,
    FeePageError,
    MarketDiscoveryError,
    NetworkError,
    ParserError,
    PermanentHttpError,
    PermanentNetworkError,
    RateLimitError,
    TransientNetworkError,
    UnsupportedMarketError,
)
from .http import HttpClient, HttpResponse
from .market_detection import COUNTRY_NAME_TO_ISO, CURRENCY_BY_COUNTRY
from .models import CrawlConfiguration, Language, Market

logger = logging.getLogger(__name__)

# Bootstrap seed list reviewed against live Stripe pages as of July 2026.
# Locale identifiers are the language-country form used by Stripe URLs.
BOOTSTRAP_MARKETS: list[Market] = [
    Market(
        stripe_market_code="us",
        account_country="US",
        country_name="United States",
        region="north_america",
        locale="en-us",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com",
        preferred_language="en",
        default_currency="USD",
        status="supported",
    ),
    Market(
        stripe_market_code="gb",
        account_country="GB",
        country_name="United Kingdom",
        region="europe",
        locale="en-gb",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com/gb",
        preferred_language="en",
        default_currency="GBP",
        status="supported",
    ),
    Market(
        stripe_market_code="de",
        account_country="DE",
        country_name="Germany",
        region="europe",
        locale="en-de",
        languages=[Language(code="en", name="English"), Language(code="de", name="Deutsch")],
        url_prefix="https://stripe.com/en-de",
        preferred_language="en",
        default_currency="EUR",
        status="supported",
    ),
    Market(
        stripe_market_code="au",
        account_country="AU",
        country_name="Australia",
        region="asia_pacific",
        locale="en-au",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com/au",
        preferred_language="en",
        default_currency="AUD",
        status="supported",
    ),
    Market(
        stripe_market_code="jp",
        account_country="JP",
        country_name="Japan",
        region="asia_pacific",
        locale="en-jp",
        languages=[Language(code="en", name="English"), Language(code="ja", name="日本語")],
        url_prefix="https://stripe.com/en-jp",
        preferred_language="en",
        default_currency="JPY",
        status="supported",
    ),
    Market(
        stripe_market_code="in",
        account_country="IN",
        country_name="India",
        region="asia_pacific",
        locale="en-in",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com/in",
        preferred_language="en",
        default_currency="INR",
        status="supported",
    ),
    Market(
        stripe_market_code="ae",
        account_country="AE",
        country_name="United Arab Emirates",
        region="middle_east_africa",
        locale="en-ae",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com/ae",
        preferred_language="en",
        default_currency="AED",
        status="supported",
    ),
    Market(
        stripe_market_code="br",
        account_country="BR",
        country_name="Brazil",
        region="south_america",
        locale="en-br",
        languages=[Language(code="en", name="English"), Language(code="pt", name="Português")],
        url_prefix="https://stripe.com/en-br",
        preferred_language="en",
        default_currency="BRL",
        status="supported",
    ),
    Market(
        stripe_market_code="id",
        account_country="ID",
        country_name="Indonesia",
        region="asia_pacific",
        locale="en-id",
        languages=[Language(code="en", name="English")],
        url_prefix="https://stripe.com/en-id",
        preferred_language="en",
        default_currency="IDR",
        status="pricing_page_unavailable",
    ),
]

# Markets known to have a direct country URL without language prefix.
DIRECT_LOCALE_MARKETS: set[str] = {"us", "gb", "au", "nz", "ie", "in", "ae"}


def _country_name_to_iso(name: str) -> str | None:
    normalized = name.strip().lower()
    return COUNTRY_NAME_TO_ISO.get(normalized)


def _locale_for_country(country_code: str, language: str = "en") -> str:
    country_code = country_code.upper()
    if country_code in DIRECT_LOCALE_MARKETS:
        return f"{language}-{country_code.lower()}"
    return f"{language}-{country_code.lower()}"


def _pricing_url_for(market: Market) -> str:
    """Return a market-locale URL that is independent of crawler IP geolocation.

    Requesting ``/en-us/pricing`` sets the ``country=US`` cookie and redirects to
    the generic ``/pricing`` path, while ``/en-gb/pricing`` redirects to ``/gb/pricing``.
    All supported markets can therefore be requested with ``/{locale}/pricing``.
    """
    return f"https://stripe.com/{market.locale}/pricing"


def _payment_methods_url_for(market: Market) -> str:
    return f"https://stripe.com/{market.locale}/pricing/local-payment-methods"


def _is_html_response(response: HttpResponse) -> bool:
    content_type = response.headers.get("content-type", "").lower()
    return "text/html" in content_type or "application/xhtml" in content_type


def _canonical_market_code(iso_code: str, _language: str = "en") -> str:
    """Return the canonical Stripe market code (country code) for a market.

    The canonical code is the lower-case ISO 3166-1 alpha-2 account country.
    Locale variants (e.g. ``en-de`` and ``de-de``) are recorded as aliases.
    """
    return iso_code.lower()


def _extract_footer_markets(tree: Any) -> tuple[list[Market], dict[str, str]]:
    """Discover markets from the country selector in the page footer.

    Returns (markets, aliases) where aliases maps discovered locale slugs to the
    canonical stripe_market_code. Deduplication is by account country so that
    multiple language links for the same country collapse into one market.
    """
    markets: list[Market] = []
    aliases: dict[str, str] = {}
    seen_countries: set[str] = set()

    # The footer selector is a list of country links grouped by region.
    # Each link text is a country name and href contains the locale.
    links = tree.xpath("//footer//a[contains(@href, 'stripe.com') or contains(@href, '/')]")
    for link in links:
        href = link.get("href", "")
        text = " ".join(link.itertext()).strip()
        if not href or not text:
            continue
        parsed = urlparse(href)
        if parsed.hostname and not parsed.hostname.endswith("stripe.com"):
            continue
        path = parsed.path.lower().rstrip("/")
        if path in {"/pricing", "/pricing/local-payment-methods"}:
            locale = "us"
        else:
            match = re.match(r"^/(?P<locale>[a-z]{2}(?:-[a-z]{2})?)(?:/pricing)?$", path)
            if not match:
                match = re.match(r"^/(?P<locale>[a-z]{2})(?:/pricing)?$", path)
            if not match:
                continue
            locale = match.group("locale")
        country_name = text
        iso_code = _country_name_to_iso(country_name)
        if not iso_code:
            continue

        canonical_code = _canonical_market_code(iso_code, locale.split("-")[0])
        if iso_code in seen_countries:
            # Record the alternate locale as an alias for the canonical market.
            if locale != canonical_code:
                aliases[locale] = canonical_code
            continue
        seen_countries.add(iso_code)
        if locale != canonical_code:
            aliases[locale] = canonical_code

        url_prefix = f"https://stripe.com/{locale}"
        markets.append(
            Market(
                stripe_market_code=canonical_code,
                account_country=iso_code,
                country_name=country_name,
                region=None,
                locale=locale,
                languages=[Language(code=locale.split("-")[0], name=None)],
                url_prefix=url_prefix,
                preferred_language=locale.split("-")[0],
                default_currency=CURRENCY_BY_COUNTRY.get(iso_code),
                status="discovered",
            )
        )
    return markets, aliases


def _extract_footer_markets_from_text(html_text: str) -> tuple[list[Market], dict[str, str]]:
    """Discover markets from the footer country list using robust text matching."""
    tree = html.fromstring(html_text)
    return _extract_footer_markets(tree)


def get_bootstrap_markets() -> list[Market]:
    """Return a small, conservative bootstrap market list."""
    return [market.model_copy() for market in BOOTSTRAP_MARKETS]


def _bootstrap_aliases() -> dict[str, str]:
    """Return locale aliases implied by the bootstrap market list."""
    aliases: dict[str, str] = {}
    for market in BOOTSTRAP_MARKETS:
        canonical = market.stripe_market_code
        if market.locale != canonical:
            aliases[market.locale] = canonical
    return aliases


async def discover_markets(
    http_client: HttpClient,
    config: CrawlConfiguration,
    discovery_url: str = "https://stripe.com/pricing/local-payment-methods",
) -> tuple[list[Market], dict[str, str]]:
    """Discover Stripe markets from the country selector in a pricing page footer.

    Returns (markets, aliases). Falls back to the bootstrap list when dynamic
    discovery fails and the caller has not disabled the fallback.
    """
    try:
        response = await http_client.get(discovery_url)
    except (NetworkError, ParserError) as exc:
        logger.warning("Market discovery request failed: %s", exc)
        if not config.refresh_market_manifest:
            return get_bootstrap_markets(), _bootstrap_aliases()
        raise MarketDiscoveryError(f"Failed to retrieve Stripe discovery page: {exc}") from exc

    try:
        markets, aliases = _extract_footer_markets_from_text(response.text)
    except Exception as exc:
        logger.warning("Could not extract markets from discovery page: %s", exc)
        if not config.refresh_market_manifest:
            return get_bootstrap_markets(), _bootstrap_aliases()
        raise MarketDiscoveryError(f"Could not extract markets from discovery page: {exc}") from exc

    if markets:
        return markets, aliases

    if not config.refresh_market_manifest:
        return get_bootstrap_markets(), _bootstrap_aliases()
    raise MarketDiscoveryError("No markets found in discovery page footer")


def _page_locale_from_url(url: str) -> str | None:
    path = urlparse(url).path.lower().strip("/")
    if path == "pricing" or path.startswith("pricing/"):
        return "us"
    match = re.match(r"^(?P<locale>[a-z]{2}-[a-z]{2})/pricing", path)
    if match:
        return match.group("locale")
    match = re.match(r"^(?P<country>[a-z]{2})/pricing", path)
    if match:
        return match.group("country")
    return None


def _page_locale_from_html(tree: Any) -> str | None:
    html_tag = tree.xpath("//html")
    if html_tag:
        lang = html_tag[0].get("lang")
        if lang:
            return lang.lower().replace("_", "-")
    return None


def _is_pricing_page(response: HttpResponse, tree: Any | None = None) -> bool:
    """Validate that a page is a plausible Stripe pricing page."""
    if response.status_code != 200:
        return False
    if not _is_html_response(response):
        return False

    if tree is None:
        try:
            tree = html.fromstring(response.text)
        except Exception:
            return False
        if tree is None:
            return False

    text = " ".join(tree.itertext())
    pricing_signals = [
        "Standard pricing",
        "Custom pricing",
        "Pricing & Fees",
        "Preise und Gebühren",
        "Local payment methods",
        "Domestic card",
        "International card",
    ]
    if not any(signal.lower() in text.lower() for signal in pricing_signals):
        return False

    requested_locale = _page_locale_from_url(response.url)
    page_locale = _page_locale_from_html(tree)
    if requested_locale and page_locale:
        requested_country = requested_locale.split("-")[-1]
        page_country = page_locale.split("-")[-1]
        if requested_country != page_country:
            return False
    return True


async def discover_fee_pages(
    http_client: HttpClient,
    market: Market,
    config: CrawlConfiguration,
) -> tuple[str, str | None]:
    """Discover and validate the canonical pricing URLs for a market.

    Returns (pricing_url, payment_methods_url | None). Raises FeePageError
    or UnsupportedMarketError on failure. Transient failures and access
    challenges are never converted into unsupported-market records.
    """
    code = market.account_country
    pricing_url = _pricing_url_for(market)
    payment_methods_url = _payment_methods_url_for(market)
    tested: list[str] = []
    transient_failure = False
    pricing_response: HttpResponse | None = None
    pricing_tree: Any | None = None

    for url in [pricing_url, payment_methods_url]:
        if url is None:
            continue
        tested.append(url)
        try:
            response = await http_client.get(url, market=market.account_country, locale=market.locale)
        except (AccessChallengeError, ContentSecurityError) as exc:
            logger.debug("Access/security failure for %s: %s", code, exc)
            transient_failure = True
            continue
        except PermanentHttpError as exc:
            if exc.status_code == 404:
                continue
            logger.debug("Permanent HTTP error for %s: %s", code, exc)
            transient_failure = True
            continue
        except (PermanentNetworkError, TransientNetworkError, RateLimitError, NetworkError) as exc:
            logger.debug("Network failure for %s: %s", code, exc)
            transient_failure = True
            continue

        try:
            tree = html.fromstring(response.text)
        except Exception as exc:
            logger.debug("Could not parse HTML for %s: %s", code, exc)
            transient_failure = True
            continue

        if response.detected_market and response.detected_market.upper() != code.upper():
            logger.warning(
                "Requested %s but %s served market %s (effective %s); treating as transient",
                code,
                url,
                response.detected_market,
                response.url,
            )
            transient_failure = True
            continue

        if url == pricing_url and _is_pricing_page(response, tree):
            pricing_response = response
            pricing_tree = tree
        elif url == payment_methods_url and _is_pricing_page(response, tree):
            pass
        else:
            # The local payment methods URL may redirect back to the main page.
            if url == payment_methods_url and str(response.url).rstrip("/") == pricing_url.rstrip("/"):
                payment_methods_url = None

    if pricing_response is None or not _is_pricing_page(pricing_response, pricing_tree):
        if transient_failure:
            raise FeePageError(f"Could not confirm a pricing page for {code}; transient_failure={transient_failure}")
        raise UnsupportedMarketError(
            f"No public pricing page found for {code}",
            tested_urls=tested,
        )

    return pricing_url, payment_methods_url


def build_market_from_code(
    country_code: str,
    language: str = "en",
    country_name: str | None = None,
    status: str = "discovered",
) -> Market:
    """Build a Market object from an ISO country code and language."""
    country_code = country_code.upper()
    locale = f"{language}-{country_code.lower()}"
    if country_code.lower() in DIRECT_LOCALE_MARKETS:
        stripe_market_code = country_code.lower()
        url_prefix = f"https://stripe.com/{country_code.lower()}"
    else:
        stripe_market_code = f"{language}-{country_code.lower()}"
        url_prefix = f"https://stripe.com/{stripe_market_code}"
    return Market(
        stripe_market_code=stripe_market_code,
        account_country=country_code,
        country_name=country_name or country_code,
        region=None,
        locale=locale,
        languages=[Language(code=language, name=None)],
        url_prefix=url_prefix,
        preferred_language=language,
        default_currency=CURRENCY_BY_COUNTRY.get(country_code),
        status=status,
    )
