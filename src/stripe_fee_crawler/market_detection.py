"""Detect the Stripe market/locale/currency actually served by a page."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from lxml import html

from .currencies import _CURRENCY_PATTERNS, CURRENCY_BY_COUNTRY

# Country codes for which Stripe uses a direct (no language prefix) canonical URL.
DIRECT_LOCALE_MARKETS = {"us", "gb", "au", "nz", "ie", "in", "ae"}

_PRICING_PAGE_SIGNALS = [
    "Standard pricing",
    "Custom pricing",
    "Pricing & Fees",
    "Preise",
    "Preise und Gebühren",
    "Local payment methods",
    "Domestic card",
    "International card",
    "per transaction",
    "per successful",
]


def _has_pricing_structure(text: str) -> bool:
    """Return True when ``text`` contains recognizable Stripe pricing signals."""
    lower = text.lower()
    return any(signal.lower() in lower for signal in _PRICING_PAGE_SIGNALS)


COUNTRY_NAME_TO_ISO: dict[str, str] = {
    "australia": "AU",
    "austria": "AT",
    "belgium": "BE",
    "brazil": "BR",
    "bulgaria": "BG",
    "canada": "CA",
    "croatia": "HR",
    "cyprus": "CY",
    "czech republic": "CZ",
    "denmark": "DK",
    "estonia": "EE",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "gibraltar": "GI",
    "greece": "GR",
    "hong kong": "HK",
    "hungary": "HU",
    "india": "IN",
    "indonesia": "ID",
    "ireland": "IE",
    "italy": "IT",
    "japan": "JP",
    "latvia": "LV",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "malaysia": "MY",
    "malta": "MT",
    "mexico": "MX",
    "netherlands": "NL",
    "new zealand": "NZ",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "singapore": "SG",
    "slovakia": "SK",
    "slovenia": "SI",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "thailand": "TH",
    "united arab emirates": "AE",
    "united kingdom": "GB",
    "united states": "US",
}


# Map HTML lang values (e.g. "en-US" or "de-DE") to ISO-3166-1 alpha-2 country.
# The country code is the part after the final hyphen.  Some pages use the
# short form "en" for the US/UK English default, which we treat as ambiguous.
def _country_from_locale(locale: str | None) -> str | None:
    if not locale:
        return None
    locale = locale.strip().lower()
    if locale in {"en", "de"}:
        return None
    if "-" in locale:
        return locale.split("-")[-1].upper() or None
    return locale.upper() if len(locale) == 2 and locale.isalpha() else None


def _detect_currency(text: str) -> str | None:
    """Return the first currency code found in a pricing phrase."""
    for pattern, currency in _CURRENCY_PATTERNS:
        if re.search(pattern, text):
            return currency
    return None


def _detect_market_from_path(path: str) -> str | None:
    """Infer the country code from the URL path.

    Examples:
      /en-de/pricing       -> DE
      /gb/pricing          -> GB
      /pricing             -> None (the generic root has no market marker)
      /de/pricing          -> DE
    """
    path = path.lower().rstrip("/")
    if path in {"", "/pricing", "/pricing/local-payment-methods"}:
        return None
    # Strip the trailing /pricing or /pricing/local-payment-methods segment.
    path = re.sub(r"/pricing(/local-payment-methods)?$", "", path)
    if not path:
        return None
    # Path now looks like /en-de, /de, /gb, /in, etc.
    segment = path.lstrip("/").lower()
    if re.fullmatch(r"[a-z]{2}-[a-z]{2}", segment):
        return segment.split("-")[-1].upper()
    if re.fullmatch(r"[a-z]{2}", segment):
        return segment.upper()
    return None


def detect_market(
    html_text: str,
    effective_url: str | None = None,
    page_title: str | None = None,
) -> dict[str, Any]:
    """Detect the market, locale and currency actually served by a Stripe page.

    Returns a dict with detected_market (ISO-3166-1 alpha-2 upper), detected_locale
    (IETF language tag as seen on the page) and detected_currency (ISO-4217).
    """
    try:
        tree = html.fromstring(html_text)
    except Exception:
        tree = None

    detected_locale: str | None = None
    detected_market: str | None = None
    detected_currency: str | None = None

    # 1. <html lang> is the strongest signal on Stripe pages.
    if tree is not None:
        html_tag = tree.xpath("//html")
        if html_tag:
            detected_locale = html_tag[0].get("lang") or html_tag[0].get("xml:lang")
            detected_market = _country_from_locale(detected_locale)

    # 2. Canonical URL path locale.
    if detected_market is None and tree is not None:
        canonical = tree.xpath("//link[@rel='canonical']/@href")
        if canonical:
            detected_market = _detect_market_from_path(urlparse(canonical[0]).path)
        if detected_market is None and effective_url:
            detected_market = _detect_market_from_path(urlparse(effective_url).path)

    # 3. Page title / heading country names.
    title_text: str | None = page_title
    if detected_market is None and title_text is None and tree is not None:
        title_node = tree.find(".//title")
        if title_node is not None:
            title_text = title_node.text
    if detected_market is None and title_text:
        title_lower = title_text.lower()
        for name, iso in COUNTRY_NAME_TO_ISO.items():
            if name in title_lower:
                detected_market = iso
                break

    # 4. Currency of the first explicit card-fee phrase.
    if detected_market is None and tree is not None:
        text = " ".join(tree.itertext())
        # Look for a fee phrase with a percentage and an amount.
        for match in re.finditer(
            r"\d(?:\.\d)?%\s*(?:\+\s*)?(?:[^\d\s]?\s*[\d.,]+\s*[¢₹₽¥]|[^\d\s]?\s*[A-Z]?\$?\s*[\d.,]+)",
            text,
        ):
            detected_currency = _detect_currency(match.group(0))
            if detected_currency:
                # Infer country from currency only when unambiguous.
                countries = [c for c, cur in CURRENCY_BY_COUNTRY.items() if cur == detected_currency]
                if len(countries) == 1:
                    detected_market = countries[0]
                break

    # Cross-check detected currency against the market's expected currency when possible.
    if detected_market and CURRENCY_BY_COUNTRY.get(detected_market):
        expected_currency = CURRENCY_BY_COUNTRY[detected_market]
        if detected_currency is None:
            detected_currency = expected_currency
        elif detected_currency != expected_currency:
            # Currency mismatch is a useful conflict signal, but we still prefer
            # the HTML-lang/URL-derived market.  Callers can flag the conflict.
            pass

    if detected_locale:
        detected_locale = detected_locale.lower()

    return {
        "detected_market": detected_market,
        "detected_locale": detected_locale,
        "detected_currency": detected_currency,
    }
