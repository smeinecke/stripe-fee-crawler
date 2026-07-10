"""Tests for market discovery and fee-page validation."""

from __future__ import annotations

import pytest
from lxml import html

from stripe_fee_crawler.discovery import (
    _extract_footer_markets,
    _is_pricing_page,
    _page_locale_from_url,
    _payment_methods_url_for,
    _pricing_url_for,
    build_market_from_code,
    discover_fee_pages,
    discover_markets,
    get_bootstrap_markets,
)
from stripe_fee_crawler.http import HttpClient, HttpResponse
from stripe_fee_crawler.models import CrawlConfiguration


def test_get_bootstrap_markets() -> None:
    markets = get_bootstrap_markets()
    codes = {m.account_country for m in markets}
    assert "DE" in codes
    assert "US" in codes


def test_build_market_from_code() -> None:
    market = build_market_from_code("DE")
    assert market.account_country == "DE"
    assert market.locale == "en-de"
    assert market.url_prefix == "https://stripe.com/en-de"


def test_build_market_from_code_direct_locale() -> None:
    market = build_market_from_code("US")
    assert market.locale == "en-us"
    assert market.url_prefix == "https://stripe.com/us"


def test_page_locale_from_url() -> None:
    assert _page_locale_from_url("https://stripe.com/en-de/pricing") == "en-de"
    assert _page_locale_from_url("https://stripe.com/pricing") == "us"
    assert _page_locale_from_url("https://stripe.com/gb/pricing") == "gb"


def test_is_pricing_page_valid() -> None:
    html_text = "<html lang='de-DE'><head><title>Preise und Gebühren | Stripe</title></head><body><p>Standard pricing</p><p>Domestic card payments</p></body></html>"
    response = HttpResponse(
        url="https://stripe.com/en-de/pricing",
        status_code=200,
        content=html_text.encode(),
        text=html_text,
        headers={"content-type": "text/html"},
    )
    tree = html.fromstring(html_text)
    assert _is_pricing_page(response, tree)


def test_is_pricing_page_invalid_locale() -> None:
    html_text = "<html lang='en-us'><head><title>Pricing</title></head><body><p>Standard pricing</p><p>Domestic card payments</p></body></html>"
    response = HttpResponse(
        url="https://stripe.com/en-de/pricing",
        status_code=200,
        content=html_text.encode(),
        text=html_text,
        headers={"content-type": "text/html"},
    )
    tree = html.fromstring(html_text)
    assert not _is_pricing_page(response, tree)


def test_extract_footer_markets() -> None:
    html_text = """
    <html><body><footer>
      <a href="/en-de">Germany</a>
      <a href="/en-gb">United Kingdom</a>
      <a href="/pricing">United States</a>
    </footer></body></html>
    """
    tree = html.fromstring(html_text)
    markets = _extract_footer_markets(tree)
    codes = {m.account_country for m in markets}
    assert "DE" in codes
    assert "GB" in codes
    assert "US" in codes


def test_pricing_url_for() -> None:
    market = build_market_from_code("DE")
    assert _pricing_url_for(market) == "https://stripe.com/en-de/pricing"


def test_payment_methods_url_for() -> None:
    market = build_market_from_code("US")
    assert _payment_methods_url_for(market) == "https://stripe.com/pricing/local-payment-methods"
    market_de = build_market_from_code("DE")
    assert _payment_methods_url_for(market_de) == "https://stripe.com/en-de/pricing/local-payment-methods"


@pytest.mark.asyncio
async def test_discover_fee_pages_with_fixture() -> None:
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        offline_fixtures={
            "https://stripe.com/en-de/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-de/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
    )
    market = build_market_from_code("DE", status="supported")
    async with HttpClient(config) as client:
        pricing_url, payment_methods_url = await discover_fee_pages(client, market, config)
    assert pricing_url == "https://stripe.com/en-de/pricing"
    assert payment_methods_url == "https://stripe.com/en-de/pricing/local-payment-methods"


@pytest.mark.asyncio
async def test_discover_markets_with_fixture() -> None:
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        offline_fixtures={
            "https://stripe.com/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
    )
    async with HttpClient(config) as client:
        markets = await discover_markets(client, config)
    codes = {m.account_country for m in markets}
    assert "DE" in codes or "US" in codes
