"""Tests for page extraction and source metadata."""

from __future__ import annotations

from stripe_fee_crawler.extract import extract_page_source, extract_pricing_entries
from stripe_fee_crawler.http import HttpResponse


def test_extract_page_source() -> None:
    html_text = "<!DOCTYPE html><html lang='en-us' data-page-id='pricing'><head><title>Pricing & Fees</title></head><body></body></html>"
    response = HttpResponse(
        url="https://stripe.com/pricing",
        status_code=200,
        content=html_text.encode(),
        text=html_text,
        headers={"etag": '"abc"', "last-modified": "Mon, 01 Jan 2024 00:00:00 GMT"},
    )
    source = extract_page_source(response)
    assert source.canonical_url == "https://stripe.com/pricing"
    assert source.page_title == "Pricing & Fees"
    assert source.page_id == "pricing"
    assert source.etag == '"abc"'


def test_extract_pricing_entries_parses_us(us_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(us_pricing_html, "https://stripe.com/pricing", page_kind="pricing")
    assert len(entries) > 0
    percentages = {e.source_text for e in entries if "2.9%" in e.source_text}
    assert percentages


def test_extract_pricing_entries_jp(jp_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(jp_pricing_html, "https://stripe.com/en-jp/pricing", page_kind="pricing")
    methods = {e.payment_method for e in entries if e.payment_method}
    assert "konbini" in methods or any("Konbini" in e.source_text for e in entries)
