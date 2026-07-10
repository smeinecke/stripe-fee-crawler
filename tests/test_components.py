"""Tests for section extraction."""

from __future__ import annotations

from stripe_fee_crawler.components import extract_sections, split_section_body_into_entries
from stripe_fee_crawler.extract import extract_pricing_entries


def test_extract_sections_main_pricing(de_pricing_html: str) -> None:
    sections = extract_sections(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    headings = [s.heading for s in sections if s.heading]
    assert "Payments" in headings
    assert "Terminal" in headings
    assert "Disputes" in headings


def test_extract_sections_local_payment_methods(de_lpm_html: str) -> None:
    sections = extract_sections(
        de_lpm_html, "https://stripe.com/en-de/pricing/local-payment-methods", page_kind="local-payment-methods"
    )
    headings = [s.heading for s in sections if s.heading]
    assert "Domestic card payments" in headings
    assert "Alipay" in headings
    assert "SEPA Direct Debit" in headings
    assert "Klarna" in headings


def test_extract_sections_us(us_pricing_html: str) -> None:
    sections = extract_sections(us_pricing_html, "https://stripe.com/pricing", page_kind="pricing")
    headings = [s.heading for s in sections if s.heading]
    assert "Payments" in headings


def test_split_section_body(de_pricing_html: str) -> None:
    sections = extract_sections(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    payments = next(s for s in sections if s.heading == "Payments")
    entries = split_section_body_into_entries(payments)
    assert any("1,5%" in phrase for phrase, _ in entries)


def test_extract_pricing_entries_count(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    assert len(entries) > 0
    assert any("card" in (e.payment_method or "") for e in entries)


def test_extract_pricing_entries_lpm(de_lpm_html: str) -> None:
    entries, _ = extract_pricing_entries(
        de_lpm_html, "https://stripe.com/en-de/pricing/local-payment-methods", page_kind="local-payment-methods"
    )
    methods = {e.payment_method for e in entries if e.payment_method}
    assert "alipay" in methods
    assert "sepa_direct_debit" in methods
    assert "klarna" in methods


def test_extract_pricing_entries_from_fixture(from_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(from_pricing_html, "https://stripe.com/pricing", page_kind="pricing")
    assert entries
    assert any("1.5" in e.source_text for e in entries)
