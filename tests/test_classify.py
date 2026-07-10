"""Tests for classification and derivation."""

from __future__ import annotations

from stripe_fee_crawler.classify import classify_entries, derive_market_fees
from stripe_fee_crawler.extract import extract_pricing_entries
from stripe_fee_crawler.models import PricingEntry


def test_classify_domestic_card(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, unclassified = classify_entries(entries)
    domestic = [r for r in rules if r.card_region == "eea" and r.card_tier == "standard"]
    assert domestic
    assert domestic[0].percentage == "1.5"
    assert domestic[0].fixed_amount == "0.25"
    assert domestic[0].fixed_currency == "EUR"


def test_classify_international_card(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    uk = [r for r in rules if r.card_region == "uk"]
    assert uk
    assert uk[0].percentage == "2.5"


def test_classify_conversion_surcharge(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    surcharge = [r for r in rules if r.currency_conversion_required and r.percentage == "2"]
    assert surcharge


def test_classify_terminal(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    terminal = [r for r in rules if r.channel == "in_person"]
    assert terminal


def test_classify_dispute(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    dispute = [r for r in rules if r.unit == "per_dispute"]
    assert dispute
    assert dispute[0].fixed_amount == "20.00"


def test_classify_lpm_sepa(de_lpm_html: str) -> None:
    entries, _ = extract_pricing_entries(
        de_lpm_html, "https://stripe.com/en-de/pricing/local-payment-methods", page_kind="local-payment-methods"
    )
    rules, _ = classify_entries(entries)
    sepa = [r for r in rules if r.payment_method == "sepa_direct_debit"]
    assert sepa
    assert sepa[0].fixed_amount == "0.35"


def test_classify_from_pricing() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="Starting at 1.5% + $0.25 for high-volume merchants",
        source_url="https://stripe.com/pricing",
        section_path=["Custom pricing"],
    )
    rules, unclassified = classify_entries([entry])
    assert rules
    assert rules[0].exactness == "from"


def test_classify_custom_only() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="Contact sales for a custom quote",
        source_url="https://stripe.com/pricing",
        section_path=["Custom pricing"],
    )
    rules, unclassified = classify_entries([entry])
    assert not rules
    assert unclassified


def test_classify_free_entry() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="Included at no additional charge",
        source_url="https://stripe.com/pricing",
        section_path=["Payments"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].exactness == "included"


def test_derive_market_fees_status(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, unclassified, status = derive_market_fees(entries)
    assert status == "partial" or status == "complete"
    assert rules


def test_non_eea_classified_as_international() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="1.5% + €0.25 for non-EEA cards",
        source_url="https://stripe.com/pricing",
        section_path=["Payments", "Online"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].card_region == "international"


def test_unknown_channel_is_non_calculable() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="1.5% + €0.25 per transaction for card payments",
        source_url="https://stripe.com/pricing",
        section_path=["Payments"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].classification_status == "non_calculable"
    assert rules[0].confidence == 0.0


def test_unknown_unit_is_non_calculable() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="€10 for card payments",
        source_url="https://stripe.com/pricing",
        section_path=["Payments", "Online"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].classification_status == "non_calculable"


def test_jpy_zero_exponent() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="3.6% + ¥0 per transaction for card payments",
        source_url="https://stripe.com/pricing",
        section_path=["Payments", "Online"],
    )
    rules, _ = classify_entries([entry])
    jpy = [r for r in rules if r.classification_status == "classified"]
    assert jpy
    assert jpy[0].fixed_amount_minor == "0"


def test_fixed_amount_minor_uses_iso_exponents() -> None:
    from stripe_fee_crawler.classify import _fixed_amount_minor

    assert _fixed_amount_minor("1234", "JPY") == "1234"
    assert _fixed_amount_minor("0.250", "BHD") == "250"
    assert _fixed_amount_minor("0.25", "EUR") == "25"
    assert _fixed_amount_minor("invalid", "USD") is None
