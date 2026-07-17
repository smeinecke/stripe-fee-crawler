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
    rules, unclassified, status, *_ = derive_market_fees(entries)
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
        source_text="1.5% + €0.25 per transaction",
        source_url="https://stripe.com/pricing",
        section_path=["Other"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].classification_status in {"non_calculable", "unsupported_fee_shape"}
    assert rules[0].confidence == 0.0


def test_unknown_unit_is_non_calculable() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="€10",
        source_url="https://stripe.com/pricing",
        section_path=["Other"],
    )
    rules, _ = classify_entries([entry])
    assert rules
    assert rules[0].classification_status in {"non_calculable", "unsupported_fee_shape"}


def test_jpy_zero_exponent() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="3.6% + ¥0 per transaction for card payments",
        source_url="https://stripe.com/pricing",
        section_path=["Payments", "Online"],
    )
    rules, _ = classify_entries([entry])
    jpy = [r for r in rules if r.classification_status == "calculable_rule"]
    assert jpy
    assert jpy[0].fixed_amount_minor == "0"


def test_fixed_amount_minor_uses_iso_exponents() -> None:
    from stripe_fee_crawler.classify import _fixed_amount_minor

    assert _fixed_amount_minor("1234", "JPY") == "1234"
    assert _fixed_amount_minor("0.250", "BHD") == "250"
    assert _fixed_amount_minor("0.25", "EUR") == "25"
    assert _fixed_amount_minor("invalid", "USD") is None


def test_us_online_domestic_card_rate(us_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(us_pricing_html, "https://stripe.com/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    domestic = [
        r
        for r in rules
        if r.classification_status == "calculable_rule"
        and r.product_id == "payments"
        and r.channel == "online"
        and r.payment_method == "card"
        and r.card_origin == "domestic"
    ]
    assert domestic, "expected US online domestic card rule"
    assert domestic[0].percentage == "2.9"
    assert domestic[0].fixed_amount == "0.30"
    assert domestic[0].fixed_currency == "USD"


def test_us_terminal_domestic_rate(us_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(us_pricing_html, "https://stripe.com/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    terminal = [
        r
        for r in rules
        if r.classification_status == "calculable_rule"
        and r.product_id == "terminal"
        and r.channel == "in_person"
        and r.payment_method == "card"
        and r.card_origin == "domestic"
    ]
    assert terminal, "expected US terminal domestic card rule"
    assert terminal[0].percentage == "2.7"
    assert terminal[0].fixed_amount == "0.05"
    assert terminal[0].fixed_currency == "USD"


def test_us_ach_direct_debit_rate_with_cap() -> None:
    entry = PricingEntry(
        entry_id="test",
        source_text="ACH Direct Debit 0.8% per transaction with a $5 cap",
        source_url="https://stripe.com/us/pricing",
        section_path=["Payment methods"],
        payment_method="ach_direct_debit",
    )
    rules, _ = classify_entries([entry], "US")
    ach = [r for r in rules if r.classification_status == "calculable_rule" and r.product_id == "ach_direct_debit"]
    assert ach, "expected ACH Direct Debit calculable rule"
    assert ach[0].percentage == "0.8"
    assert any(c.type == "maximum_fee" and c.amount == "5" and c.currency == "USD" for c in ach[0].fee_components)


def test_de_standard_eea_card_rate(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    standard_eea = [
        r
        for r in rules
        if r.classification_status == "calculable_rule"
        and r.product_id == "payments"
        and r.card_region == "eea"
        and r.card_tier == "standard"
    ]
    assert standard_eea, "expected DE standard EEA card rule"
    assert standard_eea[0].percentage == "1.5"
    assert standard_eea[0].fixed_amount == "0.25"
    assert standard_eea[0].fixed_currency == "EUR"


def test_de_international_card_rate(de_pricing_html: str) -> None:
    entries, _ = extract_pricing_entries(de_pricing_html, "https://stripe.com/en-de/pricing", page_kind="pricing")
    rules, _ = classify_entries(entries)
    international = [
        r
        for r in rules
        if r.classification_status == "calculable_rule"
        and r.product_id == "payments"
        and r.card_region == "international"
    ]
    assert international, "expected DE international card rule"
    assert international[0].percentage == "3.25"
    assert international[0].fixed_amount == "0.25"
    assert international[0].fixed_currency == "EUR"


# Negative regression fixtures for known false positives.


def test_no_calculable_from_billion_marketing_claim() -> None:
    entry = PricingEntry(
        entry_id="billion",
        source_text="100+ category leaders each process more than $1 billion per year on Stripe.",
        source_url="https://stripe.com/pricing",
        section_path=["Payments"],
    )
    rules, _ = classify_entries([entry])
    assert not any(r.classification_status == "calculable_rule" for r in rules)


def test_no_calculable_from_alphanumeric_method_name() -> None:
    entry = PricingEntry(
        entry_id="p24",
        source_text="from the P24 portal",
        source_url="https://stripe.com/pricing",
        section_path=["Payment methods", "Przelewy24"],
    )
    rules, _ = classify_entries([entry])
    assert not any(r.classification_status == "calculable_rule" for r in rules)


def test_no_calculable_from_terminal_hardware_price() -> None:
    entry = PricingEntry(
        entry_id="terminal-price",
        source_text="A$89.00",
        source_url="https://stripe.com/pricing",
        section_path=["Terminal", "Reader"],
    )
    rules, _ = classify_entries([entry])
    assert not any(r.classification_status == "calculable_rule" for r in rules)


def test_no_calculable_from_promotional_conditional_rate() -> None:
    entry = PricingEntry(
        entry_id="klarna-promo",
        source_text=(
            "Certain businesses may qualify for temporarily reduced rates on selected local payment methods, "
            "including Klarna, that Stripe enables for you. These payment methods will be available for "
            "1.7% + A$0.30 per transaction for at least 2 months."
        ),
        source_url="https://stripe.com/pricing",
        section_path=["Payment methods", "Klarna"],
    )
    rules, _ = classify_entries([entry])
    assert not any(r.classification_status == "calculable_rule" for r in rules)


def test_calculable_when_explicit_fee_phrase_present() -> None:
    entry = PricingEntry(
        entry_id="clear-fee",
        source_text="2.2% + 20p per transaction for Przelewy24",
        source_url="https://stripe.com/pricing",
        section_path=["Payment methods", "Przelewy24"],
    )
    rules, _ = classify_entries([entry])
    assert any(r.classification_status == "calculable_rule" for r in rules)
