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


def _make_entry(source_text: str, section_path: list[str] | None = None, source_order: int = 0) -> PricingEntry:
    return PricingEntry(
        entry_id=f"e{source_order}",
        source_text=source_text,
        source_url="https://stripe.com/ae/pricing",
        section_path=section_path or ["Smart Disputes"],
        source_order=source_order,
    )


def test_no_calculable_30_percent_dispute_from_included_fragment() -> None:
    """The AE/AU 30% fragment must not become a calculable dispute fee."""
    entries = [
        _make_entry("30%", source_order=0),
        _make_entry("Included with Payments", source_order=1),
        _make_entry(
            "Included at no additional charge for businesses on standard payments pricing",
            source_order=2,
        ),
    ]
    rules, _ = classify_entries(entries, "AE")
    assert not any(
        r.classification_status == "calculable_rule" and r.product_id == "disputes" and r.percentage == "30"
        for r in rules
    )
    assert any(r.source_text == "30%" and r.classification_status != "calculable_rule" for r in rules)


def test_included_standard_tier_classified() -> None:
    """An included standard-pricing statement is classified as included."""
    entries = [
        _make_entry("Included", ["Authorization Boost"], source_order=0),
        _make_entry(
            "Included at no additional charge for businesses on standard payments pricing",
            ["Authorization Boost"],
            source_order=1,
        ),
    ]
    rules, _ = classify_entries(entries, "AE")
    assert rules
    assert rules[0].exactness == "included"
    assert rules[0].classification_status in {"included", "free"}


def test_paid_custom_pricing_tier_calculable() -> None:
    """A concrete fee scoped to custom pricing is calculable, not custom-quote."""
    entry = _make_entry(
        "0.2% per successful online card transaction for accounts with custom payments pricing",
        ["Authorization Boost"],
    )
    rules, _ = classify_entries([entry], "AE")
    calculable = [r for r in rules if r.classification_status == "calculable_rule"]
    assert calculable, "expected custom-pricing paid tier to be calculable"
    assert calculable[0].percentage == "0.2"
    assert calculable[0].exactness == "exact"


def test_no_calculable_from_adjacent_marketing_percentage() -> None:
    """Marketing percentages such as '30% of customers' are not fees."""
    entry = _make_entry("30% of customers choose Stripe for payments", ["Payments"])
    rules, _ = classify_entries([entry], "AE")
    assert not any(r.classification_status == "calculable_rule" for r in rules)


def test_legitimate_percentage_dispute_fee_with_explicit_fee_wording() -> None:
    """A real Smart Disputes fee with explicit wording is a calculable add-on."""
    entry = _make_entry(
        "Smart Disputes fee 30% of the disputed amount for each dispute you win.",
        ["Smart Disputes"],
    )
    rules, _ = classify_entries([entry], "AE")
    calculable = [r for r in rules if r.classification_status == "calculable_rule"]
    assert calculable
    rule = calculable[0]
    assert rule.product_id == "smart_disputes"
    assert rule.variant_id == "won_dispute"
    assert rule.percentage == "30"
    assert rule.unit == "per_dispute"
    assert any(c.dimension == "dispute_state" and c.value == "won" for c in rule.conditions)
    assert any(c.dimension == "feature_enabled" and c.value == "smart_disputes" for c in rule.conditions)


def test_three_d_secure_pricing_plans() -> None:
    """3D Secure has a standard included variant and a custom paid variant."""
    entries = [
        _make_entry(
            "Included at no additional charge for businesses on standard payments pricing",
            ["3D Secure authentication"],
            source_order=0,
        ),
        _make_entry(
            "$0.03 per 3D Secure attempt for accounts with custom pricing.",
            ["3D Secure authentication"],
            source_order=1,
        ),
    ]
    rules, _ = classify_entries(entries, "AE")
    custom = [r for r in rules if r.classification_status == "calculable_rule" and r.product_id == "three_d_secure"]
    assert len(custom) == 1
    assert custom[0].fixed_amount == "0.03"
    assert custom[0].variant_id == "custom_pricing"
    assert any(c.dimension == "pricing_plan" and c.value == "custom" for c in custom[0].conditions)

    included = [r for r in rules if r.product_id == "three_d_secure" and r.exactness == "included"]
    assert included
    assert included[0].variant_id == "standard_pricing"
    assert any(c.dimension == "pricing_plan" and c.value == "standard" for c in included[0].conditions)


def test_authorization_boost_pricing_plans() -> None:
    """Authorization Boost has a standard included variant and a custom paid variant."""
    entries = [
        _make_entry(
            "Included at no additional charge for businesses on standard payments pricing",
            ["Authorization Boost"],
            source_order=0,
        ),
        _make_entry(
            "0.2% per successful online card transaction for accounts with custom payments pricing",
            ["Authorization Boost"],
            source_order=1,
        ),
    ]
    rules, _ = classify_entries(entries, "AE")
    custom = [
        r for r in rules if r.classification_status == "calculable_rule" and r.product_id == "authorization_boost"
    ]
    assert len(custom) == 1
    assert custom[0].percentage == "0.2"
    assert custom[0].variant_id == "custom_pricing"
    assert any(c.dimension == "pricing_plan" and c.value == "custom" for c in custom[0].conditions)

    included = [r for r in rules if r.product_id == "authorization_boost" and r.exactness == "included"]
    assert included
    assert included[0].variant_id == "standard_pricing"


def test_radar_pricing_plans() -> None:
    """Radar emits distinct standard-pricing and custom-pricing calculable variants."""
    entries = [
        _make_entry(
            "AED0.20 per screened transaction for accounts with all payment methods on standard pricing",
            ["Radar"],
            source_order=0,
        ),
        _make_entry(
            "AED0.20 per screened transaction for accounts with custom pricing for any payment method",
            ["Radar"],
            source_order=1,
        ),
    ]
    rules, _ = classify_entries(entries, "AE")
    radar = [r for r in rules if r.classification_status == "calculable_rule" and r.product_id == "radar"]
    assert len(radar) == 2
    variants = {r.variant_id for r in radar}
    assert variants == {"standard_pricing", "custom_pricing"}
    for rule in radar:
        plan = next((c.value for c in rule.conditions if c.dimension == "pricing_plan"), None)
        assert plan
        assert rule.variant_id == f"{plan}_pricing"


def test_adaptive_pricing_starting_at_customer_paid() -> None:
    """Adaptive Pricing starting-at rates are from-exact, customer-paid conversion fees."""
    entry = _make_entry(
        "Customers will be presented a conversion fee starting at 2%",
        ["Adaptive Pricing"],
    )
    rules, _ = classify_entries([entry], "AE")
    calculable = [r for r in rules if r.classification_status == "calculable_rule"]
    assert calculable
    rule = calculable[0]
    assert rule.product_id == "adaptive_pricing"
    assert rule.exactness == "from"
    assert rule.percentage == "2"
    assert any(c.dimension == "payer" and c.value == "customer" for c in rule.conditions)
    assert any(c.dimension == "fee_type" and c.value == "conversion_fee" for c in rule.conditions)


def test_add_on_no_collision_with_base_payments() -> None:
    """Add-on fees keep a distinct identity from base card-processing rules."""
    entries = [
        _make_entry(
            "1.5% + €0.25 per transaction for standard EEA cards",
            ["Payments", "Online"],
            source_order=0,
        ),
        _make_entry(
            "0.2% per successful online card transaction for accounts with custom payments pricing",
            ["Authorization Boost"],
            source_order=1,
        ),
    ]
    rules, _ = classify_entries(entries, "DE")
    base = [r for r in rules if r.product_id == "payments" and r.classification_status == "calculable_rule"]
    addon = [r for r in rules if r.product_id == "authorization_boost" and r.classification_status == "calculable_rule"]
    assert base and addon
    assert base[0].rule_id != addon[0].rule_id
