"""Tests for pricing token parsing."""

from __future__ import annotations

from decimal import Decimal

from stripe_fee_crawler.pricing_tokens import (
    _detect_decimal_separator,
    _parse_decimal,
    parse_fee_value,
    tokenize_fee_text,
)


def test_detect_decimal_separator_comma() -> None:
    assert _detect_decimal_separator("1,5% + 0,25 €") == ","


def test_detect_decimal_separator_dot() -> None:
    assert _detect_decimal_separator("2.9% + $0.30") == "."


def test_parse_decimal_comma() -> None:
    assert _parse_decimal("1,5") == Decimal("1.5")


def test_parse_decimal_thousands_dot() -> None:
    assert _parse_decimal("1.000,50") == Decimal("1000.50")


def test_parse_decimal_thousands_comma() -> None:
    assert _parse_decimal("1,000.50") == Decimal("1000.50")


def test_tokenize_percentage_and_amount() -> None:
    tokens = tokenize_fee_text("1.5% + $0.25")
    assert any(t.kind == "percentage" and t.percentage == "1.5" for t in tokens)
    assert any(t.kind == "amount" and t.amount == "0.25" and t.currency == "USD" for t in tokens)


def test_tokenize_euro_comma() -> None:
    tokens = tokenize_fee_text("1,5% + 0,25 €")
    assert any(t.kind == "percentage" and t.percentage == "1.5" for t in tokens)
    assert any(t.kind == "amount" and t.amount == "0.25" and t.currency == "EUR" for t in tokens)


def test_tokenize_free() -> None:
    tokens = tokenize_fee_text("Included at no additional charge")
    assert any(t.exactness == "included" for t in tokens)


def test_parse_fee_value_fixed_plus_percentage() -> None:
    parsed = parse_fee_value("2.9% + $0.30")
    assert parsed["percentage"] == "2.9"
    assert parsed["fixed_amount"] == "0.30"
    assert parsed["fixed_currency"] == "USD"


def test_parse_fee_value_from() -> None:
    parsed = parse_fee_value("Starting at 2.99% + $0.35")
    assert parsed["exactness"] == "from"


def test_parse_fee_value_custom() -> None:
    parsed = parse_fee_value("Contact sales for a custom quote")
    assert parsed["exactness"] == "custom"


def test_parse_fee_value_conversion_surcharge() -> None:
    parsed = parse_fee_value("+ 2% if currency conversion is required")
    assert parsed["percentage"] == "2"
    assert parsed["exactness"] == "exact"


def test_parse_fee_value_dispute() -> None:
    parsed = parse_fee_value("€20.00 for each dispute you receive")
    assert parsed["fixed_amount"] == "20.00"
    assert parsed["fixed_currency"] == "EUR"
