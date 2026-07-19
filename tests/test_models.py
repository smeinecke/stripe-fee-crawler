"""Tests for Pydantic models."""

from __future__ import annotations

import pytest

from stripe_fee_crawler.models import (
    REGRESSION_KINDS,
    ChangeReport,
    FeeRule,
    Market,
    PricingEntry,
)


def test_market_validation() -> None:
    market = Market(
        stripe_market_code="en-de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
    )
    assert market.account_country == "DE"
    assert market.url_slug == "en-de"


def test_market_invalid_country() -> None:
    with pytest.raises(ValueError):
        Market(
            stripe_market_code="en-de",
            account_country="DEU",
            country_name="Germany",
            locale="en-de",
            url_prefix="https://stripe.com/en-de",
        )


def test_pricing_entry_status() -> None:
    with pytest.raises(ValueError):
        PricingEntry(
            entry_id="x",
            source_text="test",
            source_url="https://stripe.com/pricing",
            classification_status="invalid",
        )


def test_fee_rule_exactness() -> None:
    with pytest.raises(ValueError):
        FeeRule(
            rule_id="r1",
            exactness="unknown",
        )


def test_change_report_has_regression() -> None:
    report = ChangeReport(changes=[{"kind": "removed_market", "country_code": "US", "message": "test"}])
    assert report.has_regression


def test_change_report_no_regression() -> None:
    report = ChangeReport(changes=[{"kind": "new_market", "country_code": "US", "message": "test"}])
    assert not report.has_regression


def test_regression_kinds_pinned() -> None:
    """The regression-kind set must not drift from the ChangeReport validator."""
    assert isinstance(REGRESSION_KINDS, frozenset)
    assert "removed_market" in REGRESSION_KINDS
    assert "fee_component_disappeared" in REGRESSION_KINDS
    assert "market_coverage_changed" in REGRESSION_KINDS
    assert "new_market" not in REGRESSION_KINDS
