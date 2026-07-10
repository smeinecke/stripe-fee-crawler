"""Tests for output validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from stripe_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from stripe_fee_crawler.models import Market, MarketOutput, Source
from stripe_fee_crawler.output import OutputPublisher
from stripe_fee_crawler.validation import (
    generate_core_fees_schema,
    generate_index_schema,
    generate_manifest_schema,
    generate_market_output_schema,
    generate_payment_methods_schema,
    validate_all_output,
    validate_core_fees,
    validate_manifest,
    validate_market_output,
    validate_payment_methods,
)


def test_validate_market_output_valid() -> None:
    market = Market(
        stripe_market_code="en-de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )
    validated = validate_market_output(output.model_dump())
    assert validated.market.account_country == "DE"


def test_validate_all_output(tmp_path: Path) -> None:
    market = Market(
        stripe_market_code="en-de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    result = validate_all_output(tmp_path)
    assert result["success"]


def test_validate_all_output_fails_on_invalid(tmp_path: Path) -> None:
    (tmp_path / "json").mkdir(parents=True)
    (tmp_path / "json" / "DE.json").write_text(
        '{"schema_version": 1, "market": {}, "sources": [], "entries": [], "derived_rules": [], "unclassified_entries": [], "warnings": [], "derivation_status": "invalid"}'
    )
    with pytest.raises(CrawlerValidationError):
        validate_all_output(tmp_path)


def test_validate_core_fees() -> None:
    data = {
        "schema_version": 1,
        "markets": [
            {
                "account_country": "DE",
                "stripe_market_code": "en-de",
                "locale": "en-de",
                "derivation_status": "complete",
                "rules": [],
                "unclassified_count": 0,
            }
        ],
    }
    validated = validate_core_fees(data)
    assert validated.markets[0].account_country == "DE"


def test_validate_payment_methods() -> None:
    data = {
        "schema_version": 1,
        "methods": [
            {
                "method_id": "card",
                "family": "card",
                "display_name": "Card",
                "supported_account_countries": ["DE"],
                "fee_rule_refs": ["r1"],
                "source_refs": ["https://stripe.com/pricing"],
            }
        ],
    }
    validated = validate_payment_methods(data)
    assert validated.methods[0].method_id == "card"


def test_validate_manifest() -> None:
    data = {
        "schema_version": 1,
        "markets": [],
        "unsupported": [],
        "aliases": {},
        "fee_page_urls": {},
        "transient_failures": [],
    }
    validated = validate_manifest(data)
    assert validated.markets == []


@pytest.mark.parametrize(
    ("generator", "schema_id"),
    [
        (generate_market_output_schema, "stripe-fees-v1.schema.json"),
        (generate_core_fees_schema, "core-fees-v1.schema.json"),
        (generate_payment_methods_schema, "payment-methods-v1.schema.json"),
        (generate_index_schema, "index-v1.schema.json"),
        (generate_manifest_schema, "manifest-v1.schema.json"),
    ],
)
def test_generated_schemas_have_id(generator, schema_id: str) -> None:
    schema = generator()
    assert "$id" in schema
    assert schema_id in schema["$id"]
