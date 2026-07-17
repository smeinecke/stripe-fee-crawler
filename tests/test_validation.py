"""Tests for output validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from stripe_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from stripe_fee_crawler.models import (
    CoreFeeEntry,
    CoreFeeRule,
    CoreFees,
    FeeComponent,
    FeeCondition,
    FeeEvidence,
    FeeRule,
    Market,
    MarketManifest,
    MarketOutput,
    PaymentMethodCatalog,
    PaymentMethodEntry,
    Source,
)
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
    validate_semantic,
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


def _valid_core_rule(**overrides: Any) -> CoreFeeRule:

    return CoreFeeRule(
        rule_id="r1",
        product_id="payments",
        variant_id="online_domestic_cards",
        label="1.5% + €0.25",
        provider="stripe",
        account_country="DE",
        channel="online",
        payment_method="card",
        conditions=[FeeCondition(dimension="card_origin", value="domestic")],
        fee_components=[
            FeeComponent(type="percentage", value="1.5", basis_points="150"),
            FeeComponent(type="fixed_amount", amount="0.25", currency="EUR", minor_amount="25"),
        ],
        unit="per_transaction",
        exactness="exact",
        behavior="conditional",
        classification_status="calculable_rule",
        confidence=0.85,
        fee_evidence=FeeEvidence(type="explicit_fee_phrase", confidence=0.85),
        **overrides,
    )


def _valid_rule(**overrides: Any) -> FeeRule:
    return FeeRule(
        rule_id="r1",
        entry_id="e1",
        name="card_payment",
        provider="stripe",
        channel="online",
        payment_method="card",
        percentage="1.5",
        basis_points="150",
        fixed_amount="0.25",
        fixed_amount_minor="25",
        fixed_currency="EUR",
        unit="per_transaction",
        exactness="exact",
        behavior="conditional",
        source_text="1.5% + €0.25",
        source_url="https://stripe.com/pricing",
        classification_status="classified",
        confidence=0.85,
        **overrides,
    )


def test_semantic_validation_passes() -> None:
    market = Market(
        stripe_market_code="de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    rule = _valid_core_rule()
    core_fees = CoreFees(
        markets=[
            CoreFeeEntry(
                account_country="DE",
                stripe_market_code="de",
                locale="en-de",
                derivation_status="complete",
                rules=[rule],
            )
        ]
    )
    manifest = MarketManifest(markets=[market])
    payment_methods = PaymentMethodCatalog(
        methods=[PaymentMethodEntry(method_id="card", family="card", display_name="Card")]
    )
    result = validate_semantic("/unused", core_fees=core_fees, manifest=manifest, payment_methods=payment_methods)
    assert result["success"]


def test_semantic_validation_fails_bad_currency_exponent() -> None:

    market = Market(
        stripe_market_code="de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    # JPY has exponent 0, so 1.0 JPY should be minor=1, not 100.
    rule = CoreFeeRule(
        rule_id="r1",
        product_id="payments",
        variant_id="online_domestic_cards",
        label="1.5% + ¥1",
        provider="stripe",
        account_country="DE",
        channel="online",
        payment_method="card",
        fee_components=[
            FeeComponent(type="percentage", value="1.5", basis_points="150"),
            FeeComponent(type="fixed_amount", amount="1.0", currency="JPY", minor_amount="100"),
        ],
        unit="per_transaction",
        exactness="exact",
        behavior="conditional",
        classification_status="calculable_rule",
        confidence=0.85,
        fee_evidence=FeeEvidence(type="explicit_fee_phrase", confidence=0.85),
    )
    core_fees = CoreFees(
        markets=[
            CoreFeeEntry(
                account_country="DE",
                stripe_market_code="de",
                locale="en-de",
                derivation_status="complete",
                rules=[rule],
            )
        ]
    )
    manifest = MarketManifest(markets=[market])
    payment_methods = PaymentMethodCatalog(
        methods=[PaymentMethodEntry(method_id="card", family="card", display_name="Card")]
    )
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic("/unused", core_fees=core_fees, manifest=manifest, payment_methods=payment_methods)
    assert "exponent" in str(excinfo.value).lower()


def test_semantic_validation_fails_missing_market() -> None:
    rule = _valid_core_rule()
    core_fees = CoreFees(
        markets=[
            CoreFeeEntry(
                account_country="XX",
                stripe_market_code="xx",
                locale="en-xx",
                derivation_status="complete",
                rules=[rule],
            )
        ]
    )
    manifest = MarketManifest(markets=[])
    payment_methods = PaymentMethodCatalog(
        methods=[PaymentMethodEntry(method_id="card", family="card", display_name="Card")]
    )
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic("/unused", core_fees=core_fees, manifest=manifest, payment_methods=payment_methods)
    assert "manifest" in str(excinfo.value).lower()


def test_semantic_validation_fails_on_contradictory_fee_evidence() -> None:
    """A calculable rule must not mix positive-fee and included/free evidence."""
    rule = _valid_core_rule().model_copy(
        update={
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=[
                    "30%",
                    "Included with Payments",
                    "Included at no additional charge for businesses on standard payments pricing",
                ],
                confidence=0.85,
            )
        }
    )
    core_fees = CoreFees(
        markets=[
            CoreFeeEntry(
                account_country="AE",
                stripe_market_code="ae",
                locale="en-ae",
                derivation_status="complete",
                rules=[rule],
            )
        ]
    )
    manifest = MarketManifest(
        markets=[
            Market(
                stripe_market_code="ae",
                account_country="AE",
                country_name="United Arab Emirates",
                locale="en-ae",
                url_prefix="https://stripe.com/ae",
                status="supported",
            )
        ]
    )
    payment_methods = PaymentMethodCatalog(
        methods=[PaymentMethodEntry(method_id="card", family="card", display_name="Card")]
    )
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic("/unused", core_fees=core_fees, manifest=manifest, payment_methods=payment_methods)
    assert "included/free evidence" in str(excinfo.value).lower()
