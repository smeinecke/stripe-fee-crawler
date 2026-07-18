"""Tests for output validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from stripe_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from stripe_fee_crawler.models import (
    ChangeReport,
    CoreFeeEntry,
    CoreFeeRule,
    CoreFees,
    CoverageSummary,
    CrawlReport,
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
    UnsupportedMarket,
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
    validate_data_repository,
    validate_manifest,
    validate_market_output,
    validate_payment_methods,
    validate_semantic,
)


def _write_minimal_publication_files(
    data_dir: Path,
    *,
    markets: int = 1,
    unsupported: int = 0,
    transient: int = 0,
    rules: int = 0,
    regions: int = 0,
) -> None:
    """Create the reports and README that strict validation now requires."""
    (data_dir / "meta").mkdir(parents=True, exist_ok=True)
    (data_dir / "change-report.json").write_text(json.dumps(ChangeReport().model_dump()), encoding="utf-8")
    (data_dir / "meta" / "crawl-report.json").write_text(json.dumps(CrawlReport().model_dump()), encoding="utf-8")
    (data_dir / "README.md").write_text(
        f"""<!-- STATS_START -->
| Metric | Value |
|--------|------:|
| Markets | **{markets}** |
| Derivation status | — |
| Core fee rules | **{rules}** |
| Payment methods | 0 (—) |
| Regions | {regions} (—) |
| Unsupported markets | {unsupported} |
| Transient failures | {transient} |
| Last crawled | — |
<!-- STATS_END -->
""",
        encoding="utf-8",
    )


def _valid_complete_output(country: str = "DE") -> MarketOutput:
    market = Market(
        stripe_market_code=f"en-{country.lower()}",
        account_country=country,
        country_name=f"Country {country}",
        locale=f"en-{country.lower()}",
        url_prefix=f"https://stripe.com/{country.lower()}",
        status="supported",
    )
    return MarketOutput(
        market=market,
        sources=[Source(requested_url=f"https://stripe.com/{country.lower()}/pricing")],
        derivation_status="complete",
        calculator_coverage_status="complete",
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
    _write_minimal_publication_files(tmp_path, rules=0)
    result = validate_all_output(tmp_path)
    assert result["success"]


def test_strict_validation_fails_on_blocking_fee_conflicts(tmp_path: Path) -> None:
    """Strict validation rejects any market whose coverage summary still reports blocking fee conflicts."""
    market = Market(
        stripe_market_code="en-us",
        account_country="US",
        country_name="United States",
        locale="en-us",
        url_prefix="https://stripe.com",
        status="supported",
    )
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/pricing")],
        derivation_status="partial",
        calculator_coverage_status="partial",
        coverage_summary=CoverageSummary(
            source_entries=10,
            blocking_fee_conflicts=1,
        ),
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "blocking fee conflict" in str(exc_info.value).lower()


def test_validate_all_output_fails_on_invalid(tmp_path: Path) -> None:
    (tmp_path / "json").mkdir(parents=True)
    (tmp_path / "json" / "DE.json").write_text(
        '{"schema_version": 1, "market": {}, "sources": [], "entries": [], "derived_rules": [], "unclassified_entries": [], "warnings": [], "derivation_status": "invalid"}'
    )
    _write_minimal_publication_files(tmp_path, markets=0, rules=0)
    with pytest.raises(CrawlerValidationError):
        validate_all_output(tmp_path, strict=True)


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

    defaults: dict[str, Any] = {
        "payment_method": None,
        "conditions": [],
    }
    defaults.update(overrides)
    return CoreFeeRule(
        rule_id="r1",
        product_id="payments",
        variant_id="online_domestic_cards",
        label="1.5% + €0.25",
        provider="stripe",
        account_country="DE",
        channel="online",
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
        **defaults,
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
    rule = _valid_core_rule(payment_method="card", conditions=[FeeCondition(dimension="card_origin", value="domestic")])
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
    rule = _valid_core_rule(payment_method="card")
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
            "payment_method": "card",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=[
                    "30%",
                    "Included with Payments",
                    "Included at no additional charge for businesses on standard payments pricing",
                ],
                confidence=0.85,
            ),
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


def _core_fees_with_rule(rule: CoreFeeRule) -> CoreFees:
    return CoreFees(
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


def _market_manifest_for_ae() -> MarketManifest:
    return MarketManifest(
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


def _payment_methods() -> PaymentMethodCatalog:
    return PaymentMethodCatalog(methods=[PaymentMethodEntry(method_id="card", family="card", display_name="Card")])


def test_semantic_validation_fails_missing_custom_pricing_plan() -> None:
    """A calculable custom-pricing fee must carry pricing_plan=custom."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "authorization_boost",
            "variant_id": "custom_pricing",
            "label": "0.2% per transaction for accounts with custom payments pricing",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["0.2% per transaction for accounts with custom payments pricing"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="0.2", basis_points="20")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "pricing_plan=custom" in str(excinfo.value).lower()


def test_semantic_validation_fails_missing_standard_pricing_plan() -> None:
    """A calculable standard-pricing fee must carry pricing_plan=standard."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "radar",
            "variant_id": "standard_pricing",
            "label": "AED0.20 per transaction for accounts on standard pricing",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["AED0.20 per transaction for accounts on standard pricing"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="fixed_amount", amount="0.20", currency="AED", minor_amount="20")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "pricing_plan=standard" in str(excinfo.value).lower()


def test_semantic_validation_fails_add_on_published_as_payments() -> None:
    """An add-on fee must not be published under the payments product."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "payments",
            "variant_id": "online_domestic_cards",
            "label": "Smart Disputes fee 30% for each dispute you win",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Smart Disputes fee 30% for each dispute you win"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="30", basis_points="3000")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "smart_disputes source evidence published" in str(excinfo.value).lower()


def test_semantic_validation_fails_smart_disputes_missing_feature() -> None:
    """A Smart Disputes rule must require feature_enabled=smart_disputes."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "smart_disputes",
            "variant_id": "won_dispute",
            "label": "Smart Disputes fee 30% for each dispute you win",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Smart Disputes fee 30% for each dispute you win"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="30", basis_points="3000")],
            "conditions": [FeeCondition(dimension="dispute_state", value="won")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "feature_enabled=smart_disputes" in str(excinfo.value).lower()


def test_semantic_validation_fails_starting_at_published_exact() -> None:
    """A starting-at rate must not be published as exact."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "adaptive_pricing",
            "label": "Customers will be presented a conversion fee starting at 2%",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Customers will be presented a conversion fee starting at 2%"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="2", basis_points="200")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "starting at" in str(excinfo.value).lower()


def test_semantic_validation_fails_customer_paid_missing_payer() -> None:
    """A customer-paid conversion fee must carry payer=customer."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "adaptive_pricing",
            "exactness": "from",
            "label": "Customers will be presented a conversion fee starting at 2%",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Customers will be presented a conversion fee starting at 2%"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="2", basis_points="200")],
            "conditions": [FeeCondition(dimension="fee_type", value="conversion_fee")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "payer=customer" in str(excinfo.value).lower()


def test_semantic_validation_passes_valid_adaptive_pricing_from() -> None:
    """A valid starting-at, customer-paid conversion fee passes semantic checks."""
    rule = _valid_core_rule().model_copy(
        update={
            "product_id": "adaptive_pricing",
            "variant_id": "standard",
            "exactness": "from",
            "label": "Customers will be presented a conversion fee starting at 2%",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Customers will be presented a conversion fee starting at 2%"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="2", basis_points="200")],
            "conditions": [
                FeeCondition(dimension="account_country", value="AE"),
                FeeCondition(dimension="payer", value="customer"),
                FeeCondition(dimension="fee_type", value="conversion_fee"),
            ],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    result = validate_semantic(
        "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
    )
    assert result["success"]


def test_semantic_validation_fails_market_share_evidence() -> None:
    """A calculable rule whose source text is a market-share statistic must be rejected."""
    rule = _valid_core_rule().model_copy(
        update={
            "label": "Pix 40% share of online payments",
            "fee_evidence": FeeEvidence(
                type="explicit_fee_phrase",
                phrases=["Pix 40% share of online payments"],
                confidence=0.85,
            ),
            "fee_components": [FeeComponent(type="percentage", value="40", basis_points="4000")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "market-share" in str(excinfo.value).lower()


def test_semantic_validation_fails_cross_fragment_evidence() -> None:
    """A rule whose fee evidence type is cross_fragment_fee_evidence must be rejected."""
    rule = _valid_core_rule().model_copy(
        update={
            "label": "per successful charge for international transactions",
            "fee_evidence": FeeEvidence(
                type="cross_fragment_fee_evidence",
                phrases=["40% share of online payments", "per successful charge for international transactions"],
                confidence=0.0,
            ),
            "fee_components": [FeeComponent(type="percentage", value="40", basis_points="4000")],
        }
    )
    core_fees = _core_fees_with_rule(rule)
    with pytest.raises(CrawlerValidationError) as excinfo:
        validate_semantic(
            "/unused", core_fees=core_fees, manifest=_market_manifest_for_ae(), payment_methods=_payment_methods()
        )
    assert "different source fragments" in str(excinfo.value).lower()


def _valid_repo_with_unsupported(tmp_path: Path) -> None:
    supported = _valid_complete_output("DE")
    unsupported_market = Market(
        stripe_market_code="id",
        account_country="ID",
        country_name="Indonesia",
        locale="en-id",
        url_prefix="https://stripe.com/en-id",
        status="pricing_page_unavailable",
    )
    unsupported_record = UnsupportedMarket(
        stripe_market_code="id",
        account_country="ID",
        country_name="Indonesia",
        reason="pricing_page_unavailable",
        status="pricing_page_unavailable",
        requested_urls=[],
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish(
        [supported],
        [supported.market, unsupported_market],
        [unsupported_record],
        [],
    )
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, markets=2, unsupported=1, rules=0)


def test_validate_change_report_missing(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    (tmp_path / "meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta" / "crawl-report.json").write_text(json.dumps(CrawlReport().model_dump()), encoding="utf-8")
    (tmp_path / "README.md").write_text("<!-- STATS_START -->\n<!-- STATS_END -->", encoding="utf-8")
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "change-report.json is missing" in str(exc_info.value)


def test_validate_change_report_malformed(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    (tmp_path / "change-report.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "not a valid ChangeReport" in str(exc_info.value)


def test_validate_change_report_has_regression(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    (tmp_path / "change-report.json").write_text(
        json.dumps(
            ChangeReport(
                changes=[{"kind": "removed_market", "country_code": "US"}],
            ).model_dump()
        ),
        encoding="utf-8",
    )
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "has_regression is true" in str(exc_info.value)


def test_validate_crawl_report_missing(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    (tmp_path / "change-report.json").write_text(json.dumps(ChangeReport().model_dump()), encoding="utf-8")
    (tmp_path / "README.md").write_text("<!-- STATS_START -->\n<!-- STATS_END -->", encoding="utf-8")
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "meta/crawl-report.json is missing" in str(exc_info.value)


def test_validate_readme_core_rule_count_mismatch(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("**0**", "**999**")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_all_output(tmp_path, strict=True)
    assert "README core fee rules" in str(exc_info.value)


def test_validate_require_all_complete_unclassified_in_index(tmp_path: Path) -> None:
    output = _valid_complete_output("DE")
    output = output.model_copy(
        update={"derivation_status": "unclassified", "calculator_coverage_status": "unclassified"}
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_data_repository(tmp_path, strict=True, require_all_complete=True)
    assert "supported market is not complete" in str(exc_info.value)


def test_validate_require_all_complete_explicitly_unsupported(tmp_path: Path) -> None:
    _valid_repo_with_unsupported(tmp_path)
    result = validate_data_repository(tmp_path, strict=True, require_all_complete=True)
    assert result["success"]


def test_validate_require_all_complete_missing_unsupported(tmp_path: Path) -> None:
    """A market in the manifest that is not supported and not recorded as unsupported fails."""
    supported = _valid_complete_output("DE")
    missing_market = Market(
        stripe_market_code="id",
        account_country="ID",
        country_name="Indonesia",
        locale="en-id",
        url_prefix="https://stripe.com/en-id",
        status="pricing_page_unavailable",
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([supported], [supported.market, missing_market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, markets=2, rules=0)
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_data_repository(tmp_path, strict=True, require_all_complete=True)
    assert "expected exactly one state" in str(exc_info.value)


def test_validate_completeness_index_core_status_mismatch(tmp_path: Path) -> None:
    supported = _valid_complete_output("DE")
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([supported], [supported.market], [], [])
    publisher.commit(staging, validate=False)
    _write_minimal_publication_files(tmp_path, rules=0)
    core_path = tmp_path / "json" / "core-fees.json"
    core_data = json.loads(core_path.read_text(encoding="utf-8"))
    core_data["markets"][0]["derivation_status"] = "partial"
    core_path.write_text(json.dumps(core_data), encoding="utf-8")
    with pytest.raises(CrawlerValidationError) as exc_info:
        validate_data_repository(tmp_path, strict=True, require_all_complete=True)
    assert "derivation_status mismatch" in str(exc_info.value)
