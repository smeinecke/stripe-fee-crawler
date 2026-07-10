"""Tests for regression detection and change reports."""

from __future__ import annotations

import json
from pathlib import Path

from stripe_fee_crawler.models import Market, MarketOutput, Source
from stripe_fee_crawler.output import OutputPublisher
from stripe_fee_crawler.regression import check_regression


def _build_repo(path: Path, countries: list[str], rules_count: int = 1) -> None:
    from stripe_fee_crawler.models import FeeRule

    publisher = OutputPublisher(path, timestamp=None)
    outputs: list[MarketOutput] = []
    for country in countries:
        market = Market(
            stripe_market_code=country.lower(),
            account_country=country,
            country_name=country,
            locale=f"en-{country.lower()}",
            url_prefix=f"https://stripe.com/{country.lower()}",
            status="supported",
        )
        rules: list[FeeRule] = []
        if rules_count:
            rules.append(
                FeeRule(
                    rule_id=f"{country.lower()}-card",
                    name="card_payment",
                    provider="stripe",
                    unit="per_transaction",
                    exactness="exact",
                    behavior="additive",
                    classification_status="classified",
                    confidence=0.9,
                    percentage="2.9",
                )
            )
        outputs.append(
            MarketOutput(
                market=market,
                sources=[Source(requested_url=f"https://stripe.com/{country.lower()}/pricing")],
                derived_rules=rules,
                derivation_status="complete" if rules_count else "unclassified",
            )
        )
    markets = [o.market for o in outputs]
    _, staging = publisher.publish(outputs, markets, [], [])
    publisher.commit(staging, validate=False)


def test_no_regression_identical(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _build_repo(old, ["DE", "US"])
    _build_repo(new, ["DE", "US"])
    report = check_regression(old, new)
    assert not report.has_regression


def test_detect_removed_market(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _build_repo(old, ["DE", "US"])
    _build_repo(new, ["DE"])
    report = check_regression(old, new)
    assert report.has_regression
    assert any(c.kind == "removed_market" for c in report.changes)


def test_detect_lost_core_category(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    _build_repo(old, ["DE"], rules_count=1)
    _build_repo(new, ["DE"], rules_count=0)
    # Force unclassified status by rewriting new DE.json to have no rules.
    de_path = new / "json" / "DE.json"
    data = json.loads(de_path.read_text())
    data["derived_rules"] = []
    data["derivation_status"] = "unclassified"
    de_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    report = check_regression(old, new)
    assert report.has_regression
    assert any(c.kind == "lost_core_category" for c in report.changes)
