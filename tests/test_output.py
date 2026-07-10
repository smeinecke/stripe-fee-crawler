"""Tests for deterministic output and atomic publishing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from stripe_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from stripe_fee_crawler.models import ChangeReport, Market, MarketOutput, Source
from stripe_fee_crawler.output import OutputPublisher


def _minimal_output(country: str = "DE") -> MarketOutput:
    market = Market(
        stripe_market_code="en-de",
        account_country=country,
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    return MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )


def test_deterministic_json_output(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output()
    _, staging1 = publisher.publish([output], [output.market], [], [])
    path1 = staging1 / "json" / "DE.json"
    assert path1.exists()
    data1 = path1.read_bytes()

    # Re-publish to a fresh publisher and compare bytes.
    publisher2 = OutputPublisher(tmp_path / "second", timestamp=None)
    _, staging2 = publisher2.publish([output], [output.market], [], [])
    path2 = staging2 / "json" / "DE.json"
    data2 = path2.read_bytes()
    assert data1 == data2


def test_publish_creates_index(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    de_output = _minimal_output("DE")
    us_output = _minimal_output("US")
    _, staging = publisher.publish([de_output, us_output], [de_output.market, us_output.market], [], [])
    index_path = staging / "json" / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert len(data["markets"]) == 2
    assert data["markets"][0]["account_country"] < data["markets"][1]["account_country"]


def test_publish_generates_schemas(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    schemas_dir = staging / "schemas"
    assert (schemas_dir / "stripe-fees-v1.schema.json").exists()
    assert (schemas_dir / "core-fees-v1.schema.json").exists()
    assert (schemas_dir / "payment-methods-v1.schema.json").exists()
    assert (schemas_dir / "index-v1.schema.json").exists()
    assert (schemas_dir / "manifest-v1.schema.json").exists()


def test_commit_swaps_files(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    assert (tmp_path / "json" / "DE.json").exists()


def test_rollback(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([], [], [], [])
    assert staging.exists()
    publisher.rollback(staging)
    assert not staging.exists()


def test_commit_no_change(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)

    publisher2 = OutputPublisher(tmp_path, timestamp=None)
    _, staging2 = publisher2.publish([output], [output.market], [], [])
    changed, changed_files = publisher2.commit(staging2, validate=False)
    assert not changed
    assert changed_files == []
    assert not staging2.exists()


def test_commit_detects_changes(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)

    publisher2 = OutputPublisher(tmp_path, timestamp=None)
    output2 = _minimal_output("US")
    _, staging2 = publisher2.publish([output2], [output2.market], [], [])
    changed, changed_files = publisher2.commit(staging2, validate=False)
    assert changed
    assert any("US.json" in f for f in changed_files)
    assert not (tmp_path / "json" / "DE.json").exists()


def test_staging_dir_inside_output_dir(tmp_path: Path) -> None:
    staging_dir = tmp_path / ".staging"
    publisher = OutputPublisher(tmp_path, staging_dir=staging_dir, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    assert staging == staging_dir
    assert (staging / "json" / "DE.json").exists()


def test_publish_change_report_writes_changes(tmp_path: Path) -> None:
    from stripe_fee_crawler.models import ChangeType

    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    report = ChangeReport(
        schema_version=1,
        changes=[ChangeType(kind="new_market", country_code="DE", message="new")],
        has_regression=False,
    )
    publisher.publish_change_report(staging, report)
    assert (staging / "change-report.json").exists()
    data = json.loads((staging / "change-report.json").read_text())
    assert data["has_regression"] is False
    assert len(data["changes"]) == 1


def test_publish_change_report_carries_forward(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)

    # Write an existing change-report and verify it is carried forward unchanged.
    old_report = {"schema_version": 1, "changes": [], "has_regression": False}
    (tmp_path / "change-report.json").write_text(json.dumps(old_report), encoding="utf-8")

    publisher2 = OutputPublisher(tmp_path, timestamp=None)
    _, staging2 = publisher2.publish([output], [output.market], [], [])
    publisher2.publish_change_report(staging2, ChangeReport())
    assert (staging2 / "change-report.json").exists()
    data = json.loads((staging2 / "change-report.json").read_text())
    assert data["changes"] == []


def test_commit_rolls_back_live_on_validation_failure(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    _, staging = publisher.publish([output], [output.market], [], [])
    publisher.commit(staging, validate=False)
    original_de = (tmp_path / "json" / "DE.json").read_text()

    publisher2 = OutputPublisher(tmp_path, timestamp=None)
    output2 = _minimal_output("US")
    _, staging2 = publisher2.publish([output2], [output2.market], [], [])

    def _fake_validate(path: Path, strict: bool = False) -> dict[str, Any]:
        # Fail validation only when checking the live (post-rename) output dir.
        if ".staging" not in str(path):
            return {"errors": ["live validation failed"]}
        return {"errors": []}

    with (
        mock.patch("stripe_fee_crawler.output.validate_all_output", side_effect=_fake_validate),
        pytest.raises(CrawlerValidationError),
    ):
        publisher2.commit(staging2, validate=True)

    # Backup should restore the original json/ tree and staging should be cleaned up.
    assert (tmp_path / "json" / "DE.json").read_text() == original_de
    assert not (tmp_path / "json" / "US.json").exists()
    assert not staging2.exists()


def test_publish_writes_manifest_metadata(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    output = _minimal_output("DE")
    aliases = {"en-de": "de", "de-de": "de"}
    fee_page_urls = {"de": ["https://stripe.com/en-de/pricing"]}
    _, staging = publisher.publish([output], [output.market], [], [], aliases=aliases, fee_page_urls=fee_page_urls)
    manifest_path = staging / "meta" / "markets.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["aliases"] == aliases
    assert manifest["fee_page_urls"] == fee_page_urls
