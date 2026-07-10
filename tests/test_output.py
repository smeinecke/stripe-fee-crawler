"""Tests for deterministic output and atomic publishing."""

from __future__ import annotations

import json
from pathlib import Path

from stripe_fee_crawler.models import Market, MarketOutput, Source
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
