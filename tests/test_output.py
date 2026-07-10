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
    publisher.publish_markets([output])
    path1 = publisher.staging_dir / "json" / "DE.json"
    assert path1.exists()
    data1 = path1.read_bytes()

    # Re-publish to a fresh publisher and compare bytes.
    publisher2 = OutputPublisher(tmp_path / "second", timestamp=None)
    publisher2.publish_markets([output])
    path2 = publisher2.staging_dir / "json" / "DE.json"
    data2 = path2.read_bytes()
    assert data1 == data2


def test_publish_creates_index(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    publisher.publish_markets([_minimal_output("DE"), _minimal_output("US")])
    index_path = publisher.staging_dir / "json" / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert len(data["markets"]) == 2
    assert data["markets"][0]["account_country"] < data["markets"][1]["account_country"]


def test_commit_swaps_files(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    publisher.publish_markets([_minimal_output("DE")])
    publisher.commit(validate=False)
    assert (tmp_path / "json" / "DE.json").exists()


def test_rollback(tmp_path: Path) -> None:
    publisher = OutputPublisher(tmp_path, timestamp=None)
    publisher._init_staging()
    staging = publisher.staging_dir
    assert staging.exists()
    publisher.rollback()
    assert not staging.exists()
