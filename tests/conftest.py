"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from stripe_fee_crawler.models import CrawlConfiguration


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def default_config() -> CrawlConfiguration:
    return CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        timeout=10.0,
    )


@pytest.fixture
def de_pricing_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "de-pricing.html").read_text(encoding="utf-8")


@pytest.fixture
def de_lpm_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "de-lpm.html").read_text(encoding="utf-8")


@pytest.fixture
def us_pricing_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "us-pricing.html").read_text(encoding="utf-8")


@pytest.fixture
def jp_pricing_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "jp-pricing.html").read_text(encoding="utf-8")


@pytest.fixture
def from_pricing_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "from-pricing.html").read_text(encoding="utf-8")
