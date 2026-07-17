"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from stripe_fee_crawler.exceptions import NetworkError
from stripe_fee_crawler.http import HttpClient
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


@pytest.fixture(autouse=True)
def _deny_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block real HTTP in unit tests unless explicitly marked live."""
    if request.node.get_closest_marker("live"):
        return

    original = HttpClient._request

    async def _patched_request(self: HttpClient, method: str, url: str, **kwargs: object) -> object:
        # Tests using httpx.MockTransport are allowed; real transport is not.
        if self._transport is not None:
            return await original(self, method, url, **kwargs)
        if self.config.offline_fixtures and url in self.config.offline_fixtures:
            return await original(self, method, url, **kwargs)
        raise NetworkError(f"Network denied in unit tests: {url}")

    monkeypatch.setattr(HttpClient, "_request", _patched_request)
