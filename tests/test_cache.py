"""Tests for HTTP cache activation, defaults, and persistence."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from stripe_fee_crawler.cli import _config_from_options, _parse_env_bool
from stripe_fee_crawler.discovery import build_market_from_code
from stripe_fee_crawler.exceptions import ConfigurationError
from stripe_fee_crawler.http import HttpClient
from stripe_fee_crawler.http_cache import HttpCache, _default_cache_dir, _resolve_cache_dir
from stripe_fee_crawler.models import CrawlConfiguration, CrawlReport, MarketOutput, Source
from stripe_fee_crawler.output import OutputPublisher

# ---------------------------------------------------------------------------
# Environment and default path tests
# ---------------------------------------------------------------------------


def test_default_cache_dir_uses_xdg_cache_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/custom-xdg")
    monkeypatch.delenv("HOME", raising=False)
    path = _default_cache_dir()
    assert str(path) == "/tmp/custom-xdg/stripe-fee-crawler/http"


def test_default_cache_dir_falls_back_to_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _default_cache_dir()
    assert path == tmp_path / ".cache" / "stripe-fee-crawler" / "http"


def test_resolve_cache_dir_expands_tilde_and_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", "/tmp/testhome")
    assert _resolve_cache_dir("~/foo") == Path("/tmp/testhome/foo")
    assert _resolve_cache_dir("$HOME/foo") == Path("/tmp/testhome/foo")
    assert _resolve_cache_dir("/absolute/path") == Path("/absolute/path")


def test_http_cache_uses_default_path_when_no_cache_dir_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    config = CrawlConfiguration()
    cache = HttpCache(config)
    assert cache.stats.cache_enabled is True
    assert cache.stats.cache_dir == str(tmp_path / ".cache" / "stripe-fee-crawler" / "http")
    assert cache._configured_dir == tmp_path / ".cache" / "stripe-fee-crawler" / "http"


# ---------------------------------------------------------------------------
# Boolean environment parsing
# ---------------------------------------------------------------------------


def test_parse_env_bool_true_values() -> None:
    for value in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
        assert _parse_env_bool("TEST", value) is True


def test_parse_env_bool_false_values() -> None:
    for value in (None, "", "0", "false", "FALSE", "no", "NO", "off", "OFF"):
        assert _parse_env_bool("TEST", value) is False


def test_parse_env_bool_rejects_invalid() -> None:
    with pytest.raises(ConfigurationError, match="Invalid boolean value for TEST"):
        _parse_env_bool("TEST", "maybe")


def test_config_from_options_parses_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIPE_FEE_CRAWLER_NO_CACHE", "false")
    monkeypatch.setenv("STRIPE_FEE_CRAWLER_REFRESH_CACHE", "0")
    monkeypatch.setenv("STRIPE_FEE_CRAWLER_CACHE_TTL_HOURS", "12")
    config = _config_from_options()
    assert config.no_cache is False
    assert config.refresh_cache is False
    assert config.cache_ttl_hours == 12.0


def test_config_from_options_no_cache_true_disables_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRIPE_FEE_CRAWLER_NO_CACHE", "yes")
    config = _config_from_options()
    assert config.no_cache is True


# ---------------------------------------------------------------------------
# Cache hit/miss/network statistics
# ---------------------------------------------------------------------------


def _mock_transport(calls: list[dict[str, Any]], response_text: str = "ok") -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=response_text.encode())

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_first_run_downloads_and_writes_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _mock_transport(calls, "first")
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    response = await client.get("https://stripe.com/pricing", market="US", locale="en-US")
    assert response.status_code == 200
    assert response.text == "first"
    assert response.from_cache is False
    stats = client.cache_stats
    assert stats.cache_writes == 1
    assert stats.cache_misses == 1
    assert stats.network_requests == 1
    assert stats.cache_hits == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_second_run_uses_cache_without_network(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _mock_transport(calls, "first")
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client1 = HttpClient(config, transport=transport)
    await client1.get("https://stripe.com/pricing", market="US", locale="en-US")

    calls.clear()
    should_not_be_called = _mock_transport(calls, "second")
    client2 = HttpClient(config, transport=should_not_be_called)
    response = await client2.get("https://stripe.com/pricing", market="US", locale="en-US")
    assert response.text == "first"
    assert response.from_cache is True
    stats = client2.cache_stats
    assert stats.cache_hits == 1
    assert stats.network_requests == 0
    assert stats.cache_misses == 0
    assert stats.cache_writes == 0
    assert stats.bytes_avoided > 0
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_cache_persists_across_separate_client_and_crawler_instances(tmp_path: Path) -> None:
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    transport1 = _mock_transport([], "shared")
    client1 = HttpClient(config, transport=transport1)
    await client1.get("https://stripe.com/pricing", market="US", locale="en-US")

    transport2 = _mock_transport([], "should-not-be-used")
    client2 = HttpClient(config, transport=transport2)
    response = await client2.get("https://stripe.com/pricing", market="US", locale="en-US")
    assert response.text == "shared"
    assert response.from_cache is True
    assert client2.cache_stats.cache_hits == 1
    assert client2.cache_stats.network_requests == 0


# ---------------------------------------------------------------------------
# no-cache / refresh-cache / no-store / private
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cache_flag_bypasses_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _mock_transport(calls, "data")
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path), no_cache=True)
    client = HttpClient(config, transport=transport)

    await client.get("https://stripe.com/pricing", market="US", locale="en-US")
    await client.get("https://stripe.com/pricing", market="US", locale="en-US")

    assert len(calls) == 2
    assert client.cache_stats.cache_writes == 0
    assert client.cache_stats.cache_enabled is False
    assert not any(tmp_path.rglob("*.json"))


@pytest.mark.asyncio
async def test_refresh_cache_revalidates_expired_entry(tmp_path: Path) -> None:
    from stripe_fee_crawler.http_cache import _cache_key

    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    key = _cache_key("GET", url, "US", "en-US", headers)
    entry_path = cache_dir / "entries" / key[:2] / f"{key}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "v": "2",
        "key": key,
        "url": url,
        "final_url": url,
        "status_code": 200,
        "headers": {"content-type": "text/html"},
        "content": base64.b64encode(b"<html>old</html>").decode("ascii"),
        "etag": '"old"',
        "last_modified": None,
        "fetched_at": time.time() - 25 * 3600,
        "market": "US",
        "locale": "en-US",
        "detected_market": "US",
        "detected_locale": "en-US",
        "cache_control": None,
    }
    entry_path.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        if request.headers.get("if-none-match") == '"old"':
            return httpx.Response(200, content=b"<html>refreshed</html>")
        return httpx.Response(200, content=b"<html>refreshed</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        refresh_cache=True,
    )
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>refreshed</html>"
    assert client.cache_stats.cache_revalidations >= 1
    assert client.cache_stats.network_requests == 1
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    assert base64.b64decode(data["content"]).decode("utf-8") == "<html>refreshed</html>"


@pytest.mark.asyncio
async def test_private_response_is_not_cached(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(
            200, content=f"<html>call {len(calls)}</html>".encode(), headers={"cache-control": "private"}
        )

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    r1 = await client.get("https://stripe.com/pricing", market="US", locale="en-US")
    r2 = await client.get("https://stripe.com/pricing", market="US", locale="en-US")

    assert r1.text == "<html>call 1</html>"
    assert r2.text == "<html>call 2</html>"
    assert len(calls) == 2
    assert not any(tmp_path.rglob("*.json"))


# ---------------------------------------------------------------------------
# Corrupt-cache recovery and market/locale isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrupt_cache_entry_triggers_recovery_and_counts_error(tmp_path: Path) -> None:
    from stripe_fee_crawler.http_cache import _cache_key

    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    key = _cache_key("GET", url, "US", "en-US", headers)
    entry_path = cache_dir / "entries" / key[:2] / f"{key}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text("not valid json", encoding="utf-8")

    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"<html>fresh</html>"))
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(cache_dir))
    client = HttpClient(config, transport=transport)

    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>fresh</html>"
    assert client.cache_stats.cache_errors == 1
    assert entry_path.exists()
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    assert base64.b64decode(data["content"]).decode("utf-8") == "<html>fresh</html>"


@pytest.mark.asyncio
async def test_cache_isolated_by_locale(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    responses = iter([b"en", b"de"])

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=next(responses))

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    en = await client.get("https://stripe.com/pricing", market="US", locale="en-US")
    de = await client.get("https://stripe.com/pricing", market="DE", locale="en-DE")
    en_again = await client.get("https://stripe.com/pricing", market="US", locale="en-US")

    assert en.text == "en"
    assert de.text == "de"
    assert en_again.from_cache
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Atomic publication cache safety
# ---------------------------------------------------------------------------


def test_atomic_publication_does_not_remove_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "output"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Populate the cache with an arbitrary entry file.
    entry_file = cache_dir / "entries" / "ab" / "test.json"
    entry_file.parent.mkdir(parents=True, exist_ok=True)
    entry_file.write_text("{}", encoding="utf-8")

    market = build_market_from_code("DE", status="supported")
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )
    publisher = OutputPublisher(str(output_dir), timestamp="2026-01-01T00:00:00Z")
    changed, staging = publisher.publish([output], [market], [], [])
    publisher.commit(staging, validate=False)

    assert entry_file.exists()


# ---------------------------------------------------------------------------
# Cache policy tests
# ---------------------------------------------------------------------------


def _write_cache_entry(
    cache_dir: Path,
    url: str,
    headers: dict[str, str],
    fetched_at: float,
    cache_control: str | None,
    content: bytes = b"<html>cached</html>",
    market: str = "US",
    locale: str = "en-US",
) -> Path:
    from stripe_fee_crawler.http_cache import _cache_key

    key = _cache_key("GET", url, market, locale, headers)
    entry_path = cache_dir / "entries" / key[:2] / f"{key}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "v": "2",
        "key": key,
        "url": url,
        "final_url": url,
        "status_code": 200,
        "headers": {"content-type": "text/html"},
        "content": base64.b64encode(content).decode("ascii"),
        "etag": '"abc"',
        "last_modified": None,
        "fetched_at": fetched_at,
        "market": market,
        "locale": locale,
        "detected_market": market,
        "detected_locale": locale,
        "cache_control": cache_control,
    }
    entry_path.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return entry_path


@pytest.mark.asyncio
async def test_ttl_policy_ignores_shorter_origin_max_age(tmp_path: Path) -> None:
    """Configured TTL (24h) wins over a short origin max-age when policy is ttl."""
    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    age = 2 * 3600  # 2 hours
    _write_cache_entry(
        cache_dir,
        url,
        headers,
        fetched_at=time.time() - age,
        cache_control="max-age=60",
    )

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=b"<html>network</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        cache_ttl_hours=24.0,
        cache_policy="ttl",
    )
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache is True
    assert client.cache_stats.cache_hits == 1
    assert client.cache_stats.network_requests == 0
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_http_policy_revalidates_when_origin_max_age_expired(tmp_path: Path) -> None:
    """http policy respects a short origin max-age and revalidates once it expires."""
    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    age = 2 * 3600
    _write_cache_entry(
        cache_dir,
        url,
        headers,
        fetched_at=time.time() - age,
        cache_control="max-age=60",
    )

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        if request.headers.get("if-none-match") == '"abc"':
            return httpx.Response(304, headers={"etag": '"abc"'})
        return httpx.Response(200, content=b"<html>network</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        cache_ttl_hours=24.0,
        cache_policy="http",
    )
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache is True
    assert client.cache_stats.cache_revalidations >= 1
    assert client.cache_stats.network_requests == 1
    assert client.cache_stats.cache_hits == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_ttl_policy_ignores_origin_no_cache_and_max_age_zero(tmp_path: Path) -> None:
    """In ttl policy, origin no-cache and max-age=0 do not force revalidation."""
    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    _write_cache_entry(
        cache_dir,
        url,
        headers,
        fetched_at=time.time() - 2 * 3600,
        cache_control="no-cache, max-age=0",
    )

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=b"<html>network</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        cache_ttl_hours=24.0,
        cache_policy="ttl",
    )
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache is True
    assert client.cache_stats.network_requests == 0
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_entries_older_than_configured_ttl_revalidate(tmp_path: Path) -> None:
    """An entry older than the configured TTL revalidates regardless of origin directives."""
    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    _write_cache_entry(
        cache_dir,
        url,
        headers,
        fetched_at=time.time() - 25 * 3600,
        cache_control="max-age=86400",
    )

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(304, headers={"etag": '"abc"'})

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        cache_ttl_hours=24.0,
        cache_policy="ttl",
    )
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache is True
    assert client.cache_stats.cache_revalidations >= 1
    assert client.cache_stats.network_requests == 1


@pytest.mark.asyncio
async def test_two_separate_crawler_processes_reuse_fresh_ttl_entry(tmp_path: Path) -> None:
    """A fresh entry written by one process is reused by another with ttl policy."""
    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    _write_cache_entry(
        cache_dir,
        url,
        headers,
        fetched_at=time.time() - 2 * 3600,
        cache_control="max-age=60",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("Network should not be called for a fresh ttl entry")

    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(cache_dir),
        cache_ttl_hours=24.0,
        cache_policy="ttl",
    )
    transport = httpx.MockTransport(handler)
    client = HttpClient(config, transport=transport)
    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache is True
    assert client.cache_stats.network_requests == 0


# ---------------------------------------------------------------------------
# Makefile target tests
# ---------------------------------------------------------------------------


CRAWLER_DIR = Path(__file__).resolve().parent.parent


def test_makefile_regenerate_uses_persistent_cache() -> None:
    result = subprocess.run(
        ["make", "-n", "regenerate"],
        cwd=str(CRAWLER_DIR),
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "--cache-dir" in result.stdout
    assert "--cache-ttl-hours" in result.stdout
    assert "--cache-policy" in result.stdout
    assert ".cache/stripe-fee-crawler/http" in result.stdout


def test_makefile_regenerate_strict_uses_ttl_policy() -> None:
    result = subprocess.run(
        ["make", "-n", "regenerate-strict"],
        cwd=str(CRAWLER_DIR),
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "--cache-dir" in result.stdout
    assert "--cache-ttl-hours" in result.stdout
    assert "--cache-policy" in result.stdout
    assert "--cache-policy ttl" in result.stdout or '--cache-policy "ttl"' in result.stdout
    assert "--fail-on-regression" in result.stdout


def test_makefile_regenerate_refresh_uses_refresh_cache() -> None:
    result = subprocess.run(
        ["make", "-n", "regenerate-refresh"],
        cwd=str(CRAWLER_DIR),
        capture_output=True,
        text=True,
        env={**os.environ, "XDG_CACHE_HOME": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "--refresh-cache" in result.stdout
    assert "--cache-dir" in result.stdout
    assert "--cache-policy" in result.stdout


# ---------------------------------------------------------------------------
# CrawlReport cache-stat fields
# ---------------------------------------------------------------------------


def test_crawl_report_includes_cache_stats() -> None:
    report = CrawlReport(cache_stats=HttpCache(CrawlConfiguration(cache_dir="/tmp/cache")).stats)
    dumped = report.model_dump()
    stats = dumped["cache_stats"]
    assert stats["cache_enabled"] is True
    assert stats["cache_dir"] == "/tmp/cache"
    assert stats["cache_ttl_hours"] == 24.0
    assert stats["cache_policy"] == "ttl"
    assert "cache_hits" in stats
    assert "cache_misses" in stats
    assert "cache_writes" in stats
    assert "network_requests" in stats
