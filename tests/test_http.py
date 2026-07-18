"""Tests for the HTTP client."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from stripe_fee_crawler.exceptions import ContentSecurityError
from stripe_fee_crawler.http import HttpClient, HttpResponse
from stripe_fee_crawler.models import CrawlConfiguration


@pytest.mark.asyncio
async def test_http_client_allows_stripe_domain() -> None:
    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    # Just validate that the URL passes the allowlist check via internal helper.
    assert client._is_allowed_host("https://stripe.com/pricing")
    assert client._is_allowed_host("https://www.stripe.com/pricing")
    await client.close()


@pytest.mark.asyncio
async def test_http_client_rejects_disallowed_domain() -> None:
    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    assert not client._is_allowed_host("https://example.com/pricing")
    await client.close()


@pytest.mark.asyncio
async def test_http_client_validate_url_rejects_http() -> None:
    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    with pytest.raises(ContentSecurityError):
        client._validate_url("ftp://stripe.com/pricing")
    await client.close()


@pytest.mark.asyncio
async def test_http_client_sanitize_url() -> None:
    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    from stripe_fee_crawler.http import _sanitize_url

    assert "token=<redacted>" in _sanitize_url("https://stripe.com/pricing?token=secret&foo=bar")
    await client.close()


@pytest.mark.asyncio
async def test_http_client_fixture_support() -> None:
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        offline_fixtures={"https://stripe.com/pricing": "tests/fixtures/us-pricing.html"},
    )
    client = HttpClient(config)
    response = await client.get("https://stripe.com/pricing")
    assert response.status_code == 200
    assert "Standard pricing" in response.text
    await client.close()


@pytest.mark.asyncio
async def test_http_client_detects_challenge() -> None:
    from stripe_fee_crawler.exceptions import AccessChallengeError

    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    html_text = "<html><head><title>Security Check</title></head><body><form action='captcha'><input name='captcha'/></form></body></html>"
    response = HttpResponse(
        url="https://stripe.com/pricing",
        status_code=200,
        content=html_text.encode(),
        text=html_text,
        headers={"content-type": "text/html"},
    )
    with pytest.raises(AccessChallengeError):
        client._detect_blocking_page(response)
    await client.close()


@pytest.mark.asyncio
async def test_http_client_detects_login_page() -> None:
    from stripe_fee_crawler.exceptions import AccessChallengeError

    config = CrawlConfiguration(max_workers=1, request_delay=0.0)
    client = HttpClient(config)
    html_text = "<html><head><title>Sign In</title></head><body><form>Stripe account sign-in<input type='password'/></form></body></html>"
    response = HttpResponse(
        url="https://stripe.com/login",
        status_code=200,
        content=html_text.encode(),
        text=html_text,
        headers={"content-type": "text/html"},
    )
    with pytest.raises(AccessChallengeError):
        client._detect_blocking_page(response)
    await client.close()


def _make_mock_transport(calls: list[dict[str, Any]], cookie_response: bool = False) -> httpx.MockTransport:
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        calls.append(
            {
                "url": str(request.url),
                "headers": dict(request.headers),
            }
        )
        count += 1
        if cookie_response and count == 1:
            return httpx.Response(
                200,
                text="pricing",
                headers={"set-cookie": "session=1; Path=/"},
            )
        return httpx.Response(200, text="pricing")

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_http_client_caches_fresh_responses(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _make_mock_transport(calls)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(tmp_path),
    )
    client = HttpClient(config, transport=transport)

    first = await client.get("https://stripe.com/pricing")
    assert first.status_code == 200
    assert not first.from_cache

    second = await client.get("https://stripe.com/pricing")
    assert second.status_code == 200
    assert second.from_cache
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_http_client_no_cache_bypasses_cache(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _make_mock_transport(calls)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(tmp_path),
        no_cache=True,
    )
    client = HttpClient(config, transport=transport)

    await client.get("https://stripe.com/pricing")
    await client.get("https://stripe.com/pricing")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_http_client_cookie_isolation(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    transport = _make_mock_transport(calls, cookie_response=True)
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0.0,
        cache_dir=str(tmp_path),
        no_cache=True,
    )
    client = HttpClient(config, transport=transport)

    await client.get("https://stripe.com/pricing")
    await client.get("https://stripe.com/pricing")
    assert len(calls) == 2
    assert calls[1]["headers"].get("Cookie") is None


@pytest.mark.asyncio
async def test_cache_isolated_by_market(tmp_path: Path) -> None:
    responses = iter([b"<html>US</html>", b"<html>DE</html>"])
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=next(responses))

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    us_response = await client.get("https://stripe.com/pricing", market="US", locale="en-US")
    de_response = await client.get("https://stripe.com/pricing", market="DE", locale="en-DE")
    us_again = await client.get("https://stripe.com/pricing", market="US", locale="en-US")

    assert us_response.text == "<html>US</html>"
    assert de_response.text == "<html>DE</html>"
    assert us_again.from_cache
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cache_key_includes_locale_and_market_headers(tmp_path: Path) -> None:
    from stripe_fee_crawler.http_cache import _cache_key

    headers = {"Accept": "text/html", "Accept-Language": "en-US,en;q=0.5"}
    key_us = _cache_key("GET", "https://stripe.com/pricing", "US", "en-US", headers)
    key_de = _cache_key("GET", "https://stripe.com/pricing", "DE", "en-DE", headers)
    assert key_us != key_de


@pytest.mark.asyncio
async def test_concurrent_identical_requests_download_once(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=b"<html>ok</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    r1, r2 = await asyncio.gather(
        client.get("https://stripe.com/pricing", market="US"),
        client.get("https://stripe.com/pricing", market="US"),
    )

    assert r1.text == "<html>ok</html>"
    assert r2.text == "<html>ok</html>"
    assert r2.from_cache or r1.from_cache
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_no_store_response_is_not_cached(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(
            200,
            content=f"<html>call {len(calls)}</html>".encode(),
            headers={"cache-control": "no-store"},
        )

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(tmp_path))
    client = HttpClient(config, transport=transport)

    r1 = await client.get("https://stripe.com/pricing", market="US")
    r2 = await client.get("https://stripe.com/pricing", market="US")

    assert r1.text == "<html>call 1</html>"
    assert r2.text == "<html>call 2</html>"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_no_cache_directive_does_not_force_revalidation(tmp_path: Path) -> None:
    """A fresh entry with an origin ``no-cache`` directive is served from the snapshot TTL."""
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
        "content": base64.b64encode(b"<html>cached</html>").decode("ascii"),
        "etag": '"abc"',
        "last_modified": None,
        "fetched_at": time.time(),
        "market": "US",
        "locale": "en-US",
        "detected_market": "US",
        "detected_locale": "en-US",
        "cache_control": "no-cache",
    }
    entry_path.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url), "headers": dict(request.headers)})
        return httpx.Response(200, content=b"<html>should not be used</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(cache_dir))
    client = HttpClient(config, transport=transport)

    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 0


@pytest.mark.asyncio
async def test_expired_entry_revalidates_with_etag(tmp_path: Path) -> None:
    """An entry older than the snapshot TTL is revalidated, and a 304 refreshes it."""
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
        "content": base64.b64encode(b"<html>cached</html>").decode("ascii"),
        "etag": '"abc"',
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
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(cache_dir))
    client = HttpClient(config, transport=transport)

    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 1
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    assert float(data["fetched_at"]) > entry["fetched_at"]


@pytest.mark.asyncio
async def test_corrupt_cache_entry_is_replaced(tmp_path: Path) -> None:
    from stripe_fee_crawler.http_cache import _cache_key

    cache_dir = tmp_path
    url = "https://stripe.com/pricing"
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", "Accept-Language": "en-US"}
    key = _cache_key("GET", url, "US", "en-US", headers)
    entry_path = cache_dir / "entries" / key[:2] / f"{key}.json"
    entry_path.parent.mkdir(parents=True, exist_ok=True)
    entry_path.write_text("not valid json", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>fresh</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(cache_dir))
    client = HttpClient(config, transport=transport)

    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html>fresh</html>"
    assert entry_path.exists()
    data = json.loads(entry_path.read_text(encoding="utf-8"))
    assert base64.b64decode(data["content"]).decode("utf-8") == "<html>fresh</html>"


@pytest.mark.asyncio
async def test_cache_invalidated_on_market_mismatch(tmp_path: Path) -> None:
    """A cached DE response stored under a US key must be discarded and refetched."""
    from stripe_fee_crawler.http_cache import _cache_key

    cache_dir = tmp_path
    url = "https://stripe.com/en-us/pricing"
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
        "content": base64.b64encode(b"<html lang='de-DE'>German pricing</html>").decode("ascii"),
        "etag": '"abc"',
        "last_modified": None,
        "fetched_at": time.time(),
        "market": "US",
        "locale": "en-US",
        "detected_market": "DE",
        "detected_locale": "de-DE",
        "cache_control": None,
    }
    entry_path.write_text(json.dumps(entry, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({"url": str(request.url)})
        return httpx.Response(200, content=b"<html lang='en-US'>US pricing</html>")

    transport = httpx.MockTransport(handler)
    config = CrawlConfiguration(max_workers=1, request_delay=0.0, cache_dir=str(cache_dir))
    client = HttpClient(config, transport=transport)

    response = await client.get(url, market="US", locale="en-US")
    assert response.text == "<html lang='en-US'>US pricing</html>"
    assert response.detected_market == "US"
    assert not response.from_cache
    assert len(calls) == 1
