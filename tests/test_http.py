"""Tests for the HTTP client."""

from __future__ import annotations

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
        calls.append({
            "url": str(request.url),
            "headers": dict(request.headers),
        })
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
