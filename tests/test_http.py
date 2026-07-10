"""Tests for the HTTP client."""

from __future__ import annotations

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
