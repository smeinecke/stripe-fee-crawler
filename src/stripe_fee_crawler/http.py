"""Safe, deterministic HTTP client for the Stripe fee crawler."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from lxml import html

from .exceptions import (
    AccessChallengeError,
    ContentSecurityError,
    NetworkError,
    PermanentHttpError,
    PermanentNetworkError,
    RateLimitError,
    TransientNetworkError,
)
from .http_cache import (
    cache_key,
    is_fresh,
    read_cache_entry,
    remove_cache_entry,
    requires_revalidation,
    should_persist_to_cache,
    write_cache_entry,
)
from .models import CrawlConfiguration

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "stripe-fee-crawler/0.1.0 (+https://github.com/smeinecke/stripe-fee-crawler; research only)"


def _sanitize_url(url: str) -> str:
    """Return a URL safe for logging by stripping volatile query parameters."""
    parsed = urlparse(url)
    sensitive = {
        "token",
        "session",
        "sid",
        "auth",
        "nonce",
        "csrf",
        "request_id",
        "correlation_id",
        "visitor_id",
    }
    if not parsed.query:
        return url
    pairs = re.split(r"[;&]", parsed.query)
    kept = []
    for pair in pairs:
        if not pair:
            continue
        if "=" in pair:
            key = pair.split("=", 1)[0]
            if key.lower() in sensitive:
                kept.append(f"{key}=<redacted>")
            else:
                kept.append(pair)
        else:
            kept.append(pair)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{'&'.join(kept)}"


@dataclass
class HttpResponse:
    """Normalized HTTP response."""

    url: str
    status_code: int
    content: bytes
    text: str
    headers: dict[str, str]
    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None
    from_cache: bool = False

    def __post_init__(self) -> None:
        if self.content_sha256 is None and self.content:
            self.content_sha256 = hashlib.sha256(self.content).hexdigest()


@dataclass
class CachedSource:
    """Previously published source data for conditional requests."""

    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None


class HttpClient:
    """HTTP client with retries, allowlist, conditional request, and cache support."""

    def __init__(
        self,
        config: CrawlConfiguration | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config or CrawlConfiguration()
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
        self._transport = transport

    async def close(self) -> None:
        """No-op; each request owns its own client."""
        return

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    def _is_allowed_host(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in self.config.allowed_domains)

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ContentSecurityError(f"Unsupported URL scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise ContentSecurityError(f"Missing hostname in URL: {url}")
        if not self._is_allowed_host(url):
            raise ContentSecurityError(f"Host not in allowlist: {parsed.hostname}")

    def _detect_blocking_page(self, response: HttpResponse) -> None:
        """Raise if the response looks like a login, CAPTCHA, or generic error page."""
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(f"Rate limited (429): {response.url}", retry_after=retry_after)
        if response.status_code >= 500:
            raise TransientNetworkError(f"Server error ({response.status_code}): {response.url}")
        if response.status_code in (401, 403):
            raise AccessChallengeError(f"Access denied ({response.status_code}): {response.url}")
        if response.status_code >= 400:
            raise PermanentHttpError(f"HTTP {response.status_code} for {response.url}", response.status_code)

        title, challenge_score, login_score = self._parse_challenge_signals(response.text)

        if title and self._is_challenge_title(title) and challenge_score >= 1:
            raise AccessChallengeError(f"CAPTCHA/interstitial detected: {response.url}")
        if challenge_score >= 2:
            raise AccessChallengeError(f"CAPTCHA/interstitial detected: {response.url}")
        if login_score >= 2 and self._looks_like_login_page(response.text):
            raise AccessChallengeError(f"Login page detected: {response.url}")

    def _parse_challenge_signals(self, text: str) -> tuple[str | None, int, int]:
        """Parse HTML and return (title, challenge_score, login_score)."""
        try:
            tree = html.fromstring(text)
        except Exception:
            return None, 0, 0

        title = None
        title_node = tree.find(".//title")
        if title_node is not None and title_node.text:
            title = title_node.text.strip()

        challenge_score = 0
        login_score = 0

        challenge_selectors = [
            "//form[contains(@action, 'captcha') or contains(@id, 'captcha') or contains(@class, 'captcha')]",
            "//form[contains(@action, 'challenge') or contains(@id, 'challenge') or contains(@class, 'challenge')]",
            "//iframe[contains(@src, 'captcha') or contains(@name, 'captcha')]",
            "//iframe[contains(@src, 'challenge') or contains(@name, 'challenge')]",
            "//*[contains(@class, 'cf-turnstile') or contains(@class, 'g-recaptcha') or contains(@class, 'h-captcha')]",
            "//*[contains(@id, 'rc-anchor-container') or contains(@class, 'recaptcha')]",
            "//input[contains(@name, 'captcha') or contains(@id, 'captcha')]",
            "//div[contains(@class, 'security-check') or contains(@id, 'security-check')]",
            "//div[contains(@class, 'human-verification') or contains(@id, 'human-verification')]",
        ]
        for selector in challenge_selectors:
            if tree.xpath(selector):
                challenge_score += 1
                break

        body_id = tree.xpath("//body/@id")
        body_classes = tree.xpath("//body/@class")
        body_attrs = " ".join(body_id + body_classes).lower()
        if any(token in body_attrs for token in ("captcha", "challenge", "security-check")):
            challenge_score += 1

        # Stripe pages without a recognizable pricing structure are likely challenge pages.
        if not self._has_pricing_structure(tree):
            challenge_score += 1
            login_score += 1

        login_form = tree.xpath(
            "//form[.//input[@type='password'] or contains(@action, 'signin') or contains(@action, 'login')]"
        )
        if login_form:
            login_score += 1
        if title and self._is_login_title(title):
            login_score += 1

        return title, challenge_score, login_score

    def _has_pricing_structure(self, tree: Any) -> bool:
        """Return True if the page contains recognizable pricing content."""
        text = " ".join(tree.itertext())
        signals = [
            "Standard pricing",
            "Custom pricing",
            "Pricing & Fees",
            "Preise",
            "Local payment methods",
            "Domestic card",
            "International card",
            "per transaction",
            "per successful",
        ]
        return any(signal.lower() in text.lower() for signal in signals)

    def _is_challenge_title(self, title: str) -> bool:
        lower = title.lower()
        return any(
            phrase in lower
            for phrase in (
                "captcha",
                "security check",
                "verify you are human",
                "robot check",
                "challenge",
                "are you human",
                "verify your identity",
            )
        )

    def _is_login_title(self, title: str) -> bool:
        lower = title.lower()
        return any(
            phrase in lower
            for phrase in (
                "log in",
                "login",
                "sign in",
                "signin",
                "account login",
            )
        )

    def _looks_like_login_page(self, text: str) -> bool:
        """Return True if the page contains a Stripe login form."""
        try:
            tree = html.fromstring(text)
        except Exception:
            return False
        return bool(
            tree.xpath(
                "//form[.//input[@type='password']]"
                "[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'stripe')]"
            )
        )

    def _calculate_backoff(self, attempt: int) -> float:
        base = 2.0**attempt
        jitter = random.uniform(0, 1)  # noqa: S311 # nosec B311
        return base + jitter

    def _default_headers(self) -> dict[str, str]:
        return {
            "User-Agent": self.config.user_agent or DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def _client_for_request(self) -> httpx.AsyncClient:
        """Create a fresh, isolated client for a single request.

        Each request uses its own cookie jar so cookies from one Stripe page
        cannot influence subsequent requests or markets.
        """
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.read_timeout,
            write=10.0,
            pool=10.0,
        )
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "follow_redirects": True,
            "headers": self._default_headers(),
            "cookies": httpx.Cookies(),
            "trust_env": False,
        }
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        return httpx.AsyncClient(**client_kwargs)

    def _cache_key(self, url: str, method: str) -> str:
        return cache_key(url, method)

    async def _request(
        self,
        method: str,
        url: str,
        cached: CachedSource | None = None,
        **kwargs: Any,
    ) -> HttpResponse:
        if self.config.offline_fixtures and url in self.config.offline_fixtures:
            fixture_path = self.config.offline_fixtures[url]
            with open(fixture_path, "rb") as fh:
                content = fh.read()
            return HttpResponse(
                url=url,
                status_code=200,
                content=content,
                text=content.decode("utf-8"),
                headers={"content-type": "text/html"},
            )
        self._validate_url(url)

        # Load a cached response when available. The cache is only used for GET
        # requests unless --no-cache is set.
        cache_lookup_key: str | None = None
        cached_entry: dict[str, Any] | None = None
        if method == "GET" and not self.config.no_cache:
            cache_lookup_key = self._cache_key(url, method)
            cached_entry = read_cache_entry(self.config, cache_lookup_key)

        headers: dict[str, str] = kwargs.pop("headers", {})
        # Use the most specific validators: the caller-supplied CachedSource,
        # then the on-disk cache entry, then none.
        etag = None
        last_modified = None
        if cached and cached.etag:
            etag = cached.etag
        elif cached_entry and cached_entry.get("etag"):
            etag = cached_entry["etag"]
        if cached and cached.last_modified:
            last_modified = cached.last_modified
        elif cached_entry and cached_entry.get("last_modified"):
            last_modified = cached_entry["last_modified"]

        force_revalidate = self.config.refresh_cache
        if cached_entry:
            force_revalidate = force_revalidate or requires_revalidation(cached_entry.get("headers", {}))
            if not is_fresh(self.config, cached_entry):
                force_revalidate = True
            elif not force_revalidate:
                # Fresh cache hit with no revalidation directive: return directly.
                return HttpResponse(
                    url=cached_entry["url"],
                    status_code=cached_entry["status_code"],
                    content=cached_entry["content"],
                    text=cached_entry["content"].decode("utf-8", errors="replace"),
                    headers=cached_entry["headers"],
                    etag=cached_entry.get("etag"),
                    last_modified=cached_entry.get("last_modified"),
                    from_cache=True,
                )

        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._semaphore:
                    logger.debug("%s %s (attempt %d)", method, _sanitize_url(url), attempt + 1)
                    client = self._client_for_request()
                    async with client:
                        response = await client.request(
                            method,
                            url,
                            headers=headers,
                            **kwargs,
                        )
                    final_url = str(response.url)
                    if final_url != url:
                        self._validate_url(final_url)
                    if len(response.content) > self.config.max_response_size:
                        raise ContentSecurityError(
                            f"Response size {len(response.content)} exceeds limit for {final_url}"
                        )
                    if response.status_code == 304:
                        if cached_entry and cache_lookup_key and not self.config.no_cache:
                            # Revalidation succeeded: refresh stored timestamp.
                            write_cache_entry(
                                self.config,
                                cache_lookup_key,
                                cached_entry["url"],
                                cached_entry["status_code"],
                                cached_entry["headers"],
                                cached_entry["content"],
                                etag=response.headers.get("etag") or cached_entry.get("etag"),
                                last_modified=response.headers.get("last-modified") or cached_entry.get("last_modified"),
                            )
                        return HttpResponse(
                            url=final_url,
                            status_code=304,
                            content=b"",
                            text="",
                            headers=dict(response.headers),
                            etag=response.headers.get("etag"),
                            last_modified=response.headers.get("last-modified"),
                            from_cache=True,
                        )
                    http_response = HttpResponse(
                        url=final_url,
                        status_code=response.status_code,
                        content=response.content,
                        text=response.text,
                        headers=dict(response.headers),
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                    )
                    self._detect_blocking_page(http_response)
                    if not self.config.no_cache and method == "GET" and should_persist_to_cache(http_response.headers):
                        if cache_lookup_key:
                            write_cache_entry(
                                self.config,
                                cache_lookup_key,
                                http_response.url,
                                http_response.status_code,
                                http_response.headers,
                                http_response.content,
                                etag=http_response.etag,
                                last_modified=http_response.last_modified,
                            )
                    elif cache_lookup_key:
                        remove_cache_entry(self.config, cache_lookup_key)
                    if self.config.request_delay > 0:
                        await asyncio.sleep(self.config.request_delay)
                    return http_response
            except (TransientNetworkError, httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise NetworkError(f"Failed after {attempt + 1} attempts: {url}") from exc
                retry_after = 0.0
                if isinstance(exc, TransientNetworkError) and exc.retry_after is not None:
                    try:
                        retry_after = float(exc.retry_after)
                    except ValueError:
                        retry_after = 0.0
                delay = max(retry_after, self._calculate_backoff(attempt))
                logger.warning("Transient error for %s, retrying in %.2fs: %s", _sanitize_url(url), delay, exc)
                await asyncio.sleep(delay)
            except (ContentSecurityError, PermanentNetworkError):
                raise
        if last_error:
            raise NetworkError(f"Failed to request {url}") from last_error
        raise NetworkError(f"Unexpected end of retries for {url}")

    async def get(self, url: str, cached: CachedSource | None = None) -> HttpResponse:
        return await self._request("GET", url, cached=cached)

    async def head(self, url: str) -> HttpResponse:
        return await self._request("HEAD", url)
