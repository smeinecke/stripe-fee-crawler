"""Persistent 24-hour HTTP response cache for the Stripe fee crawler.

The cache stores complete successful HTTP responses on disk, keyed by the
effective request identity (URL, market, locale, content-negotiation headers and
a crawler-specific cache version). It supports freshness checks, conditional
revalidation, atomic writes, and per-key async locking so multiple workers never
download the same unchanged resource twice.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .market_detection import detect_market
from .models import CacheStats, CrawlConfiguration

logger = logging.getLogger(__name__)

CACHE_VERSION = "2"


def _default_cache_dir() -> Path:
    """Return the persistent default cache directory.

    Resolves ``${XDG_CACHE_HOME:-$HOME/.cache}/stripe-fee-crawler/http``,
    expanding both ``~`` and environment variables.
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        base = os.path.expandvars(os.path.expanduser(xdg_cache))
    else:
        home = os.environ.get("HOME") or os.path.expanduser("~")
        base = os.path.join(os.path.expandvars(os.path.expanduser(home)), ".cache")
    return Path(base) / "stripe-fee-crawler" / "http"


def _resolve_cache_dir(cache_dir: str | None) -> Path | None:
    """Resolve a cache directory string, defaulting to the persistent XDG path."""
    if cache_dir:
        return Path(os.path.expandvars(os.path.expanduser(cache_dir)))
    return _default_cache_dir()


# Content-negotiation headers that can change which market/language is served.
_NEGOTIATION_HEADERS = {"accept", "accept-language", "accept-encoding", "accept-charset"}

# Query parameters that are known to be safe tracking tokens and do not affect
# the response body, market selection or authorization. All other query
# parameters are included in the cache key by default so that sensitive or
# content-affecting values never collide.
_VOLATILE_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "gclid",
    "fbclid",
    "ref",
    "referral",
}

# Response headers considered relevant to store and replay.
_CACHED_RESPONSE_HEADERS = {
    "content-type",
    "content-language",
    "content-length",
    "etag",
    "last-modified",
    "cache-control",
    "expires",
}


def _parse_cache_control(header: str | None) -> dict[str, str | None]:
    """Return a dict of Cache-Control directive names to optional values.

    Tokens such as ``no-store`` and ``private`` map to ``None``; directives with
    values such as ``max-age=0`` map to their value as a string.
    """
    if not header:
        return {}
    directives: dict[str, str | None] = {}
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            directives[key.strip().lower()] = value.strip().strip('"')
        else:
            directives[part.lower()] = None
    return directives


def _normalize_url(url: str) -> str:
    """Return a stable, normalized URL with sorted query parameters."""
    parsed = urlparse(url)
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in _VOLATILE_PARAMS]
    normalized_query = urlencode(sorted(pairs))
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            "",
            normalized_query,
            "",
        )
    )


def _cache_key(method: str, url: str, market: str | None, locale: str | None, headers: dict[str, str]) -> str:
    """Return a stable SHA-256 hash representing the cache key."""
    headers_lower = {k.lower(): v for k, v in headers.items()}
    identity: dict[str, Any] = {
        "v": CACHE_VERSION,
        "method": method.upper(),
        "url": _normalize_url(url),
        "accept": headers_lower.get("accept"),
        "accept_language": headers_lower.get("accept-language"),
    }
    if market:
        identity["market"] = market.upper()
    if locale:
        identity["locale"] = locale

    serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _filter_response_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    """Return only the response headers we want to replay from the cache."""
    if isinstance(headers, httpx.Headers):
        headers = dict(headers)
    return {k.lower(): v for k, v in headers.items() if k.lower() in _CACHED_RESPONSE_HEADERS and v is not None}


def _is_valid_cacheable_response(response: httpx.Response) -> bool:
    """Return True if a 200 response has a complete, cacheable body."""
    if response.status_code != 200:
        return False
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            expected = int(content_length)
        except ValueError:
            return False
        if len(response.content) != expected:
            logger.debug("Response content-length mismatch (%s vs %s); not caching", expected, len(response.content))
            return False
    return True


@dataclass
class _CacheEntry:
    """On-disk cache entry (serialized to JSON)."""

    key: str
    url: str
    final_url: str | None
    status_code: int
    headers: dict[str, str]
    content: bytes
    etag: str | None
    last_modified: str | None
    fetched_at: float
    cache_version: str
    market: str | None
    locale: str | None
    detected_market: str | None
    detected_locale: str | None
    cache_control: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "v": self.cache_version,
            "key": self.key,
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "headers": self.headers,
            "content": base64.b64encode(self.content).decode("ascii"),
            "etag": self.etag,
            "last_modified": self.last_modified,
            "fetched_at": self.fetched_at,
            "market": self.market,
            "locale": self.locale,
            "detected_market": self.detected_market,
            "detected_locale": self.detected_locale,
            "cache_control": self.cache_control,
        }

    def to_httpx_response(self, method: str) -> httpx.Response:
        """Replay this entry as an ``httpx.Response``."""
        return httpx.Response(
            self.status_code,
            content=self.content,
            headers=self.headers,
            request=httpx.Request(method, self.final_url or self.url, headers={}),
        )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _CacheEntry | None:
        try:
            version = data.get("v")
            if version != CACHE_VERSION:
                return None
            content = base64.b64decode(data.get("content", ""))
            return cls(
                key=data.get("key", ""),
                url=data["url"],
                final_url=data.get("final_url"),
                status_code=int(data["status_code"]),
                headers=data.get("headers", {}),
                content=content,
                etag=data.get("etag"),
                last_modified=data.get("last_modified"),
                fetched_at=float(data["fetched_at"]),
                cache_version=version,
                market=data.get("market"),
                locale=data.get("locale"),
                detected_market=data.get("detected_market"),
                detected_locale=data.get("detected_locale"),
                cache_control=data.get("cache_control"),
            )
        except Exception:
            return None


class _FileLock:
    """Thin asyncio-friendly wrapper around ``fcntl.flock``.

    ``filelock`` is intentionally not added as a dependency; ``fcntl`` is part of
    the standard library on the POSIX systems used to run the crawler. The lock
    is released when the context manager exits.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: int | None = None

    @contextlib.asynccontextmanager
    async def acquire(self):
        """Acquire the file lock, yielding self."""
        self._fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            await asyncio.to_thread(fcntl.flock, self._fd, fcntl.LOCK_EX)
            try:
                yield self
            finally:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None


class HttpCache:
    """Persistent on-disk HTTP response cache."""

    def __init__(self, config: CrawlConfiguration) -> None:
        self.config = config
        self._configured_dir = _resolve_cache_dir(config.cache_dir)
        self._enabled = self._configured_dir is not None and not config.no_cache
        self._cache_dir = self._configured_dir if self._enabled else None
        self._ttl_seconds = config.cache_ttl_hours * 3600.0
        self._cache_policy = config.cache_policy
        self._refresh = config.refresh_cache
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._file_locks: dict[str, _FileLock] = {}
        self.stats = CacheStats(
            cache_enabled=self._enabled,
            cache_dir=str(self._configured_dir) if self._configured_dir else None,
            cache_ttl_hours=config.cache_ttl_hours,
            cache_policy=config.cache_policy,
        )

    def _key(self, method: str, url: str, market: str | None, locale: str | None, headers: dict[str, str]) -> str:
        return _cache_key(method, url, market, locale, headers)

    def _entry_path(self, key: str) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("Cache directory is not configured")
        return self._cache_dir / "entries" / key[:2] / f"{key}.json"

    def _lock_path(self, key: str) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("Cache directory is not configured")
        return self._cache_dir / "locks" / key[:2] / f"{key}.lock"

    def _key_lock(self, key: str) -> asyncio.Lock:
        if key not in self._key_locks:
            self._key_locks[key] = asyncio.Lock()
        return self._key_locks[key]

    def _file_lock(self, key: str) -> _FileLock:
        if key not in self._file_locks:
            self._file_locks[key] = _FileLock(self._lock_path(key))
        return self._file_locks[key]

    @staticmethod
    def _locale_country(locale: str | None) -> str | None:
        """Return the country portion of a locale tag, e.g. 'en-us' -> 'us'."""
        if not locale:
            return None
        parts = locale.lower().split("-")
        return parts[-1] if len(parts) > 1 else None

    def _is_market_match(self, entry: _CacheEntry, market: str | None, locale: str | None) -> bool:
        """Return True when the cached response was served for the requested market."""
        if not entry.detected_market or not market:
            return True
        if entry.detected_market.upper() != market.upper():
            return False
        requested_country = self._locale_country(locale)
        cached_country = self._locale_country(entry.detected_locale)
        return not (requested_country and cached_country and requested_country != cached_country)

    def _is_fresh(self, entry: _CacheEntry) -> bool:
        """Return True if the stored entry may be served without revalidation.

        In ``ttl`` policy, the crawler's configured snapshot TTL controls reuse
        for public pricing pages. Origin ``no-cache``, ``max-age=0``, and shorter
        positive ``max-age`` values are intentionally ignored so that repeated
        regenerations within the local TTL do not revalidate on every run.

        In ``http`` policy, origin directives are respected normally (``no-cache``
        and ``max-age=0`` require revalidation, positive ``max-age`` is an upper
        bound, and ``Expires`` is consulted when present).
        """
        age = time.time() - entry.fetched_at
        if age >= self._ttl_seconds:
            return False

        if self._cache_policy == "http":
            directives = _parse_cache_control(entry.cache_control)

            if "no-cache" in directives:
                return False

            max_age_value = directives.get("s-maxage") or directives.get("max-age")
            if max_age_value is not None:
                try:
                    max_age = int(max_age_value)
                except ValueError:
                    max_age = None
                if max_age is not None:
                    if max_age == 0:
                        return False
                    return age < min(max_age, self._ttl_seconds)

            expires = entry.headers.get("expires")
            if expires:
                try:
                    expires_ts = parsedate_to_datetime(expires).timestamp()
                    return time.time() < expires_ts
                except Exception:  # nosec B110
                    pass

        return True

    def _read_entry(self, key: str) -> _CacheEntry | None:
        if self._cache_dir is None:
            return None
        path = self._entry_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Corrupt cache entry at %s; ignoring", path)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            return None
        entry = _CacheEntry.from_json(data)
        if entry is None:
            logger.debug("Stale or invalid cache entry at %s; ignoring", path)
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            return None
        return entry

    def _write_entry(self, entry: _CacheEntry) -> bool:
        """Write a cache entry to disk. Return True on success, False on failure."""
        if self._cache_dir is None:
            return False
        path = self._entry_path(entry.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".tmp.{path.name}.{os.getpid()}.{time.monotonic_ns()}")
        try:
            tmp.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to write cache entry %s: %s", path, exc)
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return False
        return True

    def _remove_entry(self, key: str) -> None:
        if self._cache_dir is None:
            return
        path = self._entry_path(key)
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    def _revalidation_headers(self, entry: _CacheEntry | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if entry is None:
            return headers
        if entry.etag:
            headers["If-None-Match"] = entry.etag
        if entry.last_modified:
            headers["If-Modified-Since"] = entry.last_modified
        return headers

    async def fetch(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        network: Any,
        *,
        market: str | None = None,
        locale: str | None = None,
    ) -> tuple[httpx.Response, bool]:
        """Return a response and a flag indicating whether it came from the cache.

        ``network`` must be an awaitable callable that accepts the full request
        headers dict and a dict of conditional request headers and returns an
        ``httpx.Response``.
        """
        request_headers = {**headers}
        if locale:
            request_headers["Accept-Language"] = locale

        async def _do_network(req_headers: dict[str, str], reval_headers: dict[str, str]) -> httpx.Response:
            self.stats.network_requests += 1
            return await network(req_headers, reval_headers)

        if method.upper() != "GET":
            reval = self._revalidation_headers(None)
            response = await _do_network(request_headers, reval)
            return response, False

        if not self._enabled:
            reval = self._revalidation_headers(None)
            response = await _do_network(request_headers, reval)
            return response, False

        key = self._key(method, url, market, locale, request_headers)
        async with self._key_lock(key):
            lock = self._file_lock(key)
            async with lock.acquire():
                cache_path = self._entry_path(key)
                path_existed = cache_path.exists()
                entry = await asyncio.to_thread(self._read_entry, key)
                if entry is None and path_existed:
                    self.stats.cache_errors += 1

                if entry is not None and not self._is_market_match(entry, market, locale):
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning(
                            "Cached response for %s served market %s/%s but requested %s/%s; invalidating",
                            _normalize_url(url),
                            entry.detected_market,
                            entry.detected_locale,
                            market,
                            locale,
                        )
                    await asyncio.to_thread(self._remove_entry, key)
                    entry = None

                if entry is not None and not self._refresh and self._is_fresh(entry):
                    self.stats.cache_hits += 1
                    self.stats.bytes_avoided += len(entry.content)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Cache hit for %s", _normalize_url(url))
                    return entry.to_httpx_response(method), True

                reval = self._revalidation_headers(entry)
                if reval:
                    self.stats.cache_revalidations += 1
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Cache revalidation for %s", _normalize_url(url))

                response = await _do_network(request_headers, reval)

                if response.status_code == 304:
                    if entry is not None:
                        entry.fetched_at = time.time()
                        if not await asyncio.to_thread(self._write_entry, entry):
                            self.stats.cache_errors += 1
                        else:
                            self.stats.cache_304_responses += 1
                            self.stats.bytes_avoided += len(entry.content)
                        return entry.to_httpx_response(method), True
                    # No stored body for this 304; return the upstream response.
                    return response, False

                directives = _parse_cache_control(response.headers.get("cache-control"))

                # Responses marked no-store or private must not be persisted,
                # and any previously stored copy for the same resource is removed
                # so it cannot be served again.
                if "no-store" in directives or "private" in directives:
                    await asyncio.to_thread(self._remove_entry, key)
                    self.stats.cache_misses += 1
                    return response, False

                if _is_valid_cacheable_response(response):
                    final_url = str(response.url)
                    detection = detect_market(response.text, final_url)
                    entry = _CacheEntry(
                        key=key,
                        url=url,
                        final_url=final_url,
                        status_code=response.status_code,
                        headers=_filter_response_headers(response.headers),
                        content=response.content,
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                        fetched_at=time.time(),
                        cache_version=CACHE_VERSION,
                        market=market,
                        locale=locale,
                        detected_market=detection.get("detected_market"),
                        detected_locale=detection.get("detected_locale"),
                        cache_control=response.headers.get("cache-control"),
                    )
                    if await asyncio.to_thread(self._write_entry, entry):
                        self.stats.cache_writes += 1
                    else:
                        self.stats.cache_errors += 1
                    self.stats.cache_misses += 1
                    return response, False

                # Not a cacheable 200; pass through without writing.
                self.stats.cache_misses += 1
                return response, False
