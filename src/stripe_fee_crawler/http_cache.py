"""Persistent 24-hour HTTP cache with per-request cookie isolation helpers."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .models import CrawlConfiguration

logger = logging.getLogger(__name__)

CACHE_VERSION = "1"
_CACHE_SAFE_QUERY_PARAMS = {
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


def _strip_safe_query_params(url: str) -> str:
    """Return a URL with known tracking-only query parameters removed."""
    parsed = urlparse(url)
    if not parsed.query:
        return url
    pairs = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in _CACHE_SAFE_QUERY_PARAMS
    ]
    query = urlencode(pairs)
    return urlunparse(parsed._replace(query=query))


def cache_key(url: str, method: str = "GET") -> str:
    """Stable cache key for a request.

    The key includes the method, the URL with tracking parameters removed, and
    a cache version so incompatible formats are invalidated on crawler updates.
    """
    normalized = _strip_safe_query_params(url)
    text = f"{CACHE_VERSION}:{method}:{normalized}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_path(config: CrawlConfiguration, key: str) -> Path:
    """Path to an on-disk cache entry."""
    cache_dir = Path(config.cache_dir) if config.cache_dir else Path(".cache") / "stripe-fee-crawler" / "http"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _decode_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Decode a cached entry from JSON, turning base64 content back to bytes."""
    content_b64 = entry.get("content_b64")
    content = base64.b64decode(content_b64) if content_b64 else b""
    return {**entry, "content": content}


def read_cache_entry(config: CrawlConfiguration, key: str) -> dict[str, Any] | None:
    """Read a cached response, or None if missing or unreadable."""
    path = cache_path(config, key)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Cache read failed for %s: %s", key, exc)
        return None
    return _decode_entry(raw)


def write_cache_entry(
    config: CrawlConfiguration,
    key: str,
    url: str,
    status_code: int,
    headers: dict[str, str],
    content: bytes,
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    """Persist a response to the on-disk cache."""
    path = cache_path(config, key)
    entry = {
        "version": CACHE_VERSION,
        "stored_at": time.time(),
        "url": url,
        "status_code": status_code,
        "headers": dict(headers),
        "content_b64": base64.b64encode(content).decode("ascii"),
        "etag": etag,
        "last_modified": last_modified,
    }
    try:
        path.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.debug("Cache write failed for %s: %s", key, exc)


def remove_cache_entry(config: CrawlConfiguration, key: str) -> None:
    """Remove a cached entry, if present."""
    path = cache_path(config, key)
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Cache removal failed for %s: %s", key, exc)


def is_fresh(config: CrawlConfiguration, entry: dict[str, Any]) -> bool:
    """Return True when the cached entry is still within its TTL."""
    if config.refresh_cache or config.no_cache:
        return False
    stored_at = entry.get("stored_at")
    if not stored_at:
        return False
    ttl = config.cache_ttl_hours * 3600
    return time.time() - stored_at < ttl


def cache_control_headers(headers: dict[str, str]) -> dict[str, str]:
    """Extract cache-control directives into a normalized dict."""
    value = headers.get("cache-control", "").lower()
    directives: dict[str, str] = {}
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            directives[k.strip()] = v.strip()
        else:
            directives[part] = ""
    return directives


def should_persist_to_cache(response_headers: dict[str, str]) -> bool:
    """Return True when a response may be stored on disk."""
    cc = cache_control_headers(response_headers)
    return not ("no-store" in cc or "private" in cc)


def requires_revalidation(response_headers: dict[str, str]) -> bool:
    """Return True when a cached response must be revalidated before reuse."""
    cc = cache_control_headers(response_headers)
    return bool(cc.get("no-cache") or cc.get("max-age") == "0")
