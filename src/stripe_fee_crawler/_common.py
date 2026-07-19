"""Small, cross-module helpers with no heavy dependencies."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from .models import FeeCondition

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Any:
    """Read and parse JSON from ``path``, returning ``None`` on any error."""
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _condition_key(conditions: list[FeeCondition]) -> tuple[tuple[str, str, Any], ...]:
    """Build a sorted, hashable identity key from a list of conditions."""
    return tuple(sorted((c.dimension, c.operator, str(c.value)) for c in conditions))


def _condition_key_data(conditions: list[dict[str, Any]]) -> tuple[tuple[str, str, str], ...]:
    """Build a sorted, hashable identity key from raw JSON condition dictionaries."""
    return tuple(sorted((str(c.get("dimension")), str(c.get("operator")), str(c.get("value"))) for c in conditions))


_MARKET_SHARE_PHRASES: tuple[str, ...] = (
    "share",
    "share of online payments",
    "share of online transactions",
    "share of e-commerce payments",
    "market share",
    "most popular payment method",
    "used in more than",
    "used by over",
    "adoption",
    "customers use",
    "active monthly users",
    "active global customers",
)

_MARKET_SHARE_REGEX = re.compile(r"\b(more than|over)\s+[0-9]+%?\s*(share|of)\b")


def _contains_market_share_evidence(text: str) -> bool:
    """Return True when ``text`` contains market-share or adoption wording."""
    lower = text.lower()
    return any(phrase in lower for phrase in _MARKET_SHARE_PHRASES) or bool(_MARKET_SHARE_REGEX.search(lower))
