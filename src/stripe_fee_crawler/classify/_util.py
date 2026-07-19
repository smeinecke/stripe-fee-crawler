"""Shared classification helpers."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from ..currencies import CURRENCY_CODES, CURRENCY_SYMBOLS, currency_exponent
from ..models import PricingEntry
from ..pricing_tokens import parse_fee_value
from .tables import _POSITIVE_FEE_TERMS, _Per_UNIT_NOUNS

logger = logging.getLogger(__name__)


def _fixed_amount_minor(amount: str, currency: str) -> str | None:
    """Convert a major-unit amount to minor units for a currency."""
    try:
        dec = Decimal(amount)
        exponent = currency_exponent(currency)
        multiplier = Decimal(10) ** exponent
        minor = int(dec * multiplier)
        return str(minor)
    except Exception:
        return None


def _text_has(text: str, *terms: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _text_has_lower(lower: str, *terms: str) -> bool:
    """Variant for callers that have already lowercased the haystack."""
    return any(term in lower for term in terms)


def _first_match(text: str, table: tuple[tuple[Any, str], ...]) -> str | None:
    """Return the first value whose keyword/key pattern matches ``text``.

    A table entry may be:
    * a string (substring match),
    * a tuple of strings (any substring match),
    * a frozenset of strings (all substrings must be present),
    * or a compiled ``re.Pattern`` (regex search).
    """
    lower = text.lower()
    for key, value in table:
        if isinstance(key, str):
            if key in lower:
                return value
        elif isinstance(key, tuple):
            if any(k in lower for k in key):
                return value
        elif isinstance(key, frozenset):
            if all(k in lower for k in key):
                return value
        elif isinstance(key, re.Pattern) and key.search(lower):
            return value
    return None


def _source_text_for_group(group: list[dict[str, Any]]) -> str:
    """Combine all source texts and section headings for the group."""
    parts: list[str] = []
    for item in group:
        entry = item["entry"]
        parts.extend(entry.section_path)
        parts.append(entry.source_text)
    return " ".join(p for p in parts if p).lower()


def _is_explicit_fee_phrase(text: str) -> bool:
    """Return True when the text contains explicit fee-calculation language."""
    return _text_has(text, *_POSITIVE_FEE_TERMS)


def _nearest_heading_line(entry: PricingEntry) -> str | None:
    """Return the nearest preceding heading from row-level evidence.

    Many Stripe tables emit the price as a separate text node and put the
    feature name (e.g. "Visa resolution") in the surrounding container.  We
    look for the line immediately before the first price-containing line
    that is a short, non-price label.
    """
    evidence = entry.source_evidence or ""
    if not evidence:
        return None
    # Look for the heading immediately before the entry's first price token.
    first_number_match = re.search(r"\d+(?:\.\d+)?", entry.source_text)
    if not first_number_match:
        return None
    needle = first_number_match.group(0)
    lines = evidence.splitlines()
    heading: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if needle in stripped and _looks_like_price_line(stripped.lower()):
            return heading
        if not _looks_like_price_line(stripped.lower()) and len(stripped.split()) <= 8:
            heading = stripped
    return heading


def _per_completion_noun(entry: PricingEntry) -> str | None:
    """Return a unit noun that completes a dangling 'per' in the source text."""
    source = entry.source_text.lower().rstrip()
    if not source.endswith(" per"):
        return None
    heading = _nearest_heading_line(entry)
    if not heading:
        return None
    words = heading.lower().split()
    # Prefer the last word of the heading (e.g. "Visa resolution" -> "resolution").
    for word in reversed(words):
        clean = re.sub(r"[^a-z0-9]", "", word)
        if clean in _Per_UNIT_NOUNS:
            return clean
    return None


def _is_unsupported_multi_per_shape(entry: PricingEntry, unit: str | None) -> bool:
    """Detect fee shapes with stacked per-unit dimensions that cannot be modelled."""
    text = entry.source_text.lower()
    # Capture up to three words after each "per" to distinguish repeated units
    # ("per successful charge ... per successful charge") from stacked
    # dimensions ("per institution per account holder per month").
    per_phrases = re.findall(r"\bper\b\s+((?:(?!\bper\b)[a-z0-9]+\s+){0,2}(?!\bper\b)[a-z0-9]+)", text)
    per_phrases = [p.strip() for p in per_phrases]
    if len(per_phrases) >= 3 and len(set(per_phrases)) > 1:
        return True
    if len(per_phrases) == 2 and len(set(per_phrases)) > 1:
        # Two distinct per-units are only supported when one is a recognised
        # time/fee unit (e.g. "per active user per month").
        time_units = {"month", "year", "day", "transaction", "charge", "dispute", "refund", "payout", "invoice"}
        if not any(any(t in p.split() for t in time_units) for p in per_phrases):
            return True
    return False


def _looks_like_price_line(line: str) -> bool:
    """Return True when a line is a price fragment rather than a heading."""
    if not line:
        return True
    if line in {"per", "month", "year", "transaction", "transactions", "learn more", "compare plans"}:
        return True
    stripped = line.strip()
    if re.fullmatch(r"[\d\s,.]+", stripped):
        return True
    # A heading is not just a currency code/symbol and/or an amount.
    currency_parts = sorted(set(CURRENCY_CODES) | set(CURRENCY_SYMBOLS), key=len, reverse=True)
    currency_pattern = "|".join(re.escape(c) for c in currency_parts)
    price_pattern = rf"(?i)^\s*(?:{currency_pattern})?\s*[\d\s,.]*\s*(?:{currency_pattern})?\s*(?:per)?\s*$"
    return bool(re.fullmatch(price_pattern, stripped))


def _heading_to_snake_case(heading: str) -> str:
    """Normalize a heading like 'Visa resolution' to 'visa_resolution'."""
    heading = re.sub(r"[^\w\s]", "", heading)
    return "_".join(heading.lower().split())


def _is_tap_to_pay(entry: PricingEntry) -> bool:
    return "tap to pay" in entry.source_text.lower() or entry.payment_method == "tap_to_pay"


def _dedup_repeated_phrases(text: str) -> str:
    """Remove consecutive duplicate word n-grams while preserving order.

    This collapses artefacts such as "per successful charge per successful charge"
    or "for domestic cards for domestic cards" that arise when a qualifier is
    repeated across merged fragments.
    """
    words = text.split()
    result: list[str] = []
    i = 0
    max_n = min(8, len(words) // 2) if len(words) > 4 else 3
    while i < len(words):
        duplicated = False
        for n in range(max_n, 1, -1):
            if i + n <= len(words) and result[-n:] == words[i : i + n]:
                i += n
                duplicated = True
                break
        if not duplicated:
            result.append(words[i])
            i += 1
    return " ".join(result)


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _has_base_fee(entry: PricingEntry, parsed: dict[str, Any] | None = None) -> bool:
    if parsed is None:
        parsed = parse_fee_value(entry.source_text)
    return bool(parsed.get("percentage") or parsed.get("fixed_amount"))


def _account_country_from_url(url: str) -> str | None:
    """Infer the account country from a Stripe pricing URL path."""
    match = re.search(r"/(?:([a-z]{2})-([a-z]{2})|([a-z]{2}))/pricing", url.lower())
    if match:
        return (match.group(2) or match.group(3) or "").upper()
    if "/pricing" in url.lower() and "stripe.com/pricing" in url.lower():
        return "US"
    return None


__all__ = [
    "_fixed_amount_minor",
    "_text_has",
    "_first_match",
    "_source_text_for_group",
    "_is_explicit_fee_phrase",
    "_nearest_heading_line",
    "_per_completion_noun",
    "_is_unsupported_multi_per_shape",
    "_looks_like_price_line",
    "_heading_to_snake_case",
    "_is_tap_to_pay",
    "_dedup_repeated_phrases",
    "_ordered_unique",
    "_has_base_fee",
    "_account_country_from_url",
]
