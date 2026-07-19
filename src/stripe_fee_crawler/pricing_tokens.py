"""Pricing text tokenization and normalization."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .currencies import (
    CURRENCY_CODES,
    CURRENCY_SYMBOLS,
    MINOR_CURRENCY_SYMBOLS,
)
from .models import FeeToken
from .rich_text import clean_fee_text

EXACTNESS_MARKERS: dict[str, str] = {
    "from": "from",
    "starting at": "from",
    "starting from": "from",
    "up to": "range",
    "capped at": "range",
    "cap at": "range",
    "maximum": "range",
    "max": "range",
    "minimum": "from",
    "min": "from",
    "tiered": "tiered",
    "volume based": "tiered",
    "custom": "custom",
    "contact sales": "custom",
    "included": "included",
    "free": "free",
    "no additional fee": "included",
    "no additional charge": "included",
    "at no additional charge": "included",
    "no fee": "free",
    "at no cost": "free",
    "no cost": "free",
}

_MAJOR_SYMBOLS = frozenset(CURRENCY_SYMBOLS) - MINOR_CURRENCY_SYMBOLS
_SYMBOL_PATTERN_MAJOR = "|".join(re.escape(s) for s in sorted(_MAJOR_SYMBOLS, key=len, reverse=True))
_SYMBOL_PATTERN_MINOR = "|".join(re.escape(s) for s in sorted(MINOR_CURRENCY_SYMBOLS, key=len, reverse=True))
_SYMBOL_PATTERN_ALL = "|".join(re.escape(s) for s in sorted(CURRENCY_SYMBOLS, key=len, reverse=True))
_CODE_PATTERN = "|".join(re.escape(c) for c in CURRENCY_CODES)

_AMOUNT_GROUP = r"(?P<amount>[0-9][0-9\s,.]*)"
_AMOUNT_GROUP_NB = r"(?P<amount>[0-9][0-9,.]*)"

_CURRENCY_AND_AMOUNT_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("symbol_prefix", re.compile(rf"(?P<symbol>{_SYMBOL_PATTERN_MAJOR})\s*{_AMOUNT_GROUP_NB}"), "symbol"),
    ("symbol_suffix", re.compile(rf"{_AMOUNT_GROUP}\s*(?P<symbol>{_SYMBOL_PATTERN_ALL})(?!\w)"), "symbol"),
    ("code_prefix", re.compile(rf"(?P<code>{_CODE_PATTERN})\s*{_AMOUNT_GROUP_NB}"), "code"),
    ("code_suffix", re.compile(rf"{_AMOUNT_GROUP}\s*(?P<code>{_CODE_PATTERN})(?!\w)"), "code"),
    (
        "symbol_minor_tight",
        re.compile(rf"(?P<amount>[0-9][0-9,.]*)(?P<symbol>{_SYMBOL_PATTERN_MINOR})(?!\w)"),
        "symbol",
    ),
]

_EXACTNESS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(marker) + r"\b"), exactness) for marker, exactness in EXACTNESS_MARKERS.items()
]


def _normalize_for_parsing(text: str) -> str:
    """Replace common separators and normalize spaces."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"([0-9])\s*%", r"\1%", text)
    # Currency-symbol spacing is handled by regex; do not collapse here because
    # multi-character symbols such as "A$" are parsed correctly with \s*.
    return text


def _parse_decimal(text: str) -> Decimal | None:
    """Parse a decimal string respecting localized separators."""
    text = text.strip().replace("\xa0", " ").replace(" ", "")
    if not text:
        return None
    if text.isdigit():
        return Decimal(text)

    # First handle the simple cases where only one separator type appears.
    last_dot = text.rfind(".")
    last_comma = text.rfind(",")

    if last_dot == -1 and last_comma == -1:
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    # When both separators are present, the rightmost is the decimal mark.
    if last_dot != -1 and last_comma != -1:
        text = text.replace(",", "") if last_dot > last_comma else text.replace(".", "").replace(",", ".")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    sep = "." if last_dot != -1 else ","
    parts = text.split(sep)

    # Recognize thousands groupings: every group except the first has exactly
    # three digits and the first has one to three digits.
    if all(p.isdigit() for p in parts) and 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
        try:
            return Decimal("".join(parts))
        except InvalidOperation:
            return None

    # A trailing group of one or two digits means the separator is a decimal mark.
    if len(parts[-1]) <= 2 and parts[-1].isdigit():
        if sep == ",":
            text = text.replace(".", "").replace(",", ".")
        try:
            return Decimal(text)
        except InvalidOperation:
            return None

    # Fallback: remove the separator and parse as a whole number.
    text = text.replace(sep, "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _to_basis_points(percentage: Decimal) -> str:
    """Convert a percentage decimal to basis points string."""
    return str(percentage * Decimal("100"))


def _currency_for_symbol(symbol: str) -> str | None:
    return CURRENCY_SYMBOLS.get(symbol)


def _normalize_minor_amount(amount: Decimal, amount_text: str, symbol: str) -> tuple[Decimal, bool]:
    """Convert minor-currency symbols (¢, p) to major units when appropriate."""
    if symbol not in MINOR_CURRENCY_SYMBOLS:
        return amount, False
    # If the raw amount text already contains a decimal separator, assume it is
    # already in major units and should not be divided again.
    if "." in amount_text or "," in amount_text:
        return amount, True
    return amount / Decimal("100"), True


def _extract_currency_and_amount(text: str) -> list[dict[str, Any]]:
    """Extract all currency-amount pairs from a fee phrase.

    Supports prefix/suffix symbols and ISO codes, optional whitespace, and
    minor-currency symbols such as ``20p`` (0.20 GBP) and ``5¢`` (0.05 USD).
    Requires word boundaries so product codes like ``P24`` or prose like
    ``from the P24 portal`` are not parsed as amounts, and skips marketing
    magnitudes such as ``$1 billion``.
    """
    seen_spans: list[tuple[int, int]] = []
    results: list[tuple[int, int, dict[str, Any]]] = []
    marketing_magnitudes = ("billion", "million", "trillion")

    def _char_at(idx: int) -> str | None:
        if 0 <= idx < len(text):
            return text[idx]
        return None

    for _match_type, pattern, group in _CURRENCY_AND_AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            # Skip overlaps with already-selected matches.
            if any(span[0] < end and span[1] > start for start, end in seen_spans):
                continue
            symbol_or_code = match.group(group)
            raw_amount = match.group("amount")
            amount_text = raw_amount.replace(" ", "")
            amount = _parse_decimal(amount_text)
            if amount is None:
                continue

            # Reject matches that sit inside a longer alphanumeric token.
            before = _char_at(span[0] - 1) if span[0] > 0 else None
            after = _char_at(span[1]) if span[1] < len(text) else None
            amount_start = match.start("amount")
            amount_before = _char_at(amount_start - 1) if amount_start > 0 else None
            is_prefix = _match_type in {"symbol_prefix", "code_prefix"}
            if is_prefix and before is not None and before.isalnum():
                # e.g. "US$" matched as "$"
                continue
            if not is_prefix:
                if amount_before is not None and amount_before.isalnum():
                    # e.g. "P24" -> amount 24 preceded by P
                    continue
                if after is not None and after.isalnum():
                    # e.g. "24 portal" matched as "24 p"
                    continue

            # Reject marketing magnitudes: "$1 billion", "€ 2 million", etc.
            stripped_len = len(raw_amount.rstrip())
            amount_end = amount_start + stripped_len
            tail = text[amount_end:].lstrip()
            if tail.lower().startswith(marketing_magnitudes):
                continue

            currency = _currency_for_symbol(symbol_or_code) if group == "symbol" else symbol_or_code
            is_minor = False
            if group == "symbol" and symbol_or_code in MINOR_CURRENCY_SYMBOLS:
                amount, is_minor = _normalize_minor_amount(amount, amount_text, symbol_or_code)
            results.append(
                (
                    span[0],
                    span[1],
                    {
                        "symbol": symbol_or_code,
                        "currency": currency,
                        "amount_text": amount_text,
                        "amount": str(amount),
                        "is_minor": is_minor,
                    },
                )
            )
            seen_spans.append(span)

    results.sort(key=lambda r: (r[0], r[1] - r[0]), reverse=True)
    deduped: list[dict[str, Any]] = []
    used: set[tuple[int, int]] = set()
    for start, end, data in results:
        if (start, end) in used:
            continue
        if any(start < u_end and end > u_start for u_start, u_end in used):
            continue
        used.add((start, end))
        deduped.append(data)
    deduped.reverse()
    return deduped


def _extract_percentage(text: str) -> list[dict[str, Any]]:
    """Extract all percentages from a fee phrase."""
    results: list[dict[str, Any]] = []
    for match in re.finditer(r"(?P<value>[0-9]+(?:[.,][0-9]+)?)\s*%", text):
        value_text = match.group("value").replace(" ", "")
        value = _parse_decimal(value_text)
        if value is None:
            continue
        results.append(
            {
                "percentage_text": value_text,
                "percentage": str(value),
                "basis_points": _to_basis_points(value),
            }
        )
    return results


def _extract_exactness(text: str) -> str | None:
    lower = text.lower()
    for pattern, exactness in _EXACTNESS_PATTERNS:
        if pattern.search(lower):
            return exactness
    return None


def _extract_operators(text: str) -> list[str]:
    lower = text.lower()
    operators: list[str] = []
    if "+" in text:
        operators.append("+")
    if "if" in lower:
        operators.append("if")
    if "for" in lower:
        operators.append("for")
    if "per" in lower:
        operators.append("per")
    return operators


def tokenize_fee_text(text: str) -> list[FeeToken]:
    """Tokenize a fee phrase into normalized amount/percentage tokens."""
    text = clean_fee_text(text)
    text = _normalize_for_parsing(text)
    exactness = _extract_exactness(text)
    operators = _extract_operators(text)
    operator = "+" if "+" in text else None
    tokens: list[FeeToken] = []
    amounts = _extract_currency_and_amount(text)
    for amount in amounts:
        token_id = hashlib.sha256(f"amount:{amount['amount_text']}:{amount['currency']}".encode()).hexdigest()[:16]
        tokens.append(
            FeeToken(
                raw=amount["amount_text"],
                kind="amount",
                amount=amount["amount"],
                currency=amount["currency"],
                operator=operator,
                exactness=exactness,
                token_id=token_id,
                is_minor_currency=amount.get("is_minor", False),
            )
        )
    for pct in _extract_percentage(text):
        token_id = hashlib.sha256(f"percentage:{pct['percentage']}".encode()).hexdigest()[:16]
        tokens.append(
            FeeToken(
                raw=pct["percentage_text"],
                kind="percentage",
                percentage=pct["percentage"],
                basis_points=pct["basis_points"],
                operator=operator,
                exactness=exactness,
                token_id=token_id,
            )
        )
    if not tokens and exactness:
        tokens.append(
            FeeToken(
                raw=text,
                kind="qualifier",
                exactness=exactness,
                operator=operators[0] if operators else None,
            )
        )
    return tokens


def parse_fee_value(text: str) -> dict[str, Any]:
    """Parse a single fee phrase into a normalized dictionary."""
    tokens = tokenize_fee_text(text)
    percentages = [t for t in tokens if t.kind == "percentage"]
    amounts = [t for t in tokens if t.kind == "amount"]
    # Derive exactness and operators from the tokenized result to avoid
    # re-parsing the same phrase.
    exactness = next((t.exactness for t in tokens if t.exactness), None) or _extract_exactness(text)
    operators = _extract_operators(text)
    return {
        "percentage": percentages[0].percentage if percentages else None,
        "basis_points": percentages[0].basis_points if percentages else None,
        "fixed_amount": amounts[0].amount if amounts else None,
        "fixed_currency": amounts[0].currency if amounts else None,
        "fixed_amounts": [{"amount": a.amount, "currency": a.currency} for a in amounts],
        "exactness": exactness or "exact",
        "operators": operators,
        "tokens": tokens,
        "raw": text,
    }
