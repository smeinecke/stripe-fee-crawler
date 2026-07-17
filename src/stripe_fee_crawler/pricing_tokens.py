"""Pricing text tokenization and normalization."""

from __future__ import annotations

import hashlib
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import FeeToken
from .rich_text import clean_fee_text

CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "A$": "AUD",
    "C$": "CAD",
    "HK$": "HKD",
    "S$": "SGD",
    "NZ$": "NZD",
    "CHF": "CHF",
    "R$": "BRL",
    "MX$": "MXN",
    "kr": "SEK",
    "zł": "PLN",
    "lei": "RON",
    "฿": "THB",
    "¢": "USD",
    "p": "GBP",
    "AED": "AED",
    "DKK": "DKK",
    "NOK": "NOK",
}


MINOR_CURRENCY_SYMBOLS: set[str] = {"¢", "p"}


CURRENCY_CODES: set[str] = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "INR",
    "AUD",
    "CAD",
    "HKD",
    "SGD",
    "NZD",
    "CHF",
    "BRL",
    "MXN",
    "SEK",
    "PLN",
    "RON",
    "THB",
    "AED",
    "DKK",
    "NOK",
    "IDR",
    "MYR",
    "PHP",
    "VND",
    "KRW",
    "CNY",
    "ZAR",
    "ILS",
    "CZK",
    "HUF",
    "BGN",
    "HRK",
}

# ISO 4217 currency exponents. Defaults to 2 when absent.
CURRENCY_EXPONENTS: dict[str, int] = {
    "JPY": 0,
    "KRW": 0,
    "VND": 0,
    "BHD": 3,
    "IQD": 3,
    "JOD": 3,
    "KWD": 3,
    "OMR": 3,
    "TND": 3,
}


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


OPERATOR_MARKERS: set[str] = {"+", "-", "&", "and", "or", "if", "for"}


def _normalize_for_parsing(text: str) -> str:
    """Replace common separators and normalize spaces."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"([0-9])\s*%", r"\1%", text)
    # Currency-symbol spacing is handled by regex; do not collapse here because
    # multi-character symbols such as "A$" are parsed correctly with \s*.
    return text


def _detect_decimal_separator(text: str) -> str:
    """Detect whether dot or comma is the decimal separator."""
    # Count comma/dot usage in numeric contexts with one or more decimal digits.
    comma_count = len(re.findall(r"[0-9],[0-9]+(?![0-9])", text))
    dot_count = len(re.findall(r"[0-9]\.[0-9]+(?![0-9])", text))
    if comma_count > dot_count:
        return ","
    if dot_count > comma_count:
        return "."
    # When both appear, prefer the separator followed by 1-2 digits as the decimal.
    comma_decimal = len(re.findall(r"[0-9],[0-9]{1,2}(?![0-9])", text))
    dot_decimal = len(re.findall(r"[0-9]\.[0-9]{1,2}(?![0-9])", text))
    if comma_decimal and not dot_decimal:
        return ","
    if dot_decimal and not comma_decimal:
        return "."
    return "."


def _parse_decimal(text: str) -> Decimal | None:
    """Parse a decimal string respecting localized separators."""
    text = text.strip()
    if not text:
        return None
    sep = _detect_decimal_separator(text)
    # Remove thousands separators (whichever is not the decimal separator).
    text = text.replace(".", "").replace(",", ".") if sep == "," else text.replace(",", "")
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _to_basis_points(percentage: Decimal) -> str:
    """Convert a percentage decimal to basis points string."""
    return str(percentage * Decimal("100"))


def _currency_for_symbol(symbol: str) -> str | None:
    return CURRENCY_SYMBOLS.get(symbol)


def currency_exponent(currency: str) -> int:
    """Return the ISO 4217 exponent for a currency code."""
    return CURRENCY_EXPONENTS.get(currency.upper(), 2)


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
    # Minor symbols are only valid as suffixes.
    major_symbols = {s for s in CURRENCY_SYMBOLS if s not in MINOR_CURRENCY_SYMBOLS}
    minor_symbols = MINOR_CURRENCY_SYMBOLS
    symbol_pattern_major = "|".join(re.escape(s) for s in sorted(major_symbols, key=len, reverse=True))
    symbol_pattern_minor = "|".join(re.escape(s) for s in sorted(minor_symbols, key=len, reverse=True))
    symbol_pattern_all = "|".join(re.escape(s) for s in sorted(CURRENCY_SYMBOLS, key=len, reverse=True))
    code_pattern = "|".join(re.escape(c) for c in CURRENCY_CODES)

    amount_group = r"(?P<amount>[0-9][0-9\s,.]*)"
    amount_group_nb = r"(?P<amount>[0-9][0-9,.]*)"

    patterns: list[tuple[str, str, str]] = [
        # Major-symbol prefix: A$89, $100, € 1.50
        ("symbol_prefix", rf"(?P<symbol>{symbol_pattern_major})\s*{amount_group_nb}", "symbol"),
        # Symbol suffix: 100 USD, 20p, 5¢, 100 kr
        ("symbol_suffix", rf"{amount_group}\s*(?P<symbol>{symbol_pattern_all})(?!\w)", "symbol"),
        # ISO-code prefix: USD 100
        ("code_prefix", rf"(?P<code>{code_pattern})\s*{amount_group_nb}", "code"),
        # ISO-code suffix: 100 USD
        ("code_suffix", rf"{amount_group}\s*(?P<code>{code_pattern})(?!\w)", "code"),
        # Minor-symbol suffix without whitespace: 20p, 5¢
        ("symbol_minor_tight", rf"(?P<amount>[0-9][0-9,.]*)(?P<symbol>{symbol_pattern_minor})(?!\w)", "symbol"),
    ]

    seen_spans: list[tuple[int, int]] = []
    results: list[tuple[int, int, dict[str, Any]]] = []
    marketing_magnitudes = ("billion", "million", "trillion")

    def _char_at(idx: int) -> str | None:
        if 0 <= idx < len(text):
            return text[idx]
        return None

    for _match_type, pattern, group in patterns:
        for match in re.finditer(pattern, text):
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
    for marker, exactness in EXACTNESS_MARKERS.items():
        if marker in lower:
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
                operator="+" if "+" in text else None,
                exactness=_extract_exactness(text),
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
                operator="+" if "+" in text else None,
                exactness=_extract_exactness(text),
                token_id=token_id,
            )
        )
    exactness = _extract_exactness(text)
    operators = _extract_operators(text)
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
    exactness = _extract_exactness(text)
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
