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
    "AED": "AED",
    "DKK": "DKK",
    "NOK": "NOK",
}


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
    "no fee": "free",
}


OPERATOR_MARKERS: set[str] = {"+", "-", "&", "and", "or", "if", "for"}


def _normalize_for_parsing(text: str) -> str:
    """Replace common separators and normalize spaces."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"([0-9])\s*%", r"\1%", text)
    text = re.sub(r"([0-9])\s*([€£$¥])", r"\1\2", text)
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


def _extract_currency_and_amount(text: str) -> list[dict[str, Any]]:
    """Extract all currency-amount pairs from a fee phrase."""
    results: list[dict[str, Any]] = []
    # Look for symbols immediately followed by a number (prefix notation).
    symbol_pattern = "|".join(re.escape(s) for s in CURRENCY_SYMBOLS)
    pattern = rf"(?P<symbol>{symbol_pattern})(?P<amount>[0-9][0-9\s,.]*)"
    for match in re.finditer(pattern, text):
        symbol = match.group("symbol")
        amount_text = match.group("amount").replace(" ", "")
        amount = _parse_decimal(amount_text)
        if amount is not None:
            results.append(
                {
                    "symbol": symbol,
                    "currency": _currency_for_symbol(symbol),
                    "amount_text": amount_text,
                    "amount": str(amount),
                }
            )
    # Look for symbols immediately preceded by a number (suffix notation).
    pattern = rf"(?P<amount>[0-9][0-9\s,.]*)(?P<symbol>{symbol_pattern})"
    for match in re.finditer(pattern, text):
        symbol = match.group("symbol")
        amount_text = match.group("amount").replace(" ", "")
        amount = _parse_decimal(amount_text)
        if amount is not None:
            results.append(
                {
                    "symbol": symbol,
                    "currency": _currency_for_symbol(symbol),
                    "amount_text": amount_text,
                    "amount": str(amount),
                }
            )
    # Look for currency codes followed by a number.
    code_pattern = "|".join(re.escape(c) for c in CURRENCY_CODES)
    pattern = rf"(?P<code>{code_pattern})\s*(?P<amount>[0-9][0-9\s,.]*)"
    for match in re.finditer(pattern, text):
        code = match.group("code")
        amount_text = match.group("amount").replace(" ", "")
        amount = _parse_decimal(amount_text)
        if amount is not None:
            results.append(
                {
                    "symbol": code,
                    "currency": code,
                    "amount_text": amount_text,
                    "amount": str(amount),
                }
            )
    # Look for currency codes preceded by a number.
    pattern = rf"(?P<amount>[0-9][0-9\s,.]*)\s*(?P<code>{code_pattern})"
    for match in re.finditer(pattern, text):
        code = match.group("code")
        amount_text = match.group("amount").replace(" ", "")
        amount = _parse_decimal(amount_text)
        if amount is not None:
            results.append(
                {
                    "symbol": code,
                    "currency": code,
                    "amount_text": amount_text,
                    "amount": str(amount),
                }
            )
    return results


def _extract_percentage(text: str) -> dict[str, Any] | None:
    """Extract a percentage from a fee phrase."""
    match = re.search(r"(?P<value>[0-9]+(?:[.,][0-9]+)?)\s*%", text)
    if not match:
        return None
    value_text = match.group("value").replace(" ", "")
    value = _parse_decimal(value_text)
    if value is None:
        return None
    return {
        "percentage_text": value_text,
        "percentage": str(value),
        "basis_points": _to_basis_points(value),
    }


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
            )
        )
    pct = _extract_percentage(text)
    if pct:
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
