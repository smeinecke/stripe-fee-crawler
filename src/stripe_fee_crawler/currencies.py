"""Centralized currency tables and helpers.

This module owns all static currency knowledge used by tokenization,
market detection, classification, and validation so that the tables do not
drift between modules.
"""

from __future__ import annotations

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


def currency_exponent(currency: str) -> int:
    """Return the ISO 4217 exponent for a currency code."""
    return CURRENCY_EXPONENTS.get(currency.upper(), 2)


CURRENCY_BY_COUNTRY: dict[str, str] = {
    "AU": "AUD",
    "AT": "EUR",
    "BE": "EUR",
    "BR": "BRL",
    "BG": "EUR",
    "CA": "CAD",
    "HR": "EUR",
    "CY": "EUR",
    "CZ": "CZK",
    "DK": "DKK",
    "EE": "EUR",
    "FI": "EUR",
    "FR": "EUR",
    "DE": "EUR",
    "GI": "GBP",
    "GR": "EUR",
    "HK": "HKD",
    "HU": "HUF",
    "IN": "INR",
    "ID": "IDR",
    "IE": "EUR",
    "IT": "EUR",
    "JP": "JPY",
    "LV": "EUR",
    "LI": "CHF",
    "LT": "EUR",
    "LU": "EUR",
    "MY": "MYR",
    "MT": "EUR",
    "MX": "MXN",
    "NL": "EUR",
    "NZ": "NZD",
    "NO": "NOK",
    "PL": "PLN",
    "PT": "EUR",
    "RO": "RON",
    "SG": "SGD",
    "SK": "EUR",
    "SI": "EUR",
    "ES": "EUR",
    "SE": "SEK",
    "CH": "CHF",
    "TH": "THB",
    "AE": "AED",
    "GB": "GBP",
    "US": "USD",
}


# Currency symbols / prefixes that appear in Stripe pricing text.
_CURRENCY_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:US\$|\$)\s*\d", "USD"),
    (r"\b€\s*\d", "EUR"),
    (r"\b£\s*\d", "GBP"),
    (r"\bA\$\s*\d", "AUD"),
    (r"\bCA\$\s*\d", "CAD"),
    (r"\bJP¥\s*\d", "JPY"),
    (r"\b¥\s*\d", "JPY"),
    (r"\bAED\s*\d", "AED"),
    (r"\bINR\s*\d", "INR"),
    (r"\bBRL\s*\d", "BRL"),
    (r"\bIDR\s*[\d.]+", "IDR"),
]
