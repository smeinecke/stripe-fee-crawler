"""Normalization helpers for market data and identifiers."""

from __future__ import annotations

import hashlib
import re


def stable_id(*parts: str) -> str:
    """Return a deterministic stable identifier from ordered parts."""
    normalized = "|".join(p.strip().lower() for p in parts if p and p.strip())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def normalize_country_code(code: str) -> str:
    """Normalize an ISO 3166-1 alpha-2 country code."""
    code = code.strip().upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError(f"Invalid ISO country code: {code!r}")
    return code


def normalize_locale(locale: str) -> str:
    """Normalize a locale string to lowercase with hyphens."""
    return locale.strip().lower().replace("_", "-")


def normalize_method_name(name: str) -> str:
    """Normalize a payment method name into a stable identifier."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def normalize_currency(currency: str) -> str:
    """Normalize an ISO 4217 currency code."""
    currency = currency.strip().upper()
    if len(currency) != 3 or not currency.isalpha():
        raise ValueError(f"Invalid ISO 4217 currency code: {currency!r}")
    return currency
