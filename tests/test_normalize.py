"""Tests for normalization helpers."""

from __future__ import annotations

import pytest

from stripe_fee_crawler.normalize import (
    normalize_country_code,
    normalize_currency,
    normalize_locale,
    normalize_method_name,
    stable_id,
)


def test_stable_id_deterministic() -> None:
    a = stable_id("Payments", "Domestic cards", "1.5%")
    b = stable_id("Payments", "Domestic cards", "1.5%")
    assert a == b
    assert len(a) == 16


def test_normalize_country_code() -> None:
    assert normalize_country_code("de") == "DE"


def test_normalize_country_code_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_country_code("DEU")


def test_normalize_locale() -> None:
    assert normalize_locale("en_DE") == "en-de"


def test_normalize_currency() -> None:
    assert normalize_currency("eur") == "EUR"


def test_normalize_method_name() -> None:
    assert normalize_method_name("iDEAL | Wero") == "ideal_wero"
