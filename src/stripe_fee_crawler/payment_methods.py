"""Centralized payment-method vocabulary shared across extraction, classification and output."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable

logger = logging.getLogger(__name__)


_PAYMENT_METHOD_TOKENS: tuple[str, ...] = (
    "sepa_direct_debit",
    "sepa_bank_transfer",
    "ach_direct_debit",
    "bacs_direct_debit",
    "pre_authorized_debit",
    "pad",
    "bancontact",
    "bizum",
    "blik",
    "eps",
    "ideal",
    "wero",
    "przelewy24",
    "swish",
    "twint",
    "pay_by_bank",
    "mb_way",
    "pix",
    "upi",
    "klarna",
    "billie",
    "scalapay",
    "multibanco",
    "alipay",
    "mobilepay",
    "paypal",
    "revolut_pay",
    "wechat_pay",
    "amazon_pay",
    "satispay",
    "konbini",
    "apple_pay",
    "google_pay",
    "click_to_pay",
    "cash_app_pay",
    "cash_app_afterpay",
    "afterpay",
    "clearpay",
    "affirm",
    "zip",
    "sunbit",
    "tap_to_pay",
    "link",
    "paypay",
    "card",
    "terminal",
    "bank_transfer",
)


_CARD_FAMILIES: frozenset[str] = frozenset(
    {
        "card",
        "domestic_card",
        "international_card",
        "premium_card",
        "standard_card",
    }
)

_BANK_DEBIT_FAMILIES: frozenset[str] = frozenset(
    {
        "sepa_direct_debit",
        "sepa_bank_transfer",
        "ach_direct_debit",
        "bacs_direct_debit",
    }
)

_BANK_REDIRECT_FAMILIES: frozenset[str] = frozenset(
    {
        "ideal",
        "wero",
        "bancontact",
        "eps",
        "blik",
        "przelewy24",
        "swish",
        "twint",
        "pay_by_bank",
        "mb_way",
        "pix",
        "upi",
        "bizum",
    }
)

_WALLET_FAMILIES: frozenset[str] = frozenset(
    {
        "alipay",
        "wechat_pay",
        "mobilepay",
        "paypal",
        "revolut_pay",
        "amazon_pay",
        "satispay",
    }
)

_BNPL_FAMILIES: frozenset[str] = frozenset({"klarna", "billie", "scalapay"})


def _family_for_method(method: str) -> str:
    """Map a normalized payment-method id to its family."""
    lower = method.lower()
    if lower in _CARD_FAMILIES:
        return "card"
    if lower in _BANK_DEBIT_FAMILIES:
        return "bank_debit"
    if lower in _BANK_REDIRECT_FAMILIES:
        return "bank_redirect"
    if lower in _WALLET_FAMILIES:
        return "wallet"
    if lower in _BNPL_FAMILIES:
        return "buy_now_pay_later"
    if lower == "multibanco":
        return "cash_voucher"
    return "other"


def _earliest_payment_method(text: str, max_word_index: int | None = None) -> str | None:
    """Return the earliest payment-method token in ``text``.

    When ``max_word_index`` is given, only consider matches whose first word
    starts at or before that word position.
    """
    text_lower = text.lower()
    words = text_lower.split()
    head = " ".join(words[:6])
    best: tuple[int, int, int, str] | None = None
    for method in _PAYMENT_METHOD_TOKENS:
        display = method.replace("_", " ")
        search_text = head if max_word_index is not None else text_lower
        for match in re.finditer(rf"\b{re.escape(display)}s?\b", search_text):
            word_index = len(head[: match.start()].split()) if max_word_index is not None else 0
            if max_word_index is not None and word_index > max_word_index:
                continue
            key = (word_index, match.start(), -len(display))
            if best is None or key < best[:3]:
                best = (*key, method)
    return best[3] if best else None


def _infer_payment_method_from_candidates(candidates: Iterable[str]) -> str | None:
    """Return the first payment method found across ``candidates`` in order.

    Matches are word-boundary aware and chosen by earliest word position, with
    candidates earlier in the iterable taking precedence.
    """
    best: tuple[int, int, int, int, str] | None = None
    for candidate_index, raw in enumerate(candidates):
        text_lower = raw.lower()
        for method in _PAYMENT_METHOD_TOKENS:
            display = method.replace("_", " ")
            for match in re.finditer(rf"\b{re.escape(display)}s?\b", text_lower):
                word_index = len(text_lower[: match.start()].split())
                # Earlier candidate wins, then earliest word position, then
                # earliest character start, then longest display name.
                key = (candidate_index, word_index, match.start(), -len(display))
                if best is None or key < best[:4]:
                    best = (*key, method)
    return best[4] if best else None


__all__ = [
    "_PAYMENT_METHOD_TOKENS",
    "_family_for_method",
    "_earliest_payment_method",
    "_infer_payment_method_from_candidates",
]
