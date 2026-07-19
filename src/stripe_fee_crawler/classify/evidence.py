"""Evidence and false-positive detection predicates."""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

from ..models import FeeComponent, FeeEvidence, PricingEntry
from ..pricing_tokens import CURRENCY_CODES, CURRENCY_SYMBOLS, parse_fee_value
from ._util import (
    _dedup_repeated_phrases,
    _is_explicit_fee_phrase,
    _ordered_unique,
    _per_completion_noun,
    _source_text_for_group,
    _text_has,
)
from .tables import (
    _CARD_NETWORK_TOKENS,
    _HARDWARE_PRICE_TERMS,
    _MARKET_SHARE_STATISTICS,
    _MARKETING_TERMS,
    _MIN_HARDWARE_MAJOR_AMOUNT,
    _PAYMENT_METHOD_TOKENS,
    _PROMOTIONAL_TERMS,
)

logger = logging.getLogger(__name__)


def _is_marketing_prose(text: str) -> bool:
    """Return True for statistics, volume claims, and other marketing copy."""
    lower = text.lower()
    # Volume/audience claims are marketing even if they also contain fee-adjacent
    # words such as "monthly" or "payments".
    if ("million" in lower or "billion" in lower or "trillion" in lower) and (
        "customers" in lower or "users" in lower or "addressable" in lower or "process" in lower
    ):
        return True
    # A phrase that already contains explicit fee language is a fee description,
    # not marketing prose (e.g. "Customers will be presented a conversion fee").
    if _is_explicit_fee_phrase(text):
        return False
    return _text_has(text, *_MARKETING_TERMS)


def _is_market_share_text(text: str) -> bool:
    """Return True when the text is a market-share/adoption statistic."""
    lower = text.lower()
    if not _text_has(lower, *_MARKET_SHARE_STATISTICS):
        return False
    # A statistic is only a false-positive fee if it actually contains a number.
    parsed = parse_fee_value(text)
    return bool(parsed["percentage"] or parsed["fixed_amount"])


def _is_promotional_language(text: str) -> bool:
    """Return True for conditional or sales-led pricing language."""
    return _text_has(text, *_PROMOTIONAL_TERMS)


def _has_hardware_context(text: str) -> bool:
    """Return True when the text describes a terminal/device purchase price."""
    return _text_has(text, *_HARDWARE_PRICE_TERMS)


def _is_terminal_hardware_price(
    entry: PricingEntry,
    product_id: str,
    fee_components: list[FeeComponent],
    unit: str | None,
) -> bool:
    """Detect terminal reader/device prices that should not be per-transaction fees."""
    if product_id != "terminal" and entry.payment_method != "terminal":
        return False
    text = (" ".join(entry.section_path + [entry.source_text])).lower()
    # A true terminal processing fee normally mentions a per-event unit.
    explicit_processing_context = _text_has(
        text,
        "per successful",
        "per transaction",
        "per charge",
        "per in-person",
        "authorization fee",
        "processing fee",
        "transaction fee",
        "fee",
        "fees",
    )
    fixed_components = [c for c in fee_components if c.type in {"fixed_amount", "fixed_surcharge"}]
    if not fixed_components:
        return False
    largest_fixed = max(
        (Decimal(c.amount) for c in fixed_components if c.amount and c.currency),
        default=Decimal("0"),
    )
    if largest_fixed < _MIN_HARDWARE_MAJOR_AMOUNT:
        return False
    if explicit_processing_context:
        return False
    return _has_hardware_context(text)


def _is_amount_from_method_name(
    entry: PricingEntry,
    fee_components: list[FeeComponent],
) -> bool:
    """Detect amounts that were parsed out of product/payment-method names."""
    text = entry.source_text
    for comp in fee_components:
        if comp.type in {"fixed_amount", "fixed_surcharge", "maximum_fee", "minimum_fee"} and comp.amount:
            # If the amount text appears right after an alphabetic payment-method
            # token (e.g. "P24" -> "24"), it is not a real fee amount.
            # Ignore amounts preceded by a currency symbol or ISO code.
            raw = comp.source_text or text or ""
            if not raw:
                continue
            for match in re.finditer(re.escape(comp.amount) + r"\b", raw):
                prefix = raw[: match.start()].rstrip()
                if not prefix:
                    continue
                token = prefix.split()[-1]
                if token.upper() in CURRENCY_CODES or token in CURRENCY_SYMBOLS:
                    continue
                if re.fullmatch(r"[A-Za-z]+", token):
                    return True
    return False


def _has_contradictory_fee_evidence(fee_components: list[FeeComponent]) -> bool:
    """Detect positive-fee evidence paired with included/free evidence."""
    component_types = {c.type for c in fee_components}
    positive_types = {
        "percentage",
        "fixed_amount",
        "percentage_surcharge",
        "fixed_surcharge",
        "maximum_fee",
        "minimum_fee",
    }
    has_positive = bool(component_types & positive_types)
    has_included_free = bool(component_types & {"included", "free"})
    return has_positive and has_included_free


def _fee_evidence_for_group(
    group: list[dict[str, Any]],
    product_id: str,
    fee_components: list[FeeComponent],
    unit: str | None,
) -> FeeEvidence:
    """Evaluate the evidence behind a would-be calculable rule."""
    base_entry = group[0]["entry"]
    combined = _source_text_for_group(group)
    # For price rows split from a surrounding cell, the noun after "per" or
    # the fee phrase may live in the surrounding evidence. Include it when the
    # source text itself is only an amount or ends with a dangling "per".
    text_for_evidence = combined
    if base_entry.source_text.rstrip().lower().endswith(" per"):
        per_noun = _per_completion_noun(base_entry)
        if per_noun:
            text_for_evidence = f"{combined} per {per_noun}"
        else:
            text_for_evidence = f"{combined} {base_entry.source_evidence or ''}".strip()
    elif not re.search(r"[a-z]{3,}", base_entry.source_text.lower()):
        text_for_evidence = f"{combined} {base_entry.source_evidence or ''}".strip()
    entry_ids = [item["entry"].entry_id for item in group]
    raw_phrases = [
        item["entry"].source_text
        for item in group
        if not _is_marketing_prose(_dedup_repeated_phrases(item["entry"].source_text))
    ]
    phrases = _ordered_unique(_dedup_repeated_phrases(p) for p in raw_phrases)

    # 1. Promotional / conditional pricing is never a concrete fee.
    if _is_promotional_language(combined):
        return FeeEvidence(
            type="promotional_language",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.1,
        )

    # 2. Marketing prose with numbers is not a fee. Use the base entry text
    # (not the combined group text) so a section heading with fee language does
    # not mask marketing copy in the body.
    if _is_marketing_prose(_dedup_repeated_phrases(base_entry.source_text)):
        return FeeEvidence(
            type="marketing_prose",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 2b. Cross-fragment alignment: the numeric fee value and the fee wording
    # must come from the same logical pricing record. If a market-share/statistic
    # fragment supplies the number while a different fragment supplies the fee
    # phrase, the rule is a false positive.
    positive_component_sources = {
        _dedup_repeated_phrases(c.source_text) if c.source_text else None
        for c in fee_components
        if c.type in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge"}
    }
    if positive_component_sources:
        value_from_marketing = all(_is_market_share_text(src) for src in positive_component_sources if src)
        fee_phrase_sources = [
            _dedup_repeated_phrases(e.source_text)
            for e in [item["entry"] for item in group]
            if _is_explicit_fee_phrase(_dedup_repeated_phrases(e.source_text))
            and not _is_market_share_text(_dedup_repeated_phrases(e.source_text))
        ]
        if value_from_marketing and fee_phrase_sources:
            return FeeEvidence(
                type="cross_fragment_fee_evidence",
                source_entry_ids=entry_ids,
                phrases=phrases,
                confidence=0.0,
            )

    # 3. Terminal hardware purchase prices.
    if _is_terminal_hardware_price(base_entry, product_id, fee_components, unit):
        return FeeEvidence(
            type="hardware_price",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 4. Amounts extracted from alphanumeric method/product names.
    if _is_amount_from_method_name(base_entry, fee_components):
        return FeeEvidence(
            type="alphanumeric_method_name",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 5. Positive fee evidence paired with included/free evidence in the same
    #    logical row is contradictory; the included/free statement wins.
    if _has_contradictory_fee_evidence(fee_components):
        return FeeEvidence(
            type="contradictory_fee_evidence",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 6. A real fee needs explicit fee language, a trusted table heading with a
    #    fee formula, or an unconditional per-event/per-period phrase.
    if _is_explicit_fee_phrase(text_for_evidence):
        return FeeEvidence(
            type="explicit_fee_phrase",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.85,
        )

    # Trust a heading/section_path or the source text itself when it already
    # contains a percentage or amount next to a known payment method token.
    amount_patterns = r"(?:\d\s*%|[A-Za-z]?[€£$¥₹₩]\s*\d|\d\s*[€£$¥₹₩])"
    method_pattern_tokens = sorted(set(_PAYMENT_METHOD_TOKENS) | set(_CARD_NETWORK_TOKENS), key=len, reverse=True)
    method_pattern = r"\b(" + "|".join(re.escape(m.replace("_", " ")) for m in method_pattern_tokens) + r")\b"
    text_or_path = (
        " ".join(p.lower() for p in base_entry.section_path if p)
        + " "
        + base_entry.source_text.lower()
        + " "
        + (base_entry.source_evidence or "").lower()
    ).strip()
    if re.search(amount_patterns, text_or_path) and re.search(method_pattern, text_or_path):
        return FeeEvidence(
            type="pricing_table_value",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.75,
        )

    return FeeEvidence(
        type="insufficient",
        source_entry_ids=entry_ids,
        phrases=phrases,
        confidence=0.0,
    )


__all__ = [
    "_is_marketing_prose",
    "_is_market_share_text",
    "_is_promotional_language",
    "_has_hardware_context",
    "_is_terminal_hardware_price",
    "_is_amount_from_method_name",
    "_has_contradictory_fee_evidence",
    "_fee_evidence_for_group",
]
