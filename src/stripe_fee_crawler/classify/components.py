"""Fee component and pricing-fragment predicates."""

from __future__ import annotations

import logging

from ..models import FeeComponent, PricingEntry
from ..pricing_tokens import parse_fee_value, tokenize_fee_text
from ._util import _first_match, _fixed_amount_minor, _has_base_fee, _is_tap_to_pay
from .dimensions import _entry_component_hint
from .tables import _AMOUNT_COMPONENT_TABLE

logger = logging.getLogger(__name__)


def _is_modifier_entry(entry: PricingEntry) -> bool:
    """Return True when an entry is a continuation of the previous base fee."""
    parsed = parse_fee_value(entry.source_text)
    hint = _entry_component_hint(entry)
    has_base = _has_base_fee(entry, parsed)

    if has_base:
        # A standalone base row can still contain a cap/max (e.g. "0.8% ... $5 cap").
        # It is not a modifier unless it only describes the modifier.
        return hint in {"maximum", "minimum"} and not any(t.kind == "percentage" for t in entry.tokens)
    # Modifiers are non-base lines that qualify an earlier fee: caps, minimums,
    # or included/free statements.  Surcharge fragments beginning with '+' and
    # standalone method variants (e.g. Tap to Pay) are base rules in their own
    # right.
    return hint in {"maximum", "minimum", "included", "free"} and not _is_tap_to_pay(entry)


def _amount_component_type(hint: str, text: str) -> str:
    if hint == "surcharge":
        return "fixed_surcharge"
    component = _first_match((hint or "") + " " + text.lower(), _AMOUNT_COMPONENT_TABLE)
    if component:
        return component
    if hint == "included":
        return "included"
    if hint == "free":
        return "free"
    if hint == "custom":
        return "custom_pricing"
    return "fixed_amount"


def _build_components_for_entry(entry: PricingEntry, hint: str) -> list[FeeComponent]:
    """Convert an entry's tokens into typed fee components."""

    components: list[FeeComponent] = []
    tokens = entry.tokens or tokenize_fee_text(entry.source_text)
    if not tokens:
        if hint == "included":
            components.append(
                FeeComponent(type="included", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        elif hint == "free":
            components.append(FeeComponent(type="free", source_entry_id=entry.entry_id, source_text=entry.source_text))
        elif hint == "custom":
            components.append(
                FeeComponent(type="custom_pricing", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        else:
            components.append(
                FeeComponent(type="non_calculable", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        return components

    for token in tokens:
        if token.kind == "percentage":
            if hint == "surcharge":
                comp_type = "percentage_surcharge"
            elif hint == "from":
                comp_type = "percentage"
            else:
                comp_type = "percentage"
            components.append(
                FeeComponent(
                    type=comp_type,
                    value=token.percentage,
                    basis_points=token.basis_points,
                    operator=token.operator,
                    source_entry_id=entry.entry_id,
                    source_text=entry.source_text,
                )
            )
        elif token.kind == "amount":
            comp_type = _amount_component_type(hint, entry.source_text)
            minor = _fixed_amount_minor(token.amount, token.currency) if token.amount and token.currency else None
            components.append(
                FeeComponent(
                    type=comp_type,
                    amount=token.amount,
                    currency=token.currency,
                    minor_amount=minor,
                    operator=token.operator,
                    source_entry_id=entry.entry_id,
                    source_text=entry.source_text,
                )
            )
    return components


__all__ = ["_is_modifier_entry", "_amount_component_type", "_build_components_for_entry"]
