"""Conservative classification of Stripe pricing entries into fee rules."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from .models import FeeCondition, FeeRule, PricingEntry
from .normalize import stable_id
from .pricing_tokens import parse_fee_value

logger = logging.getLogger(__name__)


def _has_fee_value(parsed: dict[str, Any]) -> bool:
    return bool(
        parsed.get("percentage") or parsed.get("fixed_amount") or parsed.get("exactness") in {"free", "included"}
    )


def _infer_card_region(phrase: str) -> str | None:
    lower = phrase.lower()
    if "eea" in lower or "european economic area" in lower:
        return "eea"
    if "uk" in lower or "united kingdom" in lower or "british" in lower:
        return "uk"
    if "international" in lower or "non-eea" in lower or "non eea" in lower:
        return "international"
    if "domestic" in lower:
        return "domestic"
    return None


def _infer_card_tier(phrase: str) -> str | None:
    lower = phrase.lower()
    if "premium" in lower:
        return "premium"
    if "standard" in lower:
        return "standard"
    return None


def _infer_channel_from_entry(entry: PricingEntry) -> str:
    if entry.channel:
        return entry.channel
    path = " ".join(entry.section_path).lower()
    if "terminal" in path or "in-person" in path or "tap to pay" in path:
        return "in_person"
    if "online" in path or "checkout" in path or "payment link" in path:
        return "online"
    return "online"


def _infer_fee_category(entry: PricingEntry) -> str | None:
    path = [p.lower() for p in entry.section_path]
    text = entry.source_text.lower()
    combined = " ".join(path) + " " + text

    if "custom pricing" in combined or "contact sales" in combined:
        return "custom_pricing"
    if "free" in text or "included" in text or "no additional fee" in text or "no fee" in text:
        if "payments" in combined or "card" in combined or entry.payment_method:
            return "payment_method"
        return "other"
    if "dispute" in combined or "chargeback" in combined:
        if "counter" in combined or "respond" in combined:
            return "dispute_counter"
        if "smart" in combined or "ai" in combined:
            return "smart_disputes"
        return "dispute"
    if "refund" in combined:
        return "refund"
    if "3d secure" in combined or "3-d secure" in combined or "customer authentication" in combined:
        return "three_d_secure"
    if "instant payout" in combined or "instant pay" in combined:
        return "instant_payout"
    if "currency conversion" in combined or "fx" in combined:
        return "currency_conversion"
    if "terminal" in combined or "in-person" in combined or "tap to pay" in combined:
        return "terminal"
    if "card" in combined:
        return "card_payment"
    if entry.payment_method:
        return "payment_method"
    if "managed payments" in combined:
        return "managed_payments"
    if "custom domain" in combined:
        return "custom_domain"
    if "post-payment invoice" in combined or "post payment invoice" in combined:
        return "post_payment_invoice"
    return None


def _infer_unit(fee_category: str | None, phrase: str) -> str:
    lower = phrase.lower()
    if fee_category in {"dispute", "dispute_counter", "smart_disputes"}:
        return "per_dispute"
    if "per month" in lower or "monthly" in lower:
        return "monthly"
    if "per year" in lower or "yearly" in lower:
        return "yearly"
    if "per invoice" in lower:
        return "per_invoice"
    if "per payout" in lower or "payout" in lower:
        return "per_payout"
    if "per transaction" in lower:
        return "per_transaction"
    if "per successful" in lower or "per charge" in lower:
        return "per_charge"
    return "per_transaction"


def _infer_exactness(parsed: dict[str, Any], phrase: str) -> str:
    exactness = parsed.get("exactness") or "exact"
    lower = phrase.lower()
    if "contact sales" in lower or "custom" in lower:
        return "custom"
    if "starting at" in lower or "from" in lower:
        return "from"
    if "up to" in lower or "capped" in lower or "cap" in lower:
        return "range"
    if "included" in lower or "no additional" in lower:
        return "included"
    if "free" in lower:
        return "free"
    return exactness


def _build_conditions(entry: PricingEntry, parsed: dict[str, Any]) -> list[FeeCondition]:
    conditions: list[FeeCondition] = []
    phrase = entry.source_text.lower()
    evidence = (entry.source_evidence or "").lower()
    path = " ".join(p.lower() for p in entry.section_path)

    if "currency conversion" in phrase or "if currency conversion is required" in phrase or "currency conversion" in evidence:
        conditions.append(FeeCondition(dimension="currency_conversion_required", value=True))
    if "standard settlement" in phrase:
        conditions.append(FeeCondition(dimension="settlement_timing", value="standard"))
    if "instant settlement" in phrase or "instant payout" in phrase:
        conditions.append(FeeCondition(dimension="settlement_timing", value="instant"))
    if "won disputes" in phrase:
        conditions.append(FeeCondition(dimension="dispute_state", value="won"))
    if "lost disputes" in phrase:
        conditions.append(FeeCondition(dimension="dispute_state", value="lost"))
    if "recurring" in path or "subscription" in path:
        conditions.append(FeeCondition(dimension="recurring", value=True))
    if "one-time" in path or "one time" in path:
        conditions.append(FeeCondition(dimension="recurring", value=False))
    if "managed payments" in path:
        conditions.append(FeeCondition(dimension="managed_payments", value=True))
    if "per successful transaction" in phrase:
        conditions.append(FeeCondition(dimension="success", value=True))
    return conditions


def _fixed_amount_minor(amount: str, currency: str) -> str | None:
    try:
        dec = Decimal(amount)
        # Use a simple 2-decimal default; caller should refine per currency.
        minor = int(dec * Decimal("100"))
        return str(minor)
    except Exception:
        return None


def _classify_entry(entry: PricingEntry) -> FeeRule | None:
    parsed = parse_fee_value(entry.source_text)
    if not _has_fee_value(parsed):
        return None

    fee_category = _infer_fee_category(entry)
    if not fee_category:
        return None

    channel = _infer_channel_from_entry(entry)
    unit = _infer_unit(fee_category, entry.source_text)
    exactness = _infer_exactness(parsed, entry.source_text)
    conditions = _build_conditions(entry, parsed)
    card_region = _infer_card_region(entry.source_text)
    card_tier = _infer_card_tier(entry.source_text)
    payment_method = entry.payment_method or ("card" if fee_category == "card_payment" else None)

    # Extract common condition flags from the conditions list for convenience.
    currency_conversion_required = any(
        c.dimension == "currency_conversion_required" and c.value for c in conditions
    ) or None
    recurring = next((c.value for c in conditions if c.dimension == "recurring" and isinstance(c.value, bool)), None)

    # Avoid classifying vague phrases without clear numeric values unless they are free/included.
    if exactness in {"custom", "from"} and not (parsed.get("percentage") or parsed.get("fixed_amount")):
        exactness = "non_calculable"

    if exactness == "non_calculable":
        return None

    rule_id = stable_id(
        entry.entry_id,
        fee_category,
        payment_method or "",
        channel,
        card_region or "",
        card_tier or "",
    )

    fixed_amount = parsed.get("fixed_amount")
    fixed_currency = parsed.get("fixed_currency")
    fixed_amount_minor = None
    if fixed_amount and fixed_currency:
        fixed_amount_minor = _fixed_amount_minor(fixed_amount, fixed_currency)

    return FeeRule(
        rule_id=rule_id,
        entry_id=entry.entry_id,
        name=fee_category,
        provider="stripe",
        channel=channel,
        payment_method=payment_method,
        card_origin="international"
        if card_region == "international"
        else ("domestic" if card_region == "domestic" else None),
        card_region=card_region,
        card_tier=card_tier,
        currency_conversion_required=currency_conversion_required,
        recurring=recurring,
        percentage=parsed.get("percentage"),
        basis_points=parsed.get("basis_points"),
        fixed_amount=fixed_amount,
        fixed_amount_minor=fixed_amount_minor,
        fixed_currency=fixed_currency,
        unit=unit,
        exactness=exactness,
        behavior="additive"
        if "+" in entry.source_text and "currency conversion" in entry.source_text.lower()
        else "additive",
        conditions=conditions,
        source_text=entry.source_text,
        source_url=entry.source_url,
        classification_status="classified",
        confidence=0.85,
        classification_evidence=[f"matched {fee_category} pattern", f"channel={channel}"],
    )


def classify_entries(entries: list[PricingEntry]) -> tuple[list[FeeRule], list[PricingEntry]]:
    """Classify pricing entries into derived rules and unclassified leftovers.

    Returns (rules, unclassified_entries).
    """
    rules: list[FeeRule] = []
    unclassified: list[PricingEntry] = []

    for entry in entries:
        rule = _classify_entry(entry)
        if rule is not None:
            rules.append(rule)
        else:
            entry = entry.model_copy(
                update={
                    "classification_status": "non_calculable",
                    "confidence": 0.0,
                    "classification_evidence": ["no clear numeric fee or unrecognized category"],
                }
            )
            unclassified.append(entry)

    return rules, unclassified


def derive_market_fees(entries: list[PricingEntry]) -> tuple[list[FeeRule], list[PricingEntry], str]:
    """Derive all rules for a market and compute the derivation status."""
    rules, unclassified = classify_entries(entries)
    status = "complete" if rules and not unclassified else ("partial" if rules else "unclassified")
    return rules, unclassified, status
