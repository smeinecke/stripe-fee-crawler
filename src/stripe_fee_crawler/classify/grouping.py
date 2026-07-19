"""Entry enrichment, grouping and rule assembly."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..models import FeeComponent, FeeCondition, FeeEvidence, FeeRule, PricingEntry
from ..normalize import stable_id
from ..pricing_tokens import parse_fee_value
from ._util import (
    _dedup_repeated_phrases,
    _has_base_fee,
    _is_explicit_fee_phrase,
    _is_unsupported_multi_per_shape,
    _ordered_unique,
    _text_has_lower,
)
from .components import _build_components_for_entry, _is_modifier_entry
from .dimensions import (
    _card_origin_for_region,
    _entry_component_hint,
    _infer_card_region,
    _infer_card_tier,
    _infer_channel,
    _infer_conditions,
    _infer_exactness,
    _infer_payment_method,
    _infer_product_id,
    _infer_unit,
    _infer_variant_id,
    _is_card_product,
    _is_payment_method_product,
)
from .evidence import _fee_evidence_for_group
from .tables import (
    _PUBLICATION_CONFIDENCE_THRESHOLD,
    CALCULABLE_RULE,
    CUSTOM_PRICING,
    FREE,
    INCLUDED,
    INFORMATIONAL,
    NON_CALCULABLE,
    REFERENCE_ONLY,
    UNSUPPORTED_SHAPE,
)

logger = logging.getLogger(__name__)


def _empty_group_rule(
    group: list[dict[str, Any]], account_country: str | None
) -> tuple[FeeRule | None, list[PricingEntry]]:
    """Attempt to derive a rule even from non-fee or informational entries."""
    base = group[0]["entry"]
    parsed = parse_fee_value(base.source_text)
    if not _has_base_fee(base, parsed):
        hint = _entry_component_hint(base)
        if hint == "custom":
            return None, [base.model_copy(update={"classification_status": CUSTOM_PRICING})]
        if hint == "included":
            status = INCLUDED
        elif hint == "free":
            status = FREE
        else:
            return None, [base]
        product_id = _infer_product_id(base)
        variant_id = _infer_variant_id(base, product_id, account_country)
        conditions = _infer_conditions(base, product_id, variant_id, account_country)
        components = _build_components_for_entry(base, hint)
        evidence = FeeEvidence(
            type=status,
            source_entry_ids=[base.entry_id],
            phrases=[base.source_text],
            confidence=0.7,
        )
        rule = FeeRule(
            rule_id=stable_id(product_id, variant_id, *[f"{c.dimension}={c.value}" for c in conditions], base.entry_id),
            entry_id=base.entry_id,
            contributing_entry_ids=[base.entry_id],
            product_id=product_id,
            variant_id=variant_id,
            label=base.source_text,
            provider="stripe",
            account_country=account_country,
            payment_method=_infer_payment_method(base),
            conditions=conditions,
            fee_components=components,
            unit="informational",
            exactness=components[0].type if components else "included",
            behavior="informational",
            source_text=base.source_text,
            source_texts=[base.source_text],
            source_url=base.source_url,
            classification_status=status,
            confidence=0.7,
            fee_evidence=evidence,
        )
        return rule, []
    return None, [base]


def _base_conditions(item: dict[str, Any], account_country: str | None) -> list[FeeCondition]:
    """Build FeeCondition objects from enriched dimensions."""
    conditions: list[FeeCondition] = []
    product_id = item.get("product_id")
    if item.get("payment_method"):
        conditions.append(FeeCondition(dimension="payment_method", value=item["payment_method"]))
    if item.get("channel"):
        conditions.append(FeeCondition(dimension="channel", value=item["channel"]))
    # Card-region dimensions only apply to card-based products.
    if _is_card_product(product_id):
        if item.get("card_region"):
            conditions.append(FeeCondition(dimension="card_region", value=item["card_region"]))
            if item.get("card_origin"):
                conditions.append(FeeCondition(dimension="card_origin", value=item["card_origin"]))
        if item.get("card_tier"):
            conditions.append(FeeCondition(dimension="card_tier", value=item["card_tier"]))
    elif item.get("card_region"):
        region = item["card_region"]
        if region == "domestic":
            conditions.append(FeeCondition(dimension="cross_border", value=False))
            conditions.append(FeeCondition(dimension="transaction_region", value="domestic"))
        else:
            conditions.append(FeeCondition(dimension="cross_border", value=True))
            conditions.append(FeeCondition(dimension="transaction_region", value=region))
    if account_country:
        conditions.append(FeeCondition(dimension="account_country", value=account_country))
    return conditions


def _compute_behavior(fee_components: list[FeeComponent], label: str, exactness: str | None) -> str:
    """Determine rule behavior from components and wording."""
    has_surcharge = any(c.type in {"percentage_surcharge", "fixed_surcharge"} for c in fee_components)
    has_alternative = bool(re.search(r"\b(or|alternative|instead of)\b", label.lower()))
    if has_surcharge:
        return "additive"
    if has_alternative:
        return "alternative"
    if any(c.type in {"included", "free"} for c in fee_components) or exactness == "custom":
        return "informational"
    return "conditional"


_ONLINE_PRODUCT_IDS = {
    "disputes",
    "instant_payouts",
    "payouts",
    "three_d_secure",
    "payments",
    "adaptive_pricing",
    "post_payment_invoice",
    "invoicing",
    "tax",
    "managed_payments",
    "radar",
    "smart_disputes",
    "ach_direct_debit",
    "subscriptions",
    "billing",
    "identity",
    "custom_domain",
    "platform",
    "connect",
    "stablecoin_payments",
    "refunds",
    "sigma",
}


def _resolve_channel_and_unit(
    base_entry: PricingEntry,
    base_item: dict[str, Any],
    product_id: str,
    payment_method: str | None,
    unit: str | None,
) -> tuple[str | None, str | None]:
    """Fill in missing channel and unit for well-understood products."""
    channel = base_item["channel"] or _infer_channel(base_entry)
    if not channel:
        source_lower = base_entry.source_text.lower()
        if (
            product_id == "terminal"
            or payment_method == "tap_to_pay"
            or _text_has_lower(source_lower, "in-person", "in person", "tap to pay")
        ):
            channel = "in_person"
        elif payment_method or product_id in _ONLINE_PRODUCT_IDS or _text_has_lower(source_lower, "card", "payment"):
            channel = "online"
    if not unit:
        if product_id == "disputes":
            unit = "per_dispute"
        elif payment_method:
            unit = "per_transaction"
    return channel, unit


def _determine_status(
    exactness: str | None,
    calculable: bool,
    fee_evidence: FeeEvidence,
    channel: str | None,
    unit: str | None,
    behavior: str | None,
) -> tuple[str, float]:
    """Resolve the classification status and confidence for a group."""
    if exactness == "custom" or (not calculable and exactness == "from"):
        classification_status = CUSTOM_PRICING
    elif exactness in {"included", "free"}:
        classification_status = INCLUDED if exactness == "included" else FREE
    elif calculable and fee_evidence.type in {
        "explicit_fee_phrase",
        "pricing_table_value",
        "structured_fee_field",
    }:
        classification_status = CALCULABLE_RULE
    else:
        classification_status = NON_CALCULABLE

    if fee_evidence.type == "promotional_language":
        classification_status = CUSTOM_PRICING
    elif fee_evidence.type in {"marketing_prose", "hardware_price", "alphanumeric_method_name"}:
        classification_status = INFORMATIONAL
    elif fee_evidence.type in {"contradictory_fee_evidence", "cross_fragment_fee_evidence"} or (
        fee_evidence.type == "insufficient" and classification_status == CALCULABLE_RULE
    ):
        classification_status = NON_CALCULABLE

    if classification_status == CALCULABLE_RULE and (not channel or not unit or not behavior):
        classification_status = NON_CALCULABLE

    confidence = fee_evidence.confidence
    if classification_status != CALCULABLE_RULE:
        confidence = 0.0
    if classification_status == CALCULABLE_RULE and confidence < _PUBLICATION_CONFIDENCE_THRESHOLD:
        classification_status = NON_CALCULABLE

    return classification_status, confidence


def _classify_group(
    group: list[dict[str, Any]], account_country: str | None
) -> tuple[FeeRule | None, list[PricingEntry]]:
    """Classify a group of related entries into one FeeRule."""
    base_item = group[0]
    base_entry = base_item["entry"]
    product_id = base_item["product_id"]
    variant_id = base_item["variant_id"]

    entry_ids: list[str] = []
    raw_source_texts: list[str] = []
    fee_components: list[FeeComponent] = []
    exactness: str | None = None
    all_conditions: list[FeeCondition] = []

    # Seed conditions with the enriched dimensions from the base entry (e.g.
    # inherited payment_method/card_region for surcharge fragments).
    all_conditions.extend(_base_conditions(base_item, account_country))

    for item in group:
        entry = item["entry"]
        entry_ids.append(entry.entry_id)
        raw_source_texts.append(entry.source_text)
        hint = _entry_component_hint(entry)
        fee_components.extend(_build_components_for_entry(entry, hint))
        parsed = parse_fee_value(entry.source_text)
        # Concrete fee rows should not be re-classified by surrounding marketing text.
        if _has_base_fee(entry, parsed) and _is_explicit_fee_phrase(entry.source_text):
            exact_text = entry.source_text
        else:
            exact_text = entry.source_text + " " + (entry.source_evidence or "")
        entry_exactness = _infer_exactness(parsed, exact_text)
        if exactness is None or entry_exactness in {"custom", "from", "range"}:
            exactness = entry_exactness
        all_conditions.extend(_infer_conditions(entry, item["product_id"], item["variant_id"], account_country))

    if not fee_components:
        return _empty_group_rule(group, account_country)

    # Merge duplicate conditions; base/enriched dimensions win over inferred.
    seen_conditions: dict[str, FeeCondition] = {}
    for cond in all_conditions:
        key = cond.dimension
        if key not in seen_conditions:
            seen_conditions[key] = cond
    conditions = list(seen_conditions.values())

    # Deduplicate repeated qualifiers in the presentation text while keeping
    # every contributing entry id for provenance.
    source_texts = _ordered_unique(_dedup_repeated_phrases(t) for t in raw_source_texts)
    label = _dedup_repeated_phrases(base_entry.source_text)

    channel = base_item["channel"] or _infer_channel(base_entry)
    unit = _infer_unit(base_entry, product_id)
    payment_method = base_item["payment_method"]
    if not payment_method and _is_payment_method_product(product_id):
        payment_method = _infer_payment_method(base_entry)

    behavior = _compute_behavior(fee_components, label, exactness)
    channel, unit = _resolve_channel_and_unit(base_entry, base_item, product_id, payment_method, unit)

    # Determine calculability and final status using positive fee evidence.
    calculable = _has_base_fee(base_entry) or any(
        c.type in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge"} for c in fee_components
    )

    fee_evidence = _fee_evidence_for_group(group, product_id, fee_components, unit)

    classification_status, confidence = _determine_status(exactness, calculable, fee_evidence, channel, unit, behavior)

    # Stacked per-unit dimensions on an otherwise unrecognised product (e.g.
    # "per institution per account holder per month") cannot be modelled
    # deterministically.
    if product_id == "unspecified" and _is_unsupported_multi_per_shape(base_entry, unit):
        classification_status = UNSUPPORTED_SHAPE

    rule_id = stable_id(
        product_id,
        variant_id,
        *[f"{c.dimension}={c.value}" for c in sorted(conditions, key=lambda x: x.dimension)],
        base_entry.entry_id,
    )

    # Build legacy flat fields from components for consumers that still read them.
    percentage_component = next((c for c in fee_components if c.type in {"percentage", "percentage_surcharge"}), None)
    percentage = percentage_component.value if percentage_component else None
    basis_points = percentage_component.basis_points if percentage_component else None
    fixed = next((c for c in fee_components if c.type == "fixed_amount"), None)
    max_fee = next((c for c in fee_components if c.type == "maximum_fee"), None)
    min_fee = next((c for c in fee_components if c.type == "minimum_fee"), None)

    if _is_card_product(product_id):
        card_region = _infer_card_region(base_entry, account_country)
        card_tier = _infer_card_tier(base_entry.source_text)
        card_origin = _card_origin_for_region(card_region, account_country)
    else:
        card_region = None
        card_tier = None
        card_origin = None

    # Deduplicate source fragments by text while preserving all entry ids.
    fragment_text_to_id: dict[str, str] = {}
    for item in group:
        e = item["entry"]
        deduped = _dedup_repeated_phrases(e.source_text)
        if deduped not in fragment_text_to_id:
            fragment_text_to_id[deduped] = e.entry_id

    rule = FeeRule(
        rule_id=rule_id,
        entry_id=base_entry.entry_id,
        contributing_entry_ids=entry_ids,
        product_id=product_id,
        variant_id=variant_id,
        label=label,
        name=product_id,
        provider="stripe",
        account_country=account_country,
        channel=channel,
        payment_method=payment_method,
        card_origin=card_origin,
        card_region=card_region,
        card_tier=card_tier,
        currency_conversion_required=any(c.dimension == "currency_conversion_required" and c.value for c in conditions)
        or None,
        percentage=percentage,
        basis_points=basis_points,
        fixed_amount=fixed.amount if fixed else None,
        fixed_amount_minor=fixed.minor_amount if fixed else None,
        fixed_currency=fixed.currency if fixed else None,
        minimum_amount=min_fee.amount if min_fee else None,
        maximum_amount=max_fee.amount if max_fee else None,
        unit=unit or "informational",
        exactness=exactness or "exact",
        behavior=behavior or "informational",
        conditions=conditions,
        additional_fees=[],
        fee_components=fee_components,
        source_text=label,
        source_texts=source_texts,
        source_url=base_entry.source_url,
        source_fragments=[{"entry_id": entry_id, "text": text} for text, entry_id in fragment_text_to_id.items()],
        classification_status=classification_status,
        confidence=confidence,
        classification_evidence=[f"product={product_id}", f"variant={variant_id}"]
        + [f"{c.dimension}={c.value}" for c in conditions],
        fee_evidence=fee_evidence,
    )

    # Add reference notes for external fee components such as PayPal fees.
    if "paypal fees" in label.lower():
        rule = rule.model_copy(update={"additional_fees": ["PayPal fees (external; not included in Stripe rate)"]})

    if classification_status == CUSTOM_PRICING:
        return None, [
            item["entry"].model_copy(
                update={
                    "classification_status": CUSTOM_PRICING,
                    "confidence": 0.0,
                    "classification_evidence": ["group classified as custom pricing"],
                }
            )
            for item in group
        ]
    if classification_status in {CALCULABLE_RULE, INCLUDED, FREE, NON_CALCULABLE, REFERENCE_ONLY}:
        return rule, []
    return None, [
        item["entry"].model_copy(
            update={
                "classification_status": classification_status,
                "confidence": 0.0,
                "classification_evidence": [f"group classification_status={classification_status}"],
            }
        )
        for item in group
    ]


def _enrich_entries(entries: list[PricingEntry], account_country: str | None) -> list[dict[str, Any]]:
    """Add derived dimensions to each entry and resolve modifier inheritance."""
    # Use the original list index as a tie-breaker so sections with the same
    # source_order stay in document order.  Keep each crawled page contiguous so
    # cross-page source_order values do not interleave and break inheritance.
    sorted_entries = sorted(enumerate(entries), key=lambda item: (item[1].source_url, item[1].source_order, item[0]))
    enriched: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for _idx, entry in sorted_entries:
        product_id = _infer_product_id(entry)
        payment_method = _infer_payment_method(entry)
        # Add-on and non-payment products should not inherit a generic card
        # payment_method from their descriptive text.
        if not _is_payment_method_product(product_id):
            payment_method = None
        channel = _infer_channel(entry)
        card_region = _infer_card_region(entry, account_country)
        card_tier = _infer_card_tier(entry.source_text)

        # Entries without their own product/method context inherit from the
        # previous entry in the same section. This helps surcharge fragments
        # (e.g. "+ 2% if currency conversion is required") that follow a base
        # fee row but are not modifiers themselves.
        is_modifier = _is_modifier_entry(entry)
        if previous and tuple(previous["entry"].section_path) == tuple(entry.section_path):
            if is_modifier and product_id == "payments" and previous["product_id"] not in {"payments", "terminal"}:
                product_id = previous["product_id"]
            if (
                is_modifier
                and product_id == "unspecified"
                and previous["product_id"] not in {"unspecified", "payments", "terminal"}
            ):
                product_id = previous["product_id"]
            if product_id == previous["product_id"] or is_modifier:
                if not payment_method and previous["payment_method"]:
                    payment_method = previous["payment_method"]
                if not card_region and previous["card_region"]:
                    card_region = previous["card_region"]
                # Card tier should not bleed across base fee rows (e.g. standard
                # vs premium in the same table), only to continuation fragments.
                if (
                    not card_tier
                    and previous["card_tier"]
                    and _entry_component_hint(entry) not in {"base", "from", "range"}
                ):
                    card_tier = previous["card_tier"]

        card_origin = _card_origin_for_region(card_region, account_country)
        variant_id = _infer_variant_id(entry, product_id, account_country)

        enriched.append(
            {
                "entry": entry,
                "product_id": product_id,
                "payment_method": payment_method,
                "channel": channel,
                "card_region": card_region,
                "card_tier": card_tier,
                "card_origin": card_origin,
                "variant_id": variant_id,
                "is_modifier": is_modifier,
                "section_key": tuple(entry.section_path),
            }
        )
        previous = enriched[-1]
    return enriched


def _group_entries(entries: list[PricingEntry], account_country: str | None) -> list[list[dict[str, Any]]]:
    """Group related pricing fragments into logical rows."""
    enriched = _enrich_entries(entries, account_country)
    groups: list[list[dict[str, Any]]] = []
    for item in enriched:
        if not item["is_modifier"]:
            groups.append([item])
            continue
        attached = False
        for group in reversed(groups):
            last = group[-1]

            # Modifiers may appear in their own row-level heading (e.g. "$5.00 cap"
            # under "Payment methods/ACH Direct Debit ...").  Compare the parent
            # section path so the cap can attach to the matching base row.
            def _parent_key(key: tuple[str, ...]) -> tuple[str, ...]:
                return key[:-1] if len(key) > 1 else key

            if _parent_key(last["section_key"]) != _parent_key(item["section_key"]):
                continue
            if last["product_id"] != item["product_id"]:
                continue
            # Do not let an included/free statement attach to an unrelated
            # positive-fee base row (e.g. a 30% marketing fragment next to an
            # included-pricing block).
            base_entry = group[0]["entry"]
            base_hint = _entry_component_hint(base_entry)
            if (
                _entry_component_hint(item["entry"]) in {"included", "free"}
                and base_hint not in {"included", "free"}
                and _has_base_fee(base_entry)
            ):
                continue
            # Modifiers must anchor to a real fee row (or an included/free row),
            # never to marketing prose that happens to carry a number.
            if not _has_base_fee(base_entry) and base_hint not in {"included", "free"}:
                continue
            if item["variant_id"] == last["variant_id"]:
                group.append(item)
                attached = True
                break
            # A generic modifier can attach to a more specific base variant.
            if item["variant_id"] in {"online_domestic_cards", "domestic_cards", "standard"}:
                group.append(item)
                attached = True
                break
        if not attached:
            # No matching base row; treat as its own base entry.
            item = dict(item)
            item["is_modifier"] = False
            groups.append([item])
    return groups


__all__ = ["_empty_group_rule", "_base_conditions", "_classify_group", "_enrich_entries", "_group_entries"]
