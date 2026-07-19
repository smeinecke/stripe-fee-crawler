"""Rule deduplication, merging and coverage helpers."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..models import CoverageSummary, FeeComponent, FeeCondition, FeeRule, PricingEntry
from ..normalize import stable_id
from ._util import _dedup_repeated_phrases, _has_base_fee, _is_explicit_fee_phrase, _ordered_unique
from .evidence import _is_market_share_text
from .tables import (
    AMBIGUOUS,
    CALCULABLE_RULE,
    CUSTOM_PRICING,
    FREE,
    IGNORED_NON_FEE,
    INCLUDED,
    INFORMATIONAL,
    NON_CALCULABLE,
    REFERENCE_ONLY,
    UNCLASSIFIED_CANDIDATE,
    UNSUPPORTED_SHAPE,
)

logger = logging.getLogger(__name__)


def _condition_key(conditions: list[FeeCondition]) -> tuple[tuple[str, str, Any], ...]:
    return tuple(sorted((c.dimension, c.operator, str(c.value)) for c in conditions))


def _fee_signature(rule: FeeRule) -> tuple[Any, ...]:
    """Return a normalized signature covering the full fee definition.

    The signature includes every fee component (percentage, fixed amount,
    surcharges, caps, floors), the unit, exactness, behavior and the pricing
    plan so that two rules with the same selector but different fee shapes are
    not silently collapsed.
    """
    components = sorted(
        rule.fee_components,
        key=lambda c: (
            c.type or "",
            c.value or "",
            c.amount or "",
            c.currency or "",
            c.basis_points or "",
            c.operator or "",
            c.minor_amount or "",
        ),
    )
    comp_tuples = tuple(
        (c.type, c.value, c.amount, c.currency, c.basis_points, c.operator, c.minor_amount) for c in components
    )
    pricing_plan = next((c.value for c in rule.conditions if c.dimension == "pricing_plan"), None)
    return (
        comp_tuples,
        rule.unit,
        rule.exactness,
        rule.behavior,
        pricing_plan,
    )


def _merge_rules(rules: list[FeeRule]) -> FeeRule:
    """Merge rules that share a selector and fee signature into one rule.

    Provenance (entry IDs, source text, fragments and evidence) is combined;
    the fee definition itself is taken from the first rule because the fee
    signature is identical.
    """
    base = rules[0]

    # Contributing entry IDs in first-seen order.
    seen_entry_ids: set[str] = set()
    contributing_entry_ids: list[str] = []
    for r in rules:
        for eid in r.contributing_entry_ids:
            if eid not in seen_entry_ids:
                seen_entry_ids.add(eid)
                contributing_entry_ids.append(eid)

    # Source texts and evidence phrases, deduplicated.
    source_texts = _ordered_unique(_dedup_repeated_phrases(t) for r in rules for t in r.source_texts if t)
    label = base.label
    source_text = base.source_text or label

    # Use the strongest evidence as the base and merge provenance from all rules.
    evidence_rules = [r for r in rules if r.fee_evidence is not None]
    best_evidence_rule = (
        max(
            evidence_rules,
            key=lambda r: (
                {"calculable_rule": 3, "classified": 2, "non_calculable": 1}.get(r.classification_status, 0),
                r.confidence,
            ),
        )
        if evidence_rules
        else base
    )
    fee_evidence = best_evidence_rule.fee_evidence
    if fee_evidence is not None:
        evidence_phrases = _ordered_unique(
            _dedup_repeated_phrases(p)
            for r in evidence_rules
            if r.fee_evidence is not None
            for p in r.fee_evidence.phrases
            if p
        )
        evidence_entry_ids = _ordered_unique(
            eid for r in evidence_rules if r.fee_evidence is not None for eid in r.fee_evidence.source_entry_ids
        )
        fee_evidence = fee_evidence.model_copy(
            update={"phrases": evidence_phrases, "source_entry_ids": evidence_entry_ids}
        )

    # Source fragments by text.
    fragment_by_text: dict[str, str | None] = {}
    for r in rules:
        for frag in r.source_fragments:
            text = frag.get("text")
            if text and text not in fragment_by_text:
                fragment_by_text[text] = frag.get("entry_id")
    source_fragments = [{"entry_id": entry_id, "text": text} for text, entry_id in fragment_by_text.items()]

    # Fee components are identical by signature; sort for deterministic output.
    fee_components = sorted(
        base.fee_components,
        key=lambda c: (
            c.type or "",
            c.value or "",
            c.amount or "",
            c.currency or "",
            c.basis_points or "",
            c.operator or "",
            c.minor_amount or "",
        ),
    )

    # Preserve the most concrete classification state and highest confidence.
    status_priority = {"calculable_rule": 3, "classified": 2, "non_calculable": 1}
    classification_status = max(
        rules, key=lambda r: status_priority.get(r.classification_status, 0)
    ).classification_status
    confidence = max(r.confidence for r in rules)

    # Recompute the rule id from the stable selector + fee signature.
    rule_id = stable_id(
        base.product_id or "",
        base.variant_id or "",
        *[f"{c.dimension}={c.value}" for c in sorted(base.conditions, key=lambda x: x.dimension)],
        str(_fee_signature(base)),
    )

    classification_evidence = _ordered_unique(item for r in rules for item in r.classification_evidence if item)

    return base.model_copy(
        update={
            "rule_id": rule_id,
            "entry_id": base.entry_id,
            "contributing_entry_ids": contributing_entry_ids,
            "label": label,
            "source_text": source_text,
            "source_texts": source_texts,
            "source_fragments": source_fragments,
            "fee_components": fee_components,
            "classification_status": classification_status,
            "confidence": confidence,
            "classification_evidence": classification_evidence,
            "fee_evidence": fee_evidence,
        }
    )


def _is_fee_candidate(rule: FeeRule) -> bool:
    """Return True when a rule represents a concrete fee candidate.

    Only calculable, classified and genuine (numeric) non-calculable rules are
    fee candidates.  Informational, custom-pricing, ignored or unclassified
    statuses may carry numbers but are not fee definitions and must not force a
    conflict with real fee rules.
    """
    if rule.classification_status in {CALCULABLE_RULE, "classified"}:
        return True
    if rule.classification_status != NON_CALCULABLE:
        return False
    return any(
        c.type
        in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge", "minimum_fee", "maximum_fee"}
        and (c.value is not None or c.amount is not None)
        for c in rule.fee_components
    )


def _merge_rule_provenance(base: FeeRule, other: FeeRule) -> FeeRule:
    """Merge source provenance from ``other`` into ``base`` without changing fees."""
    seen_ids: set[str] = set(base.contributing_entry_ids)
    contributing_entry_ids = list(base.contributing_entry_ids)
    for eid in other.contributing_entry_ids:
        if eid not in seen_ids:
            seen_ids.add(eid)
            contributing_entry_ids.append(eid)

    source_texts = _ordered_unique(_dedup_repeated_phrases(t) for t in base.source_texts + other.source_texts if t)

    fragments: dict[str, str | None] = {}
    for frag in base.source_fragments + other.source_fragments:
        text = frag.get("text")
        if text and text not in fragments:
            fragments[text] = frag.get("entry_id")
    source_fragments = [{"entry_id": entry_id, "text": text} for text, entry_id in fragments.items()]

    fee_evidence = base.fee_evidence
    if fee_evidence is not None and other.fee_evidence is not None:
        phrases = _ordered_unique(
            _dedup_repeated_phrases(p) for p in fee_evidence.phrases + other.fee_evidence.phrases if p
        )
        entry_ids = _ordered_unique(eid for eid in fee_evidence.source_entry_ids + other.fee_evidence.source_entry_ids)
        fee_evidence = fee_evidence.model_copy(update={"phrases": phrases, "source_entry_ids": entry_ids})
    elif other.fee_evidence is not None:
        fee_evidence = other.fee_evidence

    classification_evidence = _ordered_unique(base.classification_evidence + other.classification_evidence)

    return base.model_copy(
        update={
            "contributing_entry_ids": contributing_entry_ids,
            "source_texts": source_texts,
            "source_fragments": source_fragments,
            "classification_evidence": classification_evidence,
            "fee_evidence": fee_evidence,
        }
    )


def _as_conflict_rule(rule: FeeRule) -> FeeRule:
    """Return a copy of ``rule`` marked as a conflict diagnostic."""
    return rule.model_copy(
        update={
            "classification_status": "conflict",
            "confidence": 0.0,
            "classification_evidence": list(rule.classification_evidence)
            + ["conflict: semantic identity with differing fee signature"],
        }
    )


def _component_key(component: FeeComponent) -> tuple[str, ...]:
    return (
        component.type or "",
        component.value or "",
        str(component.amount) if component.amount is not None else "",
        component.currency or "",
        str(component.basis_points) if component.basis_points is not None else "",
        component.operator or "",
        str(component.minor_amount) if component.minor_amount is not None else "",
    )


def _is_component_subset(small: FeeRule, large: FeeRule) -> bool:
    """Return True when ``small``'s fee components are all present in ``large``."""
    large_keys = {_component_key(c) for c in large.fee_components}
    return all(_component_key(c) in large_keys for c in small.fee_components)


def _merge_subset_groups(groups: list[FeeRule]) -> list[FeeRule]:
    """Absorb partial duplicate groups into a more complete sibling group.

    A duplicate base fee row that lacks an attached cap/minimum is a subset of
    the same base fee row that includes the modifier.  Merging them keeps the
    full rule while preserving all provenance.
    """
    result = list(groups)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(result):
            for j, b in enumerate(result):
                if i == j:
                    continue
                if _is_component_subset(a, b):
                    merged = _merge_rule_provenance(b, a)
                    result[j] = merged
                    result.pop(i)
                    changed = True
                    break
            if changed:
                break
    return result


def _deduplicate_rules(rules: list[FeeRule]) -> list[FeeRule]:
    """Resolve rules with the same semantic identity.

    Rules that share product_id + variant_id + conditions and also share the
    same fee signature are merged.  Rules that share the selector but differ in
    fee signature are kept as ``conflict`` candidates so nothing is silently
    discarded.
    """
    # order preserves first-seen selector order; buckets map a selector to a
    # list of rule groups, each group sharing one fee signature.
    order: list[tuple[str, str | None, Any]] = []
    buckets: dict[
        tuple[str, str | None, Any],
        list[list[FeeRule]],
    ] = {}

    for rule in rules:
        selector = (rule.product_id or "", rule.variant_id, _condition_key(rule.conditions))
        sig = _fee_signature(rule)
        if selector not in buckets:
            buckets[selector] = []
            order.append(selector)
        groups = buckets[selector]
        for group in groups:
            if _fee_signature(group[0]) == sig:
                group.append(rule)
                break
        else:
            groups.append([rule])

    result: list[FeeRule] = []
    for selector in order:
        sig_groups = buckets[selector]
        # Merge within each signature group first.
        merged_groups = [_merge_rules(g) for g in sig_groups]

        # A partial duplicate (e.g. a duplicate base row missing its cap) is a
        # subset of the same row that includes the modifier.  Absorb it so it
        # does not create a spurious conflict.
        merged_groups = _merge_subset_groups(merged_groups)

        # Rules for an unrecognized product with multiple conflicting fee
        # shapes are informational by definition.  Collapse those groups into
        # a single informational diagnostic so unrelated marketing numbers do
        # not create spurious conflicts.  A lone unspecified group keeps its
        # original non-calculable status.
        if selector[0] == "unspecified" and len(merged_groups) > 1:
            final = merged_groups[0]
            for other in merged_groups[1:]:
                final = _merge_rule_provenance(final, other)
            final = final.model_copy(
                update={
                    "rule_id": stable_id(
                        selector[0] or "",
                        selector[1] or "",
                        *[f"{c.dimension}={c.value}" for c in sorted(final.conditions, key=lambda x: x.dimension)],
                        "informational",
                    ),
                    "classification_status": "informational",
                    "confidence": 0.0,
                    "fee_components": [],
                    "percentage": None,
                    "basis_points": None,
                    "fixed_amount": None,
                    "fixed_amount_minor": None,
                    "fixed_currency": None,
                    "minimum_amount": None,
                    "maximum_amount": None,
                    "unit": "informational",
                    "behavior": "informational",
                }
            )
            result.append(final)
            continue

        fee_candidates = [r for r in merged_groups if _is_fee_candidate(r)]
        non_fee_groups = [r for r in merged_groups if not _is_fee_candidate(r)]

        if len(fee_candidates) == 1:
            # One clear fee: attach non-fee fragments to it for provenance.
            final = fee_candidates[0]
            for ng in non_fee_groups:
                final = _merge_rule_provenance(final, ng)
            result.append(final)
        elif len(fee_candidates) > 1:
            # Multiple real fee definitions for the same selector: publish none
            # as authoritative and keep every candidate as a conflict.
            for r in fee_candidates:
                result.append(_as_conflict_rule(r))
            for r in non_fee_groups:
                result.append(_as_conflict_rule(r))
        else:
            # No real fee candidate; emit the non-fee groups.  Merge them by
            # provenance so duplicate identities do not leak into the output.
            if non_fee_groups:
                final = non_fee_groups[0]
                for ng in non_fee_groups[1:]:
                    final = _merge_rule_provenance(final, ng)
                result.append(final)
    return result


def _is_marketing_or_statistical(entry: PricingEntry) -> bool:
    """Return True when an entry is clearly marketing or a platform statistic."""
    text = (entry.source_text + " " + (entry.source_evidence or "")).lower()
    source_lower = entry.source_text.lower()
    # Market/adoption statistics are never merchant fees, even when they also
    # contain fee-adjacent words such as "pricing".
    if _is_market_share_text(entry.source_text):
        return True
    # A row that already describes a concrete fee should never be discarded.
    if _is_explicit_fee_phrase(text) and _has_base_fee(entry):
        return False
    return (
        ("99.999%" in text and "uptime" in text)
        or "250 million" in text
        or "api requests" in text
        or "maximise your revenue" in text
        or "maximize your revenue" in text
        or ("pci compliant" in text and not re.search(r"\d+(?:\.\d+)?%?\s*(?:per|month|year|transaction)", text))
        or "return on investment" in text
        or "category leaders" in text
        or "process more than" in text
        or "process over" in text
        or (source_lower.startswith("included:") and not _has_base_fee(entry))
        or "from startups to fortune" in text
        or "start building your integration" in text
        or "explore how" in text
        or "chose stripe" in text
        or "simplified cross-border payments with stripe" in text
        or ("migrated" in text and "months" in text)
        or "of subscription volume" in text
        or "of customers" in text
        or "tapped into" in text
        or "increased" in text
        and "with stripe" in text
        or "learn why" in text
        or "read the docs" in text
        or "customers use stripe" in text
    )


def _derive_status(rules: list[FeeRule], unclassified: list[PricingEntry]) -> str:
    authoritative = [r for r in rules if r.classification_status != "conflict"]
    unresolved = [
        e
        for e in unclassified
        if e.classification_status
        not in {IGNORED_NON_FEE, INFORMATIONAL, REFERENCE_ONLY, CUSTOM_PRICING, INCLUDED, FREE, UNSUPPORTED_SHAPE}
    ]
    if not rules and not unclassified:
        return "unclassified"
    if not authoritative:
        return "partial"
    if not unresolved:
        return "complete"
    return "partial"


def _numeric_source_entries(entries: list[PricingEntry]) -> list[PricingEntry]:
    """Return entries that carry a numeric fee value, excluding marketing stats."""
    return [
        e
        for e in entries
        if _has_base_fee(e)
        and e.classification_status
        not in {IGNORED_NON_FEE, INFORMATIONAL, REFERENCE_ONLY, CUSTOM_PRICING, INCLUDED, FREE, UNSUPPORTED_SHAPE}
    ]


def _coverage_summary(
    entries: list[PricingEntry],
    rules: list[FeeRule],
    unclassified: list[PricingEntry],
) -> CoverageSummary:
    numeric_entries = _numeric_source_entries(entries)
    referenced_ids: set[str] = set()
    for rule in rules:
        for eid in rule.contributing_entry_ids:
            referenced_ids.add(eid)
    for entry in unclassified:
        referenced_ids.add(entry.entry_id)
    referenced_numeric = [e for e in numeric_entries if e.entry_id in referenced_ids]
    dropped_numeric = len(numeric_entries) - len(referenced_numeric)

    counts: dict[str, int] = {
        "source_entries": len(entries),
        "numeric_source_entries": len(numeric_entries),
        "referenced_numeric_entries": len(referenced_numeric),
        "dropped_numeric_entries": dropped_numeric,
    }

    rule_status_fields = {
        CALCULABLE_RULE: "calculable_rules",
        NON_CALCULABLE: "non_calculable_rules",
        "conflict": "conflicting_rule_identities",
        INCLUDED: "included",
        FREE: "free",
        INFORMATIONAL: "informational",
        CUSTOM_PRICING: "custom_pricing",
        UNSUPPORTED_SHAPE: "unsupported_fee_shapes",
    }
    entry_status_fields = {
        UNCLASSIFIED_CANDIDATE: "unclassified_fee_candidates",
        AMBIGUOUS: "ambiguous_entries",
        UNSUPPORTED_SHAPE: "unsupported_fee_shapes",
        IGNORED_NON_FEE: "ignored_non_fee",
        REFERENCE_ONLY: "reference_only",
        INCLUDED: "included",
        FREE: "free",
        INFORMATIONAL: "informational",
        CUSTOM_PRICING: "custom_pricing",
    }

    for rule in rules:
        status = rule.classification_status
        if status == "conflict":
            counts["conflicting_rule_identities"] = counts.get("conflicting_rule_identities", 0) + 1
            if _is_fee_candidate(rule):
                counts["blocking_fee_conflicts"] = counts.get("blocking_fee_conflicts", 0) + 1
            else:
                counts["informational_conflicts"] = counts.get("informational_conflicts", 0) + 1
            continue
        field = rule_status_fields.get(status)
        if field:
            counts[field] = counts.get(field, 0) + 1

    for entry in unclassified:
        field = entry_status_fields.get(entry.classification_status)
        if field:
            counts[field] = counts.get(field, 0) + 1

    numeric_candidates = [
        e
        for e in unclassified
        if _has_base_fee(e)
        and e.classification_status
        not in {IGNORED_NON_FEE, INFORMATIONAL, REFERENCE_ONLY, CUSTOM_PRICING, INCLUDED, FREE, UNSUPPORTED_SHAPE}
    ]
    counts["numeric_fee_candidates"] = len(numeric_candidates)
    return CoverageSummary(**counts)


def _calculator_coverage_status(
    entries: list[PricingEntry],
    rules: list[FeeRule],
    unclassified: list[PricingEntry],
) -> str:
    authoritative = [r for r in rules if r.classification_status not in {"conflict"}]
    if not authoritative:
        if any(r.classification_status == "conflict" for r in rules) or unclassified:
            return "partial"
        return "unclassified"
    # If any remaining unclassified entry has a numeric fee value, coverage is partial.
    # Custom-priced, included/free, and explicitly unsupported-shape entries are
    # considered resolved.
    resolved_unclassified_statuses = {
        IGNORED_NON_FEE,
        INFORMATIONAL,
        REFERENCE_ONLY,
        CUSTOM_PRICING,
        INCLUDED,
        FREE,
        UNSUPPORTED_SHAPE,
    }
    numeric_unclassified = [
        e for e in unclassified if _has_base_fee(e) and e.classification_status not in resolved_unclassified_statuses
    ]
    if numeric_unclassified:
        return "partial"
    return "complete"


__all__ = [
    "_condition_key",
    "_fee_signature",
    "_merge_rules",
    "_is_fee_candidate",
    "_merge_rule_provenance",
    "_as_conflict_rule",
    "_component_key",
    "_is_component_subset",
    "_merge_subset_groups",
    "_deduplicate_rules",
    "_is_marketing_or_statistical",
    "_derive_status",
    "_numeric_source_entries",
    "_coverage_summary",
    "_calculator_coverage_status",
]
