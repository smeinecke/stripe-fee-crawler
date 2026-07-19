"""Public classification API."""

from __future__ import annotations

import logging

from ..models import CoverageSummary, FeeRule, Market, PricingEntry
from ._util import _account_country_from_url, _fixed_amount_minor, _has_base_fee
from .dedup import (
    _calculator_coverage_status,
    _coverage_summary,
    _deduplicate_rules,
    _derive_status,
    _is_marketing_or_statistical,
)
from .evidence import _is_promotional_language
from .grouping import _classify_group, _group_entries
from .tables import (
    AMBIGUOUS,
    CUSTOM_PRICING,
    FREE,
    IGNORED_NON_FEE,
    INCLUDED,
    INFORMATIONAL,
    REFERENCE_ONLY,
    UNCLASSIFIED_CANDIDATE,
    UNSUPPORTED_SHAPE,
)

logger = logging.getLogger(__name__)


def classify_entries(
    entries: list[PricingEntry],
    account_country: str | None = None,
) -> tuple[list[FeeRule], list[PricingEntry]]:
    """Classify pricing entries into derived rules and unclassified leftovers."""
    if account_country is None and entries:
        account_country = _account_country_from_url(entries[0].source_url)

    rules: list[FeeRule] = []
    unclassified: list[PricingEntry] = []

    # Marketing and platform-statistic entries (99.999% uptime, 250M API
    # requests, etc.) carry numbers but are not merchant fees.  Exclude them
    # from grouping so they never become fee rules.
    filtered_entries: list[PricingEntry] = []
    for entry in entries:
        if _is_marketing_or_statistical(entry):
            unclassified.append(
                entry.model_copy(
                    update={
                        "classification_status": IGNORED_NON_FEE,
                        "confidence": 0.0,
                        "classification_evidence": ["marketing or platform statistic, not a fee"],
                    }
                )
            )
        else:
            filtered_entries.append(entry)

    groups = _group_entries(filtered_entries, account_country)
    for group in groups:
        rule, leftovers = _classify_group(group, account_country)
        if rule:
            rules.append(rule)
        for leftover in leftovers:
            status = leftover.classification_status or UNCLASSIFIED_CANDIDATE
            if status not in {
                CUSTOM_PRICING,
                INCLUDED,
                FREE,
                INFORMATIONAL,
                REFERENCE_ONLY,
                UNSUPPORTED_SHAPE,
                AMBIGUOUS,
            }:
                # Non-numeric text left over after grouping is informational by
                # default; promotional text without a concrete price is custom.
                if _is_promotional_language(leftover.source_text):
                    status = CUSTOM_PRICING
                elif not _has_base_fee(leftover):
                    status = INFORMATIONAL
                else:
                    status = UNCLASSIFIED_CANDIDATE
            leftover = leftover.model_copy(
                update={
                    "classification_status": status,
                    "confidence": 0.0,
                    "classification_evidence": ["no clear calculable fee or unsupported shape"],
                }
            )
            unclassified.append(leftover)

    # Collapse duplicate semantic identities from repeated sections (e.g.
    # "Standard pricing" mirroring "Payments").
    rules = _deduplicate_rules(rules)
    return rules, unclassified


def derive_market_fees(
    entries: list[PricingEntry],
    market: Market | None = None,
    account_country: str | None = None,
) -> tuple[list[FeeRule], list[PricingEntry], str, CoverageSummary, str]:
    """Derive all rules for a market and compute coverage."""
    if market is not None:
        account_country = market.account_country
    rules, unclassified = classify_entries(entries, account_country=account_country)
    status = _derive_status(rules, unclassified)
    coverage = _coverage_summary(entries, rules, unclassified)
    calc_status = _calculator_coverage_status(entries, rules, unclassified)
    return rules, unclassified, status, coverage, calc_status


__all__ = ["classify_entries", "derive_market_fees", "_fixed_amount_minor"]
