"""Regression detection for published Stripe fee data."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from .models import ChangeReport, ChangeType

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _country_code(item: dict[str, Any]) -> str | None:
    code = item.get("account_country") or item.get("stripe_market_code")
    if code:
        return code.upper()
    return None


def _market_set(data_dir: Path) -> set[str]:
    """Return the full set of discovered market country codes."""
    manifest_path = data_dir / "meta" / "markets.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        codes: set[str] = set()
        for item in manifest.get("markets", []):
            code = _country_code(item)
            if code:
                codes.add(code)
        for item in manifest.get("unsupported", []):
            code = _country_code(item)
            if code:
                codes.add(code)
        for item in manifest.get("transient_failures", []):
            code = _country_code(item)
            if code:
                codes.add(code)
        return codes
    # Fallback for older/test repositories without a manifest.
    return _supported_set(data_dir) | _unsupported_set(data_dir) | _transient_set(data_dir)


def _supported_set(data_dir: Path) -> set[str]:
    """Return country codes of supported markets with a fee page."""
    index_path = data_dir / "json" / "index.json"
    if not index_path.exists():
        return set()
    index = _load_json(index_path)
    codes: set[str] = set()
    for entry in index.get("markets", []):
        status = entry.get("derivation_status")
        if status in {"complete", "partial"}:
            code = entry.get("account_country")
            if code:
                codes.add(code.upper())
    return codes


def _unsupported_set(data_dir: Path) -> set[str]:
    path = data_dir / "meta" / "unsupported-markets.json"
    if not path.exists():
        return set()
    data = _load_json(path)
    return {code for item in data if (code := _country_code(item))}


def _transient_set(data_dir: Path) -> set[str]:
    path = data_dir / "meta" / "transient-failures.json"
    if not path.exists():
        return set()
    data = _load_json(path)
    return {code for item in data if (code := _country_code(item))}


def _market_data(data_dir: Path, country: str) -> dict[str, Any]:
    path = data_dir / "json" / f"{country}.json"
    if not path.exists():
        return {}
    return _load_json(path)


def _entry_count(data: dict[str, Any]) -> int:
    return len(data.get("entries", [])) + len(data.get("derived_rules", [])) + len(data.get("unclassified_entries", []))


def _rule_count(data: dict[str, Any]) -> int:
    return len(data.get("derived_rules", []))


def _section_count(data: dict[str, Any]) -> int:
    return len(data.get("sections", []))


def _rule_value(rule: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (rule.get("percentage"), rule.get("fixed_amount"), rule.get("fixed_currency"))


def _safe_decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(value)
    except Exception:
        return None


def _relative_change(before: Decimal, after: Decimal) -> float:
    if before == 0:
        return float("inf") if after != 0 else 0.0
    return float((after - before) / before)


def _detect_changes(old_dir: Path, new_dir: Path, thresholds: dict[str, Any] | None = None) -> list[ChangeType]:
    changes: list[ChangeType] = []
    if thresholds is None:
        thresholds = {
            "section_drop_ratio": 0.5,
            "entry_drop_ratio": 0.5,
            "rule_drop_ratio": 0.5,
            "market_drop_ratio": 0.2,
            "percentage_change_ratio": 0.5,
            "fixed_change_ratio": 0.5,
        }

    old_markets = _market_set(old_dir)
    new_markets = _market_set(new_dir)
    old_supported = _supported_set(old_dir)
    new_supported = _supported_set(new_dir)
    new_unsupported = _unsupported_set(new_dir)
    new_transient = _transient_set(new_dir)

    removed = old_markets - new_markets
    added = new_markets - old_markets

    for country in sorted(removed):
        changes.append(
            ChangeType(
                kind="removed_market",
                country_code=country,
                message=f"Market {country} was present in the previous dataset but missing now",
            )
        )
    for country in sorted(added):
        changes.append(
            ChangeType(
                kind="new_market",
                country_code=country,
                message=f"Market {country} is newly supported",
            )
        )

    if old_markets and len(removed) / len(old_markets) > thresholds["market_drop_ratio"]:
        changes.append(
            ChangeType(
                kind="sharp_market_drop",
                before=len(old_markets),
                after=len(new_markets),
                message="Sharp drop in supported markets",
            )
        )

    for country in sorted(old_markets & new_markets):
        old_data = _market_data(old_dir, country)
        new_data = _market_data(new_dir, country)
        old_sections = _section_count(old_data)
        new_sections = _section_count(new_data)
        old_entries = _entry_count(old_data)
        new_entries = _entry_count(new_data)
        old_rules = _rule_count(old_data)
        new_rules = _rule_count(new_data)

        if old_sections and new_sections / old_sections < 1 - thresholds["section_drop_ratio"]:
            changes.append(
                ChangeType(
                    kind="sharp_section_drop",
                    country_code=country,
                    before=old_sections,
                    after=new_sections,
                    message=f"Section count for {country} dropped sharply",
                )
            )
        if old_entries and new_entries / old_entries < 1 - thresholds["entry_drop_ratio"]:
            changes.append(
                ChangeType(
                    kind="sharp_entry_drop",
                    country_code=country,
                    before=old_entries,
                    after=new_entries,
                    message=f"Entry count for {country} dropped sharply",
                )
            )
        if old_rules and new_rules == 0:
            changes.append(
                ChangeType(
                    kind="lost_core_category",
                    country_code=country,
                    before=old_rules,
                    after=0,
                    message=f"All derived rules disappeared for {country}",
                )
            )
        elif old_rules and new_rules / old_rules < 1 - thresholds["rule_drop_ratio"]:
            changes.append(
                ChangeType(
                    kind="sharp_rule_drop",
                    country_code=country,
                    before=old_rules,
                    after=new_rules,
                    message=f"Rule count for {country} dropped sharply",
                )
            )

        old_status = old_data.get("derivation_status")
        new_status = new_data.get("derivation_status")
        if old_status in {"complete", "partial"} and new_status == "unclassified":
            changes.append(
                ChangeType(
                    kind="classified_to_unclassified",
                    country_code=country,
                    before=old_status,
                    after=new_status,
                    message=f"Derivation status for {country} regressed to unclassified",
                )
            )

        # Compare rule values for suspiciously large changes.
        old_rules_by_id = {r.get("rule_id"): r for r in old_data.get("derived_rules", []) if r.get("rule_id")}
        new_rules_by_id = {r.get("rule_id"): r for r in new_data.get("derived_rules", []) if r.get("rule_id")}
        for rule_id in set(old_rules_by_id) & set(new_rules_by_id):
            old_rule = old_rules_by_id[rule_id]
            new_rule = new_rules_by_id[rule_id]
            old_pct = _safe_decimal(old_rule.get("percentage"))
            new_pct = _safe_decimal(new_rule.get("percentage"))
            if old_pct is not None and new_pct is not None:
                rel = _relative_change(old_pct, new_pct)
                if abs(rel) > thresholds["percentage_change_ratio"]:
                    changes.append(
                        ChangeType(
                            kind="large_percentage_change",
                            country_code=country,
                            identifier=rule_id,
                            before=str(old_pct),
                            after=str(new_pct),
                            message=f"Percentage for {rule_id} changed by {rel:.0%}",
                        )
                    )
            old_fixed = _safe_decimal(old_rule.get("fixed_amount"))
            new_fixed = _safe_decimal(new_rule.get("fixed_amount"))
            if old_fixed is not None and new_fixed is not None and old_fixed != 0:
                rel = _relative_change(old_fixed, new_fixed)
                if abs(rel) > thresholds["fixed_change_ratio"]:
                    changes.append(
                        ChangeType(
                            kind="large_fixed_change",
                            country_code=country,
                            identifier=rule_id,
                            before=str(old_fixed),
                            after=str(new_fixed),
                            message=f"Fixed amount for {rule_id} changed by {rel:.0%}",
                        )
                    )
            if old_rule.get("fixed_currency") != new_rule.get("fixed_currency"):
                changes.append(
                    ChangeType(
                        kind="currency_changed",
                        country_code=country,
                        identifier=rule_id,
                        before=old_rule.get("fixed_currency"),
                        after=new_rule.get("fixed_currency"),
                        message=f"Currency for {rule_id} changed",
                    )
                )

        # Detect duplicate stable identifiers within the new dataset.
        seen_ids: set[str] = set()
        for rule in new_data.get("derived_rules", []):
            rule_id = rule.get("rule_id")
            if rule_id and rule_id in seen_ids:
                changes.append(
                    ChangeType(
                        kind="duplicate_identifier",
                        country_code=country,
                        identifier=rule_id,
                        message=f"Duplicate rule identifier {rule_id} in {country}",
                    )
                )
            seen_ids.add(rule_id)

    # Compare supported-to-unsupported/transient transitions.
    for code in sorted(old_supported - new_supported):
        if code in new_unsupported:
            changes.append(
                ChangeType(
                    kind="supported_to_unsupported",
                    country_code=code,
                    message=f"{code} moved from supported to unsupported",
                )
            )
        elif code in new_transient:
            changes.append(
                ChangeType(
                    kind="supported_to_transient",
                    country_code=code,
                    message=f"{code} moved from supported to transient failure",
                )
            )
        elif code not in new_markets:
            changes.append(
                ChangeType(
                    kind="discovered_to_missing",
                    country_code=code,
                    message=f"{code} disappeared from the discovered market set",
                )
            )

    return changes


def check_regression(
    old_data_dir: str | Path,
    new_data_dir: str | Path,
    thresholds: dict[str, Any] | None = None,
) -> ChangeReport:
    """Compare two published datasets and produce a deterministic change report."""
    old_dir = Path(old_data_dir)
    new_dir = Path(new_data_dir)
    changes = _detect_changes(old_dir, new_dir, thresholds)
    return ChangeReport(
        schema_version=1,
        generated_at=None,
        changes=changes,
        has_regression=any(
            c.kind
            in {
                "removed_market",
                "discovered_to_missing",
                "supported_to_transient",
                "supported_to_unsupported",
                "removed_section",
                "removed_entry",
                "lost_core_category",
                "structural_regression",
                "sharp_section_drop",
                "sharp_entry_drop",
                "sharp_rule_drop",
                "sharp_market_drop",
                "classified_to_unclassified",
                "fee_value_disappeared",
                "duplicate_identifier",
                "currency_changed",
                "source_url_changed",
                "large_percentage_change",
                "large_fixed_change",
                "schema_incompatible",
                "parser_output_empty",
            }
            for c in changes
        ),
    )
