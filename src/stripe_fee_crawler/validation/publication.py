"""Publication-level and repository validation checks."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .._common import _condition_key_data, _read_json
from .._git import _crawler_revision
from ..exceptions import ValidationError as CrawlerValidationError
from ..market_detection import _detect_market_from_path
from ..models import (
    ChangeReport,
    CoreFeeRule,
    CoreFees,
    CrawlReport,
    MarketIndex,
    MarketManifest,
    MarketOutput,
    SchemaVersionInfo,
    UnsupportedMarket,
)
from .schemas import (
    validate_core_fees,
    validate_index,
    validate_manifest,
    validate_market_output,
    validate_payment_methods,
)
from .semantic_rules import _is_calculable_status, validate_semantic

logger = logging.getLogger(__name__)


def _validate_publication(output_dir: Path) -> list[str]:
    """Strict publication checks for the generated data repository."""
    errors: list[str] = []
    json_dir = output_dir / "json"
    core_fees_path = json_dir / "core-fees.json"

    market_has_derived_rules = False
    market_all_non_calculable: list[str] = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name in {"index.json", "core-fees.json", "payment-methods.json"}:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            errors.append(f"{path.name}: cannot read market output: {exc}")
            continue
        coverage = data.get("coverage_summary", {})
        dropped = coverage.get("dropped_numeric_entries", 0)
        if dropped:
            errors.append(f"{path.name}: dropped {dropped} numeric source entries")
        blocking_conflicts = coverage.get("blocking_fee_conflicts", 0)
        if blocking_conflicts:
            errors.append(f"{path.name}: {blocking_conflicts} blocking fee conflict(s) remain unresolved")
        rules = data.get("derived_rules", [])
        # Detect a silent first-value-wins or partial conflict situation:
        # a selector may not simultaneously publish authoritative and conflict
        # candidates.
        identities: dict[tuple[str, str | None, Any], list[dict[str, Any]]] = {}
        for r in rules:
            identity = (r.get("product_id") or "", r.get("variant_id"), _condition_key_data(r.get("conditions", [])))
            identities.setdefault(identity, []).append(r)
        for identity, identity_rules in identities.items():
            statuses = {r.get("classification_status") for r in identity_rules}
            if "conflict" in statuses and statuses != {"conflict"}:
                ids = [r.get("rule_id") for r in identity_rules]
                errors.append(
                    f"{path.name}: semantic identity {identity} has both authoritative and conflict rules: {ids}"
                )

        non_conflict = [r for r in rules if r.get("classification_status") != "conflict"]
        if non_conflict:
            market_has_derived_rules = True
            if not any(_is_calculable_status(r.get("classification_status", "")) for r in non_conflict):
                market_all_non_calculable.append(path.stem)

    if market_all_non_calculable:
        errors.append(f"All derived rules are non-calculable for markets: {', '.join(market_all_non_calculable)}")

    core_rule_count = 0
    if core_fees_path.exists():
        try:
            with open(core_fees_path, encoding="utf-8") as fh:
                core_fees = json.load(fh)
            core_rule_count = sum(len(m.get("rules", [])) for m in core_fees.get("markets", []))
        except Exception as exc:
            errors.append(f"core-fees.json: cannot read: {exc}")

    if market_has_derived_rules and core_rule_count == 0:
        errors.append("derived rules exist but core-fees.json contains no calculable rules")

    change_report_path = output_dir / "change-report.json"
    if not change_report_path.exists():
        errors.append("change-report.json is missing")
    else:
        try:
            with open(change_report_path, encoding="utf-8") as fh:
                change_report = ChangeReport.model_validate(json.load(fh))
            if change_report.has_regression:
                errors.append("change-report.json has_regression is true")
        except Exception as exc:
            errors.append(f"change-report.json: not a valid ChangeReport: {exc}")

    crawl_report_path = output_dir / "meta" / "crawl-report.json"
    if not crawl_report_path.exists():
        errors.append("meta/crawl-report.json is missing")
    else:
        try:
            with open(crawl_report_path, encoding="utf-8") as fh:
                CrawlReport.model_validate(json.load(fh))
        except Exception as exc:
            errors.append(f"meta/crawl-report.json: not a valid CrawlReport: {exc}")

    crawler_revision_path = output_dir / "meta" / "crawler-revision.json"
    if crawler_revision_path.exists():
        try:
            with open(crawler_revision_path, encoding="utf-8") as fh:
                crawler_revision = json.load(fh).get("crawler_revision")
        except Exception as exc:
            errors.append(f"meta/crawler-revision.json: cannot read: {exc}")
            crawler_revision = None
    else:
        crawler_revision = None

    submodule_revision = _crawler_revision(output_dir / "crawler")
    if crawler_revision and submodule_revision and crawler_revision != submodule_revision:
        errors.append(
            f"crawler submodule revision {submodule_revision[:12]} does not match "
            f"generated revision {crawler_revision[:12]}"
        )

    # README metrics must be consistent with the generated artifacts.
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        errors.append("README.md is missing")
    else:
        try:
            readme_text = readme_path.read_text(encoding="utf-8")
            _validate_readme_metrics(readme_text, output_dir, errors)
        except Exception as exc:
            errors.append(f"README.md: cannot validate: {exc}")

    return errors


def _validate_market_source_integrity(output_dir: Path) -> list[str]:
    """Strict cross-market and source-integrity checks.

    Rejects published data where a market's pages were served for a different
    country, where the primary card fee uses the wrong currency, or where the
    same source entry id appears under unrelated markets.
    """
    errors: list[str] = []
    json_dir = output_dir / "json"
    meta_dir = output_dir / "meta"

    # Load manifest and index for cross-reference.
    manifest: MarketManifest | None = None
    if (meta_dir / "markets.json").exists():
        with open(meta_dir / "markets.json", encoding="utf-8") as fh:
            manifest = MarketManifest.model_validate(json.load(fh))
    manifest_countries = {m.account_country.upper() for m in manifest.markets} if manifest else set()

    index: MarketIndex | None = None
    if (json_dir / "index.json").exists():
        with open(json_dir / "index.json", encoding="utf-8") as fh:
            index = MarketIndex.model_validate(json.load(fh))
    index_by_country = {e.account_country.upper(): e for e in index.markets} if index else {}

    core_fees_path = json_dir / "core-fees.json"
    core_fees_by_country: dict[str, list[CoreFeeRule]] = {}
    if core_fees_path.exists():
        with open(core_fees_path, encoding="utf-8") as fh:
            core_fees = CoreFees.model_validate(json.load(fh))
        for entry in core_fees.markets:
            core_fees_by_country[entry.account_country.upper()] = entry.rules

    for path in sorted(json_dir.glob("*.json")):
        if path.name in {"index.json", "core-fees.json", "payment-methods.json"}:
            continue
        country_code = path.stem.upper()
        if country_code not in manifest_countries:
            errors.append(f"{path.name}: no matching market in meta/markets.json")
            continue

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            market_output = MarketOutput.model_validate(data)
        except Exception as exc:
            errors.append(f"{path.name}: cannot validate market output: {exc}")
            continue

        market = market_output.market
        expected_country = market.account_country.upper()
        expected_currency = market.default_currency

        # 1. Source metadata must match the market.
        for source in market_output.sources:
            if source.detected_market and source.detected_market.upper() != expected_country:
                errors.append(
                    f"{path.name}: source {source.requested_url!r} served market "
                    f"{source.detected_market} but file is for {expected_country}"
                )

            for url in (source.effective_url, source.canonical_url, source.requested_url):
                if not url:
                    continue
                detected = _detect_market_from_path(url)
                if detected and detected.upper() != expected_country:
                    errors.append(
                        f"{path.name}: source URL {url!r} contains explicit market {detected} "
                        f"but file is for {expected_country}"
                    )
                    break

        # 2. Index source_urls must match the market's canonical/requested URLs.
        actual_source_urls: set[str] = set()
        for source in market_output.sources:
            url = source.canonical_url or source.requested_url or source.effective_url
            if url:
                actual_source_urls.add(url)
        if expected_country in index_by_country:
            index_urls = set(index_by_country[expected_country].source_urls)
            if index_urls != actual_source_urls and actual_source_urls:
                missing = index_urls - actual_source_urls
                extra = actual_source_urls - index_urls
                details = []
                if missing:
                    details.append(f"missing from country file: {sorted(missing)}")
                if extra:
                    details.append(f"missing from index: {sorted(extra)}")
                errors.append(f"{path.name}: index source_urls mismatch ({'; '.join(details)})")

        # 3. Primary card fees must use the market's default currency.
        if expected_currency:
            primary_rules = [
                r
                for r in market_output.derived_rules
                if _is_calculable_status(r.classification_status)
                and (r.product_id == "payments" or r.payment_method == "card")
                and r.fixed_currency is not None
            ]
            for rule in primary_rules:
                if rule.fixed_currency and rule.fixed_currency.upper() != expected_currency.upper():
                    errors.append(
                        f"{path.name}: primary card fee {rule.rule_id} uses currency "
                        f"{rule.fixed_currency}, expected {expected_currency} for {expected_country}"
                    )

        # 4. Core-fees rules for this market should also respect the currency.
        for rule in core_fees_by_country.get(expected_country, []):
            if rule.product_id == "payments" and expected_currency:
                for comp in rule.fee_components:
                    if (
                        comp.type == "fixed_amount"
                        and comp.currency
                        and comp.currency.upper() != expected_currency.upper()
                    ):
                        errors.append(
                            f"core-fees/{expected_country}: primary card fee {rule.rule_id} "
                            f"uses currency {comp.currency}, expected {expected_currency}"
                        )

    return errors


def validate_all_output(output_dir: str | Path, strict: bool = True, semantic: bool = True) -> dict[str, Any]:
    """Validate all generated JSON files in an output directory.

    Returns a summary dict. Raises CrawlerValidationError on failure in strict mode.
    When ``semantic`` is True (the default), additional semantic checks run after
    Pydantic schema validation.
    """
    output_dir = Path(output_dir)
    errors: list[str] = []
    validated: list[str] = []

    json_dir = output_dir / "json"
    meta_dir = output_dir / "meta"

    if json_dir.exists():
        for path in sorted(json_dir.glob("*.json")):
            if path.name == "index.json":
                try:
                    with open(path, encoding="utf-8") as fh:
                        validate_index(json.load(fh))
                    validated.append(str(path))
                except Exception as exc:
                    errors.append(f"index.json: {exc}")
            elif path.name == "core-fees.json":
                try:
                    with open(path, encoding="utf-8") as fh:
                        validate_core_fees(json.load(fh))
                    validated.append(str(path))
                except Exception as exc:
                    errors.append(f"core-fees.json: {exc}")
            elif path.name == "payment-methods.json":
                try:
                    with open(path, encoding="utf-8") as fh:
                        validate_payment_methods(json.load(fh))
                    validated.append(str(path))
                except Exception as exc:
                    errors.append(f"payment-methods.json: {exc}")
            else:
                try:
                    with open(path, encoding="utf-8") as fh:
                        validate_market_output(json.load(fh))
                    validated.append(str(path))
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")

    if meta_dir.exists():
        for path in sorted(meta_dir.glob("*.json")):
            if path.name == "markets.json":
                try:
                    with open(path, encoding="utf-8") as fh:
                        validate_manifest(json.load(fh))
                    validated.append(str(path))
                except Exception as exc:
                    errors.append(f"markets.json: {exc}")
            else:
                validated.append(str(path))

    schema_version_path = meta_dir / "schema-version.json"
    if schema_version_path.exists():
        try:
            with open(schema_version_path, encoding="utf-8") as fh:
                SchemaVersionInfo.model_validate(json.load(fh))
            validated.append(str(schema_version_path))
        except Exception as exc:
            errors.append(f"schema-version.json: {exc}")

    if semantic:
        try:
            validate_semantic(output_dir)
        except CrawlerValidationError as exc:
            errors.append(str(exc))

    if strict:
        errors.extend(_validate_publication(output_dir))
        errors.extend(_validate_market_source_integrity(output_dir))

    if strict and errors:
        raise CrawlerValidationError("Output validation failed:\n" + "\n".join(errors))

    return {"validated": validated, "errors": errors, "success": not errors}


def _country_code_from_item(item: dict[str, Any]) -> str | None:
    code = item.get("account_country") or item.get("stripe_market_code")
    if code:
        return code.upper()
    return None


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.load(path.open(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("markets", []) or data.get("items", [])
    return []


def _validate_readme_metrics(readme_text: str, output_dir: Path, errors: list[str]) -> None:
    def _readme_int(pattern: str) -> int | None:
        m = re.search(pattern, readme_text)
        if not m:
            return None
        return int(m.group(1).replace(",", ""))

    core_fees_path = output_dir / "json" / "core-fees.json"
    core_rule_count = 0
    if core_fees_path.exists():
        try:
            core_fees = validate_core_fees(json.load(core_fees_path.open(encoding="utf-8")))
            core_rule_count = sum(len(m.rules) for m in core_fees.markets)
        except Exception as exc:
            errors.append(f"core-fees.json: cannot count rules for README check: {exc}")

    readme_rules = _readme_int(r"\|\s*Core fee rules\s*\|\s*\*\*(\d[\d,]*)\*\*\s*\|")
    if readme_rules is not None and readme_rules != core_rule_count:
        errors.append(f"README core fee rules ({readme_rules}) does not match core-fees.json ({core_rule_count})")

    manifest_path = output_dir / "meta" / "markets.json"
    unsupported_path = output_dir / "meta" / "unsupported-markets.json"
    transient_path = output_dir / "meta" / "transient-failures.json"

    all_markets: set[str] = set()
    for item in _load_json_list(manifest_path):
        code = _country_code_from_item(item)
        if code:
            all_markets.add(code)
    for item in _load_json_list(unsupported_path):
        code = _country_code_from_item(item)
        if code:
            all_markets.add(code)
    for item in _load_json_list(transient_path):
        code = _country_code_from_item(item)
        if code:
            all_markets.add(code)

    unsupported_count = len(_load_json_list(unsupported_path))
    transient_count = len(_load_json_list(transient_path))

    readme_markets = _readme_int(r"\|\s*Markets\s*\|\s*\*\*(\d[\d,]*)\*\*\s*\|")
    if readme_markets is not None and readme_markets != len(all_markets):
        errors.append(
            f"README market count ({readme_markets}) does not match discovered market count ({len(all_markets)})"
        )

    readme_unsupported = _readme_int(r"\|\s*Unsupported markets\s*\|\s*(\d[\d,]*)\s*\|")
    if readme_unsupported is not None and readme_unsupported != unsupported_count:
        errors.append(
            f"README unsupported count ({readme_unsupported}) does not match unsupported-markets.json ({unsupported_count})"
        )

    readme_transient = _readme_int(r"\|\s*Transient failures\s*\|\s*(\d[\d,]*)\s*\|")
    if readme_transient is not None and readme_transient != transient_count:
        errors.append(
            f"README transient count ({readme_transient}) does not match transient-failures.json ({transient_count})"
        )


def _safe_load[T](
    path: Path,
    loader: Callable[[Any], T],
    errors: list[str],
    label: str,
) -> T | None:
    """Load JSON from ``path`` and validate it with ``loader``.

    Errors are appended to ``errors`` and ``None`` is returned on failure.
    """
    data = _read_json(path)
    if data is None:
        if path.exists():
            errors.append(f"{label}: cannot read")
        return None
    try:
        return loader(data)
    except Exception as exc:
        errors.append(f"{label}: cannot validate: {exc}")
        return None


def _validate_completeness(output_dir: Path) -> list[str]:
    """Verify that every discovered market is in exactly one supported/unsupported/transient state."""
    errors: list[str] = []

    manifest_path = output_dir / "meta" / "markets.json"
    index_path = output_dir / "json" / "index.json"
    core_fees_path = output_dir / "json" / "core-fees.json"
    unsupported_path = output_dir / "meta" / "unsupported-markets.json"
    transient_path = output_dir / "meta" / "transient-failures.json"

    manifest = _safe_load(manifest_path, validate_manifest, errors, "markets.json")
    index = _safe_load(index_path, validate_index, errors, "index.json")
    core_fees = _safe_load(core_fees_path, validate_core_fees, errors, "core-fees.json")
    unsupported = (
        _safe_load(
            unsupported_path,
            lambda data: [UnsupportedMarket.model_validate(item) for item in data],
            errors,
            "unsupported-markets.json",
        )
        or []
    )
    transient = (
        _safe_load(
            transient_path,
            lambda data: [UnsupportedMarket.model_validate(item) for item in data],
            errors,
            "transient-failures.json",
        )
        or []
    )

    discovered: dict[str, str] = {}
    if manifest:
        for market in manifest.markets:
            discovered[market.account_country] = "manifest"
        for item in manifest.unsupported:
            if item.account_country:
                discovered.setdefault(item.account_country, "manifest_unsupported")
        for item in manifest.transient_failures:
            if item.account_country:
                discovered.setdefault(item.account_country, "manifest_transient")
    for item in unsupported:
        if item.account_country:
            discovered.setdefault(item.account_country, "unsupported_file")
    for item in transient:
        if item.account_country:
            discovered.setdefault(item.account_country, "transient_file")
    if index:
        for entry in index.markets:
            discovered.setdefault(entry.account_country, "index")

    supported_by_index: dict[str, Any] = {}
    if index:
        for entry in index.markets:
            supported_by_index[entry.account_country] = entry

    unsupported_set = {u.account_country for u in unsupported if u.account_country}
    transient_set = {t.account_country for t in transient if t.account_country}
    core_by_country: dict[str, Any] = {}
    if core_fees:
        for entry in core_fees.markets:
            core_by_country[entry.account_country] = entry

    for country in sorted(discovered):
        states: list[str] = []
        in_index = country in supported_by_index
        in_core = country in core_by_country
        in_unsupported = country in unsupported_set
        in_transient = country in transient_set
        if in_index:
            states.append("supported")
        if in_unsupported:
            states.append("unsupported")
        if in_transient:
            states.append("transient")
        if len(states) != 1:
            errors.append(f"{country}: expected exactly one state, got {states}")
            continue

        state = states[0]
        if state == "supported":
            entry = supported_by_index[country]
            if entry.derivation_status != "complete" or entry.calculator_coverage_status != "complete":
                errors.append(
                    f"{country}: supported market is not complete "
                    f"({entry.derivation_status}/{entry.calculator_coverage_status})"
                )
            if not in_core:
                errors.append(f"{country}: supported market is missing from core-fees.json")
            else:
                core = core_by_country[country]
                if core.derivation_status != entry.derivation_status:
                    errors.append(
                        f"{country}: derivation_status mismatch between index ({entry.derivation_status}) "
                        f"and core-fees ({core.derivation_status})"
                    )
                if core.calculator_coverage_status != entry.calculator_coverage_status:
                    errors.append(
                        f"{country}: calculator_coverage_status mismatch between index ({entry.calculator_coverage_status}) "
                        f"and core-fees ({core.calculator_coverage_status})"
                    )
        elif state == "unsupported":
            record = next((u for u in unsupported if u.account_country == country), None)
            if record and not record.requested_urls and not isinstance(record.requested_urls, list):
                errors.append(f"{country}: unsupported record requested_urls must be a list")
            if record and not record.reason:
                errors.append(f"{country}: unsupported record must have a reason")
        elif state == "transient":
            record = next((t for t in transient if t.account_country == country), None)
            if record and not record.reason:
                errors.append(f"{country}: transient record must have a reason")

    for country in sorted(core_by_country):
        if country not in supported_by_index:
            errors.append(f"{country}: core-fees.json market {country} is not in index.json")
        if country in unsupported_set or country in transient_set:
            errors.append(f"{country}: core-fees.json market is also listed as unsupported/transient")

    for country in sorted(supported_by_index):
        if country not in discovered:
            errors.append(f"{country}: index.json market {country} is not in the discovered market set")

    return errors


def validate_data_repository(
    data_repo_dir: str | Path,
    strict: bool = True,
    require_all_complete: bool = False,
) -> dict[str, Any]:
    """Validate the contents of the stripe-fee-data repository."""
    result = validate_all_output(Path(data_repo_dir), strict=strict, semantic=True)
    if require_all_complete and result["success"]:
        completeness_errors = _validate_completeness(Path(data_repo_dir))
        result["errors"].extend(completeness_errors)
        result["success"] = not result["errors"]
        if strict and result["errors"]:
            raise CrawlerValidationError("Repository completeness check failed:\n" + "\n".join(result["errors"]))
    return result
