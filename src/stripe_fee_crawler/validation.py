"""Output validation and JSON schema loading."""

from __future__ import annotations

import json
import logging
import subprocess  # nosec B404
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    CoreFeeRule,
    CoreFees,
    FeeComponent,
    FeeCondition,
    MarketIndex,
    MarketManifest,
    MarketOutput,
    PaymentMethodCatalog,
    SchemaVersionInfo,
)
from .pricing_tokens import currency_exponent

logger = logging.getLogger(__name__)


def validate_market_output(data: dict[str, Any]) -> MarketOutput:
    """Validate a per-market output dictionary."""
    try:
        return MarketOutput.model_validate(data)
    except ValidationError as exc:
        raise CrawlerValidationError(f"Market output validation failed: {exc}") from exc


def validate_index(data: dict[str, Any]) -> MarketIndex:
    """Validate the market index dictionary."""
    try:
        return MarketIndex.model_validate(data)
    except ValidationError as exc:
        raise CrawlerValidationError(f"Index validation failed: {exc}") from exc


def validate_core_fees(data: dict[str, Any]) -> CoreFees:
    """Validate the consolidated core-fees dictionary."""
    try:
        return CoreFees.model_validate(data)
    except ValidationError as exc:
        raise CrawlerValidationError(f"Core fees validation failed: {exc}") from exc


def validate_payment_methods(data: dict[str, Any]) -> PaymentMethodCatalog:
    """Validate the payment-methods catalog dictionary."""
    try:
        return PaymentMethodCatalog.model_validate(data)
    except ValidationError as exc:
        raise CrawlerValidationError(f"Payment methods validation failed: {exc}") from exc


def validate_manifest(data: dict[str, Any]) -> MarketManifest:
    """Validate the market manifest dictionary."""
    try:
        return MarketManifest.model_validate(data)
    except ValidationError as exc:
        raise CrawlerValidationError(f"Manifest validation failed: {exc}") from exc


def generate_market_output_schema() -> dict[str, Any]:
    """Generate the JSON schema for per-market output."""
    schema = MarketOutput.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/stripe-fee-data/schemas/stripe-fees-v1.schema.json"
    return schema


def generate_core_fees_schema() -> dict[str, Any]:
    """Generate the JSON schema for the consolidated core fees file."""
    schema = CoreFees.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/stripe-fee-data/schemas/core-fees-v1.schema.json"
    return schema


def generate_payment_methods_schema() -> dict[str, Any]:
    """Generate the JSON schema for the payment-methods catalog."""
    schema = PaymentMethodCatalog.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/stripe-fee-data/schemas/payment-methods-v1.schema.json"
    return schema


def generate_index_schema() -> dict[str, Any]:
    """Generate the JSON schema for the market index file."""
    schema = MarketIndex.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/stripe-fee-data/schemas/index-v1.schema.json"
    return schema


def generate_manifest_schema() -> dict[str, Any]:
    """Generate the JSON schema for the market manifest file."""
    schema = MarketManifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/stripe-fee-data/schemas/manifest-v1.schema.json"
    return schema


def _crawler_submodule_revision(data_dir: Path) -> str | None:
    """Return the HEAD revision of the ``crawler`` submodule, if present."""
    crawler_dir = data_dir / "crawler"
    if not crawler_dir.exists():
        return None
    try:
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "HEAD"],
            cwd=crawler_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


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
        rules = data.get("derived_rules", [])
        if not rules:
            continue
        market_has_derived_rules = True
        if not any(_is_calculable_status(r.get("classification_status", "")) for r in rules):
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
    if change_report_path.exists():
        try:
            with open(change_report_path, encoding="utf-8") as fh:
                change_report = json.load(fh)
            if change_report.get("has_regression"):
                errors.append("change-report.json has_regression is true")
        except Exception as exc:
            errors.append(f"change-report.json: cannot read: {exc}")

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

    submodule_revision = _crawler_submodule_revision(output_dir)
    if crawler_revision and submodule_revision and crawler_revision != submodule_revision:
        errors.append(
            f"crawler submodule revision {submodule_revision[:12]} does not match "
            f"generated revision {crawler_revision[:12]}"
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

    if strict and errors:
        raise CrawlerValidationError("Output validation failed:\n" + "\n".join(errors))

    return {"validated": validated, "errors": errors, "success": not errors}


def _is_calculable_status(status: str) -> bool:
    return status in {"classified", "calculable_rule"}


def _validate_component_currency_exponents(
    rule: CoreFeeRule,
    comp: FeeComponent,
    market_code: str,
    errors: list[str],
) -> None:
    if comp.type not in {"fixed_amount", "maximum_fee", "minimum_fee", "fixed_surcharge"}:
        return
    if not comp.amount or not comp.currency:
        return
    expected_exponent = currency_exponent(comp.currency)
    try:
        expected_minor = int(Decimal(comp.amount) * (10**expected_exponent))
    except Exception:
        errors.append(f"{market_code}/{rule.rule_id}: cannot compute minor units for {comp.amount} {comp.currency}")
        return
    if comp.minor_amount is None or int(comp.minor_amount) != expected_minor:
        errors.append(
            f"{market_code}/{rule.rule_id}: minor_amount for {comp.amount} {comp.currency} does not match "
            f"exponent {expected_exponent} (expected {expected_minor})"
        )


def _validate_rule_currency_exponents(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    if not _is_calculable_status(rule.classification_status):
        return
    for comp in rule.fee_components:
        _validate_component_currency_exponents(rule, comp, market_code, errors)


def _validate_rule_calculator_ready(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    if not _is_calculable_status(rule.classification_status):
        return
    if not rule.channel:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing channel")
    if not rule.unit:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing unit")
    if not rule.behavior:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing behavior")
    if not rule.fee_components:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule has no fee components")
        return
    base_types = {"percentage", "fixed_amount"}
    modifier_types = {"percentage_surcharge", "fixed_surcharge"}
    calculable_types = base_types | modifier_types | {"maximum_fee", "minimum_fee"}
    if not any(comp.type in calculable_types for comp in rule.fee_components):
        errors.append(f"{market_code}/{rule.rule_id}: classified rule has no calculable fee component")
        return
    has_base = any(comp.type in base_types for comp in rule.fee_components)
    has_modifier = any(comp.type in modifier_types for comp in rule.fee_components)
    if has_modifier and not has_base:
        # A surcharge-only rule is only valid when it applies a modifier condition.
        modifier_dimensions = {"currency_conversion_required", "card_origin", "dispute_state", "transaction_type"}
        if not any(c.dimension in modifier_dimensions for c in rule.conditions):
            errors.append(
                f"{market_code}/{rule.rule_id}: base fee appears to be classified only as a surcharge/modifier"
            )


def _validate_rule_percentage_consistency(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    for comp in rule.fee_components:
        if comp.type not in {"percentage", "percentage_surcharge"}:
            continue
        if not comp.value or not comp.basis_points:
            continue
        try:
            if Decimal(comp.basis_points) != Decimal(comp.value) * Decimal("100"):
                errors.append(
                    f"{market_code}/{rule.rule_id}: basis_points {comp.basis_points} != value {comp.value} * 100"
                )
        except Exception as exc:
            errors.append(f"{market_code}/{rule.rule_id}: percentage/basis_points comparison failed: {exc}")


def _condition_key(conditions: list[FeeCondition]) -> tuple[tuple[str, str, Any], ...]:
    return tuple(sorted((c.dimension, c.operator, str(c.value)) for c in conditions))


def _validate_core_fees_semantic(
    core_fees: CoreFees,
    manifest: MarketManifest,
    payment_methods: PaymentMethodCatalog,
) -> list[str]:
    errors: list[str] = []
    market_codes = {m.stripe_market_code for m in manifest.markets}
    for entry in core_fees.markets:
        if entry.stripe_market_code not in market_codes:
            errors.append(
                f"core-fees/{entry.account_country}: stripe_market_code {entry.stripe_market_code} "
                "not present in manifest.markets"
            )
        seen_identities: dict[tuple[str, str | None, Any], list[str]] = {}
        for rule in entry.rules:
            _validate_rule_currency_exponents(rule, entry.stripe_market_code, errors)
            _validate_rule_calculator_ready(rule, entry.stripe_market_code, errors)
            _validate_rule_percentage_consistency(rule, entry.stripe_market_code, errors)
            known_methods = {m.method_id for m in payment_methods.methods}
            if rule.payment_method and rule.payment_method not in known_methods:
                errors.append(
                    f"{entry.stripe_market_code}/{rule.rule_id}: payment_method {rule.payment_method} "
                    "not found in payment-methods catalog"
                )
            identity = (rule.product_id or "", rule.variant_id, _condition_key(rule.conditions))
            seen_identities.setdefault(identity, []).append(rule.rule_id)
        for identity, rule_ids in seen_identities.items():
            if len(rule_ids) > 1:
                errors.append(f"{entry.stripe_market_code}: semantic identity conflict for {identity}: {rule_ids}")
    return errors


def validate_semantic(
    output_dir: str | Path,
    core_fees: CoreFees | None = None,
    manifest: MarketManifest | None = None,
    payment_methods: PaymentMethodCatalog | None = None,
) -> dict[str, Any]:
    """Run semantic checks beyond Pydantic schema validation.

    Returns a summary dict. Raises CrawlerValidationError on failure.
    """
    output_dir = Path(output_dir)
    errors: list[str] = []

    if core_fees is None:
        core_fees_path = output_dir / "json" / "core-fees.json"
        if core_fees_path.exists():
            with open(core_fees_path, encoding="utf-8") as fh:
                core_fees = validate_core_fees(json.load(fh))
    if manifest is None:
        manifest_path = output_dir / "meta" / "markets.json"
        if manifest_path.exists():
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = validate_manifest(json.load(fh))
    if payment_methods is None:
        pm_path = output_dir / "json" / "payment-methods.json"
        if pm_path.exists():
            with open(pm_path, encoding="utf-8") as fh:
                payment_methods = validate_payment_methods(json.load(fh))

    if core_fees and manifest and payment_methods:
        errors.extend(_validate_core_fees_semantic(core_fees, manifest, payment_methods))
    elif core_fees is None:
        errors.append("core-fees.json not found; cannot run semantic checks")
    elif manifest is None:
        errors.append("markets.json not found; cannot run semantic checks")
    elif payment_methods is None:
        errors.append("payment-methods.json not found; cannot run semantic checks")

    if errors:
        raise CrawlerValidationError("Semantic validation failed:\n" + "\n".join(errors))

    return {"success": True, "errors": errors}


def validate_data_repository(
    data_repo_dir: str | Path,
    strict: bool = True,
    require_all_complete: bool = False,
) -> dict[str, Any]:
    """Validate the contents of the stripe-fee-data repository."""
    result = validate_all_output(Path(data_repo_dir), strict=strict, semantic=True)
    if require_all_complete and result["success"]:
        core_fees_path = Path(data_repo_dir) / "json" / "core-fees.json"
        if core_fees_path.exists():
            with open(core_fees_path, encoding="utf-8") as fh:
                core_fees = CoreFees.model_validate(json.load(fh))
            for entry in core_fees.markets:
                if entry.derivation_status not in {"complete"}:
                    result["errors"].append(
                        f"{entry.account_country}: derivation_status is {entry.derivation_status!r}"
                    )
                if entry.calculator_coverage_status not in {"complete"}:
                    result["errors"].append(
                        f"{entry.account_country}: calculator_coverage_status is {entry.calculator_coverage_status!r}"
                    )
        else:
            result["errors"].append("core-fees.json not found; cannot verify completeness")
        result["success"] = not result["errors"]
        if strict and result["errors"]:
            raise CrawlerValidationError("Repository completeness check failed:\n" + "\n".join(result["errors"]))
    return result
