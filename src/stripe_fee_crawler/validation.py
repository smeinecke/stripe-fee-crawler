"""Output validation and JSON schema loading."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    CoreFees,
    FeeRule,
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

    if strict and errors:
        raise CrawlerValidationError("Output validation failed:\n" + "\n".join(errors))

    return {"validated": validated, "errors": errors, "success": not errors}


def _validate_rule_currency_exponents(rule: FeeRule, market_code: str, errors: list[str]) -> None:
    if rule.classification_status != "classified" or not rule.fixed_amount or not rule.fixed_currency:
        return
    expected_exponent = currency_exponent(rule.fixed_currency)
    multiplier = 10**expected_exponent
    try:
        expected_minor = int(Decimal(rule.fixed_amount) * multiplier)
    except Exception:
        errors.append(
            f"{market_code}/{rule.rule_id}: cannot compute minor units for {rule.fixed_amount} {rule.fixed_currency}"
        )
        return
    if rule.fixed_amount_minor is None or int(rule.fixed_amount_minor) != expected_minor:
        errors.append(
            f"{market_code}/{rule.rule_id}: fixed_amount_minor {rule.fixed_amount_minor} does not match "
            f"{rule.fixed_currency} exponent {expected_exponent} (expected {expected_minor})"
        )


def _validate_rule_calculator_ready(rule: FeeRule, market_code: str, errors: list[str]) -> None:
    if rule.classification_status != "classified":
        return
    if not rule.channel:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing channel")
    if not rule.unit:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing unit")
    if not rule.behavior:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule missing behavior")
    if not rule.percentage and not rule.fixed_amount:
        errors.append(f"{market_code}/{rule.rule_id}: classified rule has no percentage or fixed amount")


def _validate_rule_percentage_consistency(rule: FeeRule, market_code: str, errors: list[str]) -> None:
    if not rule.percentage or not rule.basis_points:
        return
    try:
        if Decimal(rule.basis_points) != Decimal(rule.percentage) * Decimal("100"):
            errors.append(
                f"{market_code}/{rule.rule_id}: basis_points {rule.basis_points} != "
                f"percentage {rule.percentage} * 100"
            )
    except Exception as exc:
        errors.append(f"{market_code}/{rule.rule_id}: percentage/basis_points comparison failed: {exc}")


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


def validate_data_repository(data_repo_dir: str | Path) -> dict[str, Any]:
    """Validate the contents of the stripe-fee-data repository."""
    return validate_all_output(Path(data_repo_dir), strict=True)
