"""Semantic validation rules for classified fee data."""

from __future__ import annotations

import json
import logging
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

from .._common import _condition_key, _contains_market_share_evidence
from ..currencies import CURRENCY_SYMBOLS, MINOR_CURRENCY_SYMBOLS, currency_exponent
from ..exceptions import ValidationError as CrawlerValidationError
from ..models import (
    CoreFeeRule,
    CoreFees,
    FeeComponent,
    MarketManifest,
    PaymentMethodCatalog,
)
from ..pricing_tokens import _parse_decimal
from .schemas import (
    validate_core_fees,
    validate_manifest,
    validate_payment_methods,
)

logger = logging.getLogger(__name__)


def _is_calculable_status(status: str) -> bool:
    return status in {"classified", "calculable_rule"}


_PUBLICATION_CONFIDENCE_THRESHOLD = 0.7


def _currency_label_markers(currency: str) -> set[str]:
    """Return the currency code plus any known symbols for that currency."""
    markers = {currency.upper(), currency.lower()}
    for symbol, code in CURRENCY_SYMBOLS.items():
        if code.upper() == currency.upper():
            markers.add(symbol)
            markers.add(symbol.lower())
    return markers


def _minor_symbol_for_currency(currency: str) -> str | None:
    """Return the minor-currency symbol (e.g. 'p' for GBP, '¢' for USD)."""
    for symbol in MINOR_CURRENCY_SYMBOLS:
        if CURRENCY_SYMBOLS.get(symbol, "").upper() == currency.upper():
            return symbol
    return None


def _label_references_component(
    label: str,
    amount: str,
    currency: str | None,
    minor_amount: str | None = None,
) -> bool:
    """Check whether the source label contains the amount and currency."""
    if not amount:
        return True
    lower = label.lower()
    # Direct amount string match (e.g. "0.25" or "0,25").
    if amount in lower:
        return True
    # Try parsing numeric tokens in the label and compare value, reusing the
    # same localized-decimal parser used during extraction.
    amount_dec = _parse_decimal(amount)
    if amount_dec is not None:
        for match in re.finditer(r"[0-9][0-9\s,.]*", lower):
            candidate_dec = _parse_decimal(match.group())
            if candidate_dec is not None and candidate_dec == amount_dec:
                return True
    # For minor-currency amounts the label may contain the raw minor text ("20p").
    minor_symbol = _minor_symbol_for_currency(currency) if currency else None
    if minor_symbol and minor_amount:
        pattern = rf"{re.escape(minor_amount)}\s*{re.escape(minor_symbol)}"
        if re.search(pattern, lower):
            return True
    return False


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

    for comp in rule.fee_components:
        _validate_component_currency_exponents(rule, comp, market_code, errors)


def _validate_rule_calculator_ready(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:

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
        # A surcharge-only rule is only valid when it applies a modifier condition
        # (currency conversion, card/international origin, dispute/refund state, or
        # cross-border transaction region).
        modifier_dimensions = {
            "currency_conversion_required",
            "card_origin",
            "dispute_state",
            "transaction_type",
            "cross_border",
            "transaction_region",
        }
        if not any(c.dimension in modifier_dimensions for c in rule.conditions):
            errors.append(
                f"{market_code}/{rule.rule_id}: base fee appears to be classified only as a surcharge/modifier"
            )

    # Positive-fee evidence is required for publication.
    evidence = rule.fee_evidence
    positive_evidence_types = {"explicit_fee_phrase", "pricing_table_value", "structured_fee_field"}
    negative_evidence_types = {
        "marketing_prose",
        "promotional_language",
        "hardware_price",
        "alphanumeric_method_name",
        "insufficient",
        "contradictory_fee_evidence",
    }
    if evidence is None:
        errors.append(f"{market_code}/{rule.rule_id}: calculable rule has no fee_evidence")
    elif evidence.type in negative_evidence_types:
        errors.append(f"{market_code}/{rule.rule_id}: calculable rule has negative fee_evidence {evidence.type}")
    elif evidence.type not in positive_evidence_types:
        errors.append(
            f"{market_code}/{rule.rule_id}: calculable rule has unsupported fee_evidence type {evidence.type}"
        )

    if rule.confidence < _PUBLICATION_CONFIDENCE_THRESHOLD:
        errors.append(f"{market_code}/{rule.rule_id}: confidence {rule.confidence} below publication threshold")

    # Suspicious-rule audit.
    label = (rule.label or "").lower()
    has_percentage = any(comp.type in {"percentage", "percentage_surcharge"} for comp in rule.fee_components)
    fixed_components = [c for c in rule.fee_components if c.type == "fixed_amount"]
    largest_fixed = max(
        (Decimal(c.amount) for c in fixed_components if c.amount),
        default=Decimal("0"),
    )
    largest_fixed_minor = max(
        (Decimal(c.minor_amount) for c in fixed_components if c.minor_amount),
        default=Decimal("0"),
    )

    if rule.unit == "yearly" and (evidence is None or evidence.type not in positive_evidence_types):
        errors.append(f"{market_code}/{rule.rule_id}: yearly rule lacks explicit fee evidence")

    # Flag fixed-only per-transaction amounts that are implausibly large in
    # minor units (roughly USD 10+ equivalent).
    if rule.unit == "per_transaction" and largest_fixed_minor >= Decimal("1000") and not has_percentage:
        errors.append(
            f"{market_code}/{rule.rule_id}: large one-time fixed amount {largest_fixed} labeled per_transaction"
        )

    if evidence and evidence.type == "alphanumeric_method_name":
        errors.append(f"{market_code}/{rule.rule_id}: amount appears to be extracted from a product/method name")

    # The source label should contain or reference the fee components.
    if fixed_components:
        for comp in fixed_components:
            if not comp.amount:
                continue
            if not _label_references_component(label, comp.amount, comp.currency, comp.minor_amount):
                errors.append(
                    f"{market_code}/{rule.rule_id}: source label does not reference fixed amount "
                    f"{comp.amount} {comp.currency}"
                )
                continue
            if comp.currency:
                markers = _currency_label_markers(comp.currency)
                if not any(m in label for m in markers):
                    errors.append(
                        f"{market_code}/{rule.rule_id}: source label does not reference currency {comp.currency}"
                    )


def _validate_rule_contradictory_evidence(
    rule: CoreFeeRule,
    market_code: str,
    errors: list[str],
) -> None:
    """Reject calculable rules that mix positive-fee and included/free evidence."""

    if rule.fee_evidence and rule.fee_evidence.type == "contradictory_fee_evidence":
        errors.append(f"{market_code}/{rule.rule_id}: calculable rule has contradictory fee evidence")
        return
    label = (rule.label or "").lower()
    evidence_phrases = [p.lower() for p in (rule.fee_evidence.phrases if rule.fee_evidence else [])]
    combined = f"{label} {' '.join(evidence_phrases)}"
    has_positive_fee = bool(re.search(r"[0-9]\s*%|[0-9]\s*[€£$¥A-Z]", combined))
    # "interest-free" describes the buyer's financing, not the merchant fee,
    # and must not be treated as a contradictory "free" statement.
    included_free_check = re.sub(r"interest[-\s]?free", "", combined, flags=re.IGNORECASE)
    has_included_free = bool(
        re.search(
            r"(?<!\bnot\s)\bincluded\b|(?<!\bnot\s)\bfree\b|\bno additional charge\b|\bat no cost\b|\bno cost\b|\bno fee\b",
            included_free_check,
        )
    )
    if has_positive_fee and has_included_free:
        errors.append(f"{market_code}/{rule.rule_id}: calculable rule combines positive fee and included/free evidence")


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


_ADDON_PRODUCTS: set[str] = {
    "authorization_boost",
    "radar",
    "smart_disputes",
    "three_d_secure",
}

_PAYMENTS_VARIANTS: set[str] = {
    "online_domestic_cards",
    "online_international_cards",
    "online_premium_cards",
    "in_person_domestic_cards",
    "in_person_international_cards",
    "in_person_premium_cards",
    "domestic_cards",
    "international_cards",
    "tap_to_pay",
}


def _rule_source_combined(rule: CoreFeeRule) -> str:
    """Return a lower-cased combination of the rule label and evidence phrases."""
    label = (rule.label or "").lower()
    phrases = [p.lower() for p in (rule.fee_evidence.phrases if rule.fee_evidence else [])]
    return f"{label} {' '.join(phrases)}"


def _is_positive_component_source(text: str | None) -> bool:
    """Return True when a component's source text is a trusted fee-value node."""
    if not text:
        return False
    lower = text.lower()
    has_fee_language = bool(
        re.search(r"\b(fee|charge|transaction|per\s+(charge|transaction|successful charge))\b", lower)
        or "+ " in text
        or "%" in text
    )
    return has_fee_language and not _contains_market_share_evidence(text)


def _validate_rule_market_share_evidence(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Reject calculable rules whose evidence includes market-share statistics."""

    combined = _rule_source_combined(rule)
    if not _contains_market_share_evidence(combined):
        return
    # A calculable rule is only allowed to contain market-share text if the
    # actual fee value comes from a separate, trusted pricing-value node.
    component_sources = {
        c.source_text
        for c in rule.fee_components
        if c.type in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge"}
    }
    if not any(_is_positive_component_source(src) for src in component_sources):
        errors.append(
            f"{market_code}/{rule.rule_id}: calculable rule contains market-share evidence without a trusted fee-value source"
        )


def _validate_rule_cross_fragment_evidence(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Reject rules whose numeric fee value and fee wording come from different source fragments."""
    if rule.fee_evidence and rule.fee_evidence.type == "cross_fragment_fee_evidence":
        errors.append(f"{market_code}/{rule.rule_id}: fee value and fee wording come from different source fragments")


def _validate_rule_pricing_plan(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Every rule with a pricing-plan phrase must carry the matching condition."""

    combined = _rule_source_combined(rule)
    has_custom = "custom payments pricing" in combined or "custom pricing" in combined
    has_standard = "standard payments pricing" in combined or "standard pricing" in combined
    cond_values = {c.dimension: c.value for c in rule.conditions}
    if has_custom and cond_values.get("pricing_plan") != "custom":
        errors.append(
            f"{market_code}/{rule.rule_id}: source evidence says custom pricing but pricing_plan=custom condition missing"
        )
    if has_standard and cond_values.get("pricing_plan") != "standard":
        errors.append(
            f"{market_code}/{rule.rule_id}: source evidence says standard pricing but pricing_plan=standard condition missing"
        )


def _validate_rule_add_on_identity(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Add-on products must not be published as base payment-processing rules."""

    combined = _rule_source_combined(rule)
    if rule.product_id in _ADDON_PRODUCTS and rule.variant_id in _PAYMENTS_VARIANTS:
        errors.append(
            f"{market_code}/{rule.rule_id}: add-on product {rule.product_id} published as payments variant {rule.variant_id}"
        )
    add_on_in_source = None
    if "authorization boost" in combined or "authorisation boost" in combined:
        add_on_in_source = "authorization_boost"
    elif "smart dispute" in combined or "smart disputes" in combined:
        add_on_in_source = "smart_disputes"
    elif "3d secure" in combined or "3-d secure" in combined:
        add_on_in_source = "three_d_secure"
    elif "radar" in combined:
        add_on_in_source = "radar"
    if add_on_in_source and rule.product_id != add_on_in_source:
        errors.append(
            f"{market_code}/{rule.rule_id}: {add_on_in_source} source evidence published as product {rule.product_id}"
        )


def _validate_rule_smart_disputes(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Smart Disputes rules must require the Smart Disputes feature."""

    if rule.product_id != "smart_disputes":
        return
    cond_values = {c.dimension: c.value for c in rule.conditions}
    if cond_values.get("feature_enabled") != "smart_disputes":
        errors.append(
            f"{market_code}/{rule.rule_id}: smart_disputes rule missing feature_enabled=smart_disputes condition"
        )
    if cond_values.get("dispute_state") != "won":
        errors.append(f"{market_code}/{rule.rule_id}: smart_disputes rule missing dispute_state=won condition")


def _validate_rule_exactness_semantics(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Starting-at / starting-from phrases must be published as exactness=from."""

    combined = _rule_source_combined(rule)
    if ("starting at" in combined or "starting from" in combined) and rule.exactness != "from":
        errors.append(
            f"{market_code}/{rule.rule_id}: source evidence uses 'starting at' but exactness is {rule.exactness!r}"
        )


def _validate_rule_payer_condition(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Customer-paid conversion fees must carry a payer=customer condition."""

    combined = _rule_source_combined(rule)
    if "conversion fee" not in combined:
        return
    if "customer" not in combined and "customers" not in combined:
        return
    cond_values = {c.dimension: c.value for c in rule.conditions}
    if cond_values.get("payer") != "customer":
        errors.append(f"{market_code}/{rule.rule_id}: customer-paid conversion fee missing payer=customer condition")


_CARD_ONLY_DIMENSIONS = {"card_region", "card_origin", "card_tier"}


def _leading_method_in_label(label: str, method_by_display: dict[str, str]) -> str | None:
    """Return the payment method id if the rule label starts with a method name.

    Prefers the longest display-name match that begins earliest in the label, and
    ignores methods that appear after leading fee/qualifier text so that
    "0.2% per successful online card transaction" is not treated as a card rule.
    """
    if not label:
        return None
    words = label.split()[:6]
    head = " ".join(words).lower()
    candidates: list[tuple[int, int, int, str]] = []
    for display_name, method_id in method_by_display.items():
        pattern = re.compile(rf"\b{re.escape(display_name.lower())}\b")
        for match in pattern.finditer(head):
            word_index = len(head[: match.start()].split())
            # Allow a method name at the very start or immediately after a single
            # leading word such as a currency (e.g. "USD Bank Transfer").
            if word_index > 1:
                continue
            candidates.append((word_index, match.start(), -len(display_name), method_id))
    if not candidates:
        return None
    # Earliest word index, then earliest character start, then longest length.
    candidates.sort()
    return candidates[0][3]


def _validate_rule_product_identity(
    rule: CoreFeeRule,
    market_code: str,
    errors: list[str],
    method_by_display: dict[str, str],
) -> None:
    """Reject mismatches between payment method, product, and label.

    Examples of mismatches that must not be published:
      * payment_method=ach_direct_debit but product_id=disputes
      * a label beginning with "ACH Direct Debit" but product_id=refunds
      * unit=per_dispute on a payment-method base rate
    """
    leading_method = _leading_method_in_label(rule.label or "", method_by_display)
    if (
        leading_method
        and leading_method != "card"
        and rule.product_id
        not in {
            "payments",
            "terminal",
            leading_method,
        }
    ):
        errors.append(
            f"{market_code}/{rule.rule_id}: label begins with payment method {leading_method} "
            f"but product_id is {rule.product_id}"
        )
    elif leading_method == "card" and rule.product_id not in {"payments", "terminal"}:
        errors.append(
            f"{market_code}/{rule.rule_id}: label begins with card payment method but product_id is {rule.product_id}"
        )

    if rule.payment_method:
        if rule.product_id in {"payments"} and rule.payment_method != "card":
            errors.append(
                f"{market_code}/{rule.rule_id}: product {rule.product_id} has payment_method "
                f"{rule.payment_method} (expected card)"
            )
        if rule.product_id == "terminal" and rule.payment_method not in {"card", "terminal", "tap_to_pay"}:
            errors.append(
                f"{market_code}/{rule.rule_id}: terminal product has unexpected payment_method {rule.payment_method}"
            )
        if rule.product_id not in {"payments", "terminal"} and rule.payment_method != rule.product_id:
            errors.append(
                f"{market_code}/{rule.rule_id}: payment_method {rule.payment_method} does not match "
                f"product_id {rule.product_id}"
            )

    if rule.unit == "per_dispute" and rule.product_id not in {"disputes", "smart_disputes"}:
        fee_state_values = {c.value for c in rule.conditions if c.dimension in {"transaction_type", "dispute_state"}}
        if not (fee_state_values & {"dispute", "lost", "won", "received", "countered"}):
            errors.append(
                f"{market_code}/{rule.rule_id}: unit=per_dispute used for non-dispute product {rule.product_id}"
            )


def _validate_rule_non_card_conditions(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Non-card products must not carry card-only dimensions."""
    if rule.product_id in {"payments", "terminal"}:
        return
    for cond in rule.conditions:
        if cond.dimension in _CARD_ONLY_DIMENSIONS:
            errors.append(
                f"{market_code}/{rule.rule_id}: non-card product {rule.product_id} contains "
                f"card-only condition {cond.dimension}={cond.value}"
            )


def _validate_rule_evidence_duplicates(rule: CoreFeeRule, market_code: str, errors: list[str]) -> None:
    """Evidence phrases must not contain duplicate text."""
    if rule.fee_evidence and len(rule.fee_evidence.phrases) != len(set(rule.fee_evidence.phrases)):
        errors.append(f"{market_code}/{rule.rule_id}: fee_evidence.phrases contains duplicate text")


def _validate_core_fees_semantic(
    core_fees: CoreFees,
    manifest: MarketManifest,
    payment_methods: PaymentMethodCatalog,
) -> list[str]:
    errors: list[str] = []
    market_codes = {m.stripe_market_code for m in manifest.markets}
    method_by_display = {m.display_name: m.method_id for m in payment_methods.methods if m.display_name}
    for entry in core_fees.markets:
        if entry.stripe_market_code not in market_codes:
            errors.append(
                f"core-fees/{entry.account_country}: stripe_market_code {entry.stripe_market_code} "
                "not present in manifest.markets"
            )
        seen_identities: dict[tuple[str, str | None, Any], list[str]] = {}
        for rule in entry.rules:
            if _is_calculable_status(rule.classification_status):
                _validate_rule_currency_exponents(rule, entry.stripe_market_code, errors)
                _validate_rule_calculator_ready(rule, entry.stripe_market_code, errors)
                _validate_rule_contradictory_evidence(rule, entry.stripe_market_code, errors)
                _validate_rule_percentage_consistency(rule, entry.stripe_market_code, errors)
                _validate_rule_pricing_plan(rule, entry.stripe_market_code, errors)
                _validate_rule_add_on_identity(rule, entry.stripe_market_code, errors)
                _validate_rule_smart_disputes(rule, entry.stripe_market_code, errors)
                _validate_rule_exactness_semantics(rule, entry.stripe_market_code, errors)
                _validate_rule_payer_condition(rule, entry.stripe_market_code, errors)
                _validate_rule_market_share_evidence(rule, entry.stripe_market_code, errors)
            _validate_rule_cross_fragment_evidence(rule, entry.stripe_market_code, errors)
            _validate_rule_product_identity(rule, entry.stripe_market_code, errors, method_by_display)
            _validate_rule_non_card_conditions(rule, entry.stripe_market_code, errors)
            _validate_rule_evidence_duplicates(rule, entry.stripe_market_code, errors)
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
