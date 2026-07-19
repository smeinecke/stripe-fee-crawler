"""Dimension inference for fee rules."""

from __future__ import annotations

import logging
import re
from typing import Any

from ..models import FeeCondition, PricingEntry
from ..payment_methods import _earliest_payment_method
from ..pricing_tokens import parse_fee_value
from ._util import (
    _first_match,
    _has_base_fee,
    _heading_to_snake_case,
    _is_explicit_fee_phrase,
    _is_tap_to_pay,
    _nearest_heading_line,
    _text_has_lower,
)
from .tables import (
    _CARD_TIER_TABLE,
    _CONTRACT_LENGTH_TABLE,
    _COUNTRY_NAME_TO_CODE,
    _DIRECT_DEBIT_DEFAULT_UNITS,
    _EEA_COUNTRY_CODES,
    _METHOD_DEFAULT_UNITS,
    _PAYMENT_METHOD_TOKENS,
    _PAYMENT_METHOD_VARIANT_TABLE,
    _PRICING_PLAN_TABLE,
    _PRICING_TIER_TABLE,
    _PRODUCT_KEYWORDS,
    _PRODUCT_UNIT_DEFAULTS,
    _SETTLEMENT_TIMING_TABLE,
    _TRANSACTION_TYPE_TABLE,
    _UNIT_KEYWORD_TABLE,
)

logger = logging.getLogger(__name__)


def _infer_card_region(entry: PricingEntry, account_country: str | None = None) -> str | None:
    """Find the earliest card-region marker in the source text.

    Avoid matching "foreign exchange" / FX wording as a card-region signal.
    """
    text = " ".join(entry.section_path + [entry.source_text]).lower()
    region_markers = [
        (r"\bnon-eea\b", "international"),
        (r"\bnon eea\b", "international"),
        (r"\beea\b", "eea"),
        (r"\beuropean economic area\b", "eea"),
        (r"\buk\b", "uk"),
        (r"\bunited kingdom\b", "uk"),
        (r"\bbritish\b", "uk"),
        (r"\binternational\b", "international"),
        (r"\bforeign(?! exchange)\b", "international"),
        (r"\bdomestic\b", "domestic"),
        (r"\bissued outside(?: of)?\b", "international"),
        (r"\bissued in\b", "domestic"),
    ]
    earliest: tuple[int, str] | None = None
    for pattern, region in region_markers:
        for match in re.finditer(pattern, text):
            if earliest is None or match.start() < earliest[0]:
                earliest = (match.start(), region)
    if earliest:
        return earliest[1]
    # Generic card-payment entries with no other region marker are domestic for
    # the merchant's account country.  Tap to Pay is not a card-region variant.
    if "tap to pay" in text or entry.payment_method == "tap_to_pay":
        return None
    if account_country and "card" in text:
        return "domestic"
    return None


def _infer_card_tier(phrase: str) -> str | None:
    """Infer card tier only when the surrounding words refer to cards/tiers."""
    lower = phrase.lower()
    if "card" not in lower and "tier" not in lower and "scheme" not in lower:
        return None
    return _first_match(lower, _CARD_TIER_TABLE)


def _infer_card_type(text: str, account_country: str | None = None) -> str | None:
    """Detect explicit debit/credit card wording.

    Phrases such as "debit card transactions", "credit cards", or
    "Domestic debit card" are unambiguous.  Generic "cards" without a type
    marker is left as None so it does not force a false split.

    For India the generic domestic/international card rate is the credit-card
    rate because a separate domestic debit MDR rule exists.
    """
    lower = text.lower()
    # Debit markers must be checked before credit because "credit" is sometimes
    # used in generic marketing copy.
    if re.search(r"\bdebit\s+cards?\b|\bcards?\s+debit\b", lower):
        return "debit"
    if re.search(r"\bcredit\s+cards?\b|\bcards?\s+credit\b", lower):
        return "credit"
    # India splits the domestic/international card pricing by card type; the
    # generic "cards issued in/outside India" rate is the credit-card rate.
    if (
        account_country == "IN"
        and "card" in lower
        and "debit" not in lower
        and ("issued in" in lower or "issued outside" in lower)
    ):
        return "credit"
    return None


def _infer_card_network(entry: PricingEntry) -> list[str] | None:
    """Detect card network names in an entry's own text.

    Returns a list because some entries describe combined networks such as
    "Mastercard and Visa cards".
    """
    lower = entry.source_text.lower()
    networks: list[str] = []
    if "american express" in lower or "amex" in lower:
        networks.append("amex")
    if "mastercard" in lower:
        networks.append("mastercard")
    if "visa" in lower:
        networks.append("visa")
    return networks if networks else None


def _is_international_surcharge(text: str) -> bool:
    """Return True when a non-card surcharge is explicitly for international transactions."""
    lower = text.lower()
    return "for international transactions" in lower and ("+" in lower or "surcharge" in lower)


def _country_name_to_code(name: str) -> str | None:
    return _COUNTRY_NAME_TO_CODE.get(name.strip().lower().rstrip(","))


def _infer_customer_country(text: str, account_country: str | None) -> list[str] | None:
    """Parse a comma/and-separated list of country names into ISO codes.

    Only returns codes that are different from the merchant account country;
    domestic countries are left to the transaction_region/card_origin logic.
    """
    # Match a leading country list before the first price operator or digit.
    match = re.match(r"([A-Za-z][A-Za-z\s,]+(?:,\s*and\s+)?[A-Za-z])", text)
    if not match:
        return None
    list_text = match.group(1)
    # Split on commas and the word "and".
    parts = re.split(r",|\band\b", list_text)
    codes: list[str] = []
    for part in parts:
        code = _country_name_to_code(part)
        if code:
            codes.append(code)
    codes = sorted(set(codes))
    if not codes:
        return None
    # If the list only contains the merchant's own country it is a domestic row
    # and should not be tagged with customer_country.
    if account_country and set(codes) == {account_country}:
        return None
    return codes


def _infer_payment_method_variant(text: str, payment_method: str | None) -> str | None:
    """Detect Link/Affirm style sub-variants from trailing qualifiers."""
    lower = text.lower()
    # Link variants: "2.6% + ... for Instant Bank Payments", "... for Klarna"
    if payment_method == "link":
        variant = _first_match(lower, _PAYMENT_METHOD_VARIANT_TABLE[:2])
        if variant:
            return variant
    # Affirm variants: "Affirm Standard ...", "Affirm Enhanced ..."
    return _first_match(lower, _PAYMENT_METHOD_VARIANT_TABLE[2:])


def _infer_pricing_tier(text: str) -> str | None:
    """Detect standard/enhanced/premium tier qualifiers."""
    return _first_match(text, _PRICING_TIER_TABLE)


def _infer_contract_length(text: str) -> str | None:
    """Distinguish monthly vs annual subscription commitments."""
    return _first_match(text, _CONTRACT_LENGTH_TABLE)


def _infer_integration_type(entry: PricingEntry) -> str | None:
    """Distinguish Stripe Tax no-code vs API integration."""
    text = entry.source_text.lower()
    evidence = (entry.source_evidence or "").lower()
    if "no-code" in text or "no code" in text:
        return "no_code"
    if "api integration" in text or "payment apis" in text:
        return "api"

    # Stripe Tax cells contain multiple price rows (no-code, API, API overage).
    # Locate the price within the surrounding evidence and look up for the
    # nearest integration heading; if the price is not present in the evidence,
    # the entry belongs to the first integration section described there.
    price_match = re.search(r"\d+(?:\.\d+)?", entry.source_text)
    price_needle = price_match.group(0) if price_match else None
    if evidence:
        lines = evidence.splitlines()
        price_line = -1
        if price_needle:
            for i, line in enumerate(lines):
                if price_needle in line:
                    price_line = i
                    break
        if price_line >= 0:
            for i in range(price_line, -1, -1):
                line_l = lines[i].lower()
                if "api integration" in line_l or "payment apis" in line_l:
                    return "api"
                if "no-code integration" in line_l or "no code integration" in line_l:
                    return "no_code"
        # Price not in evidence, or no heading above it: use the first explicit
        # integration marker in the cell.
        for line in lines:
            line_l = line.lower()
            if "api integration" in line_l or "payment apis" in line_l:
                return "api"
            if (
                "no-code integration" in line_l
                or "no code integration" in line_l
                or "billing, checkout, invoicing, and payment links" in line_l
                or "taxes calculated and collected" in line_l
            ):
                return "no_code"
    return None


def _infer_product_feature(entry: PricingEntry, product_id: str) -> str | None:
    """Derive a feature/plan slug from the section heading or evidence context."""
    path = entry.section_path
    heading: str | None = None
    if path:
        heading = path[-1]
    if product_id == "radar":
        lower = heading.lower() if heading else ""
        if "fraud teams" in lower:
            return "fraud_teams"
        if "machine learning" in lower:
            return "machine_learning"
        return None
    if product_id == "tax":
        lower = heading.lower() if heading else ""
        if "complete" in lower:
            return "complete"
        if "basic" in lower:
            return "basic"
        return None
    if product_id in {"disputes", "smart_disputes"}:
        # Dispute-prevention add-ons list features in the evidence heading.
        nearest = _nearest_heading_line(entry)
        if nearest:
            return _heading_to_snake_case(nearest)
    return None


def _method_from_surcharge_context(entry: PricingEntry) -> str | None:
    """Find the payment method named in a surcharge's qualifier phrase.

    Surcharges such as ``+ $0.10 per authorization for Tap to Pay`` name the
    method in the same ``for``/``of``/``per``/``optional`` qualifier.  We only
    trust the surcharge's own source_text; the surrounding section evidence is too
    broad and can otherwise pull in unrelated methods from sibling rows.

    When the surcharge source text trails off with ``for`` or ``for optional``,
    the very next line of per-entry evidence is the method being qualified.
    """
    text = entry.source_text or ""
    text_lower = text.lower()
    for method in _PAYMENT_METHOD_TOKENS:
        display = method.replace("_", " ")
        if re.search(rf"\b(?:for|of|per|optional)\s+{re.escape(display)}s?\b", text_lower):
            return method

    # Trailing "for" or "for optional" defers the method to the next evidence line.
    if re.search(r"\b(for|for optional)\s*$", text_lower):
        first_evidence = (entry.source_evidence or "").splitlines()[0] if (entry.source_evidence or "") else ""
        first_evidence_lower = first_evidence.lower()
        for method in _PAYMENT_METHOD_TOKENS:
            display = method.replace("_", " ")
            if re.search(rf"\b{re.escape(display)}s?\b", first_evidence_lower):
                return method
    return None


def _infer_payment_method(entry: PricingEntry) -> str | None:
    evidence = entry.source_evidence or ""
    text = " ".join(entry.section_path + [entry.source_text, evidence]).lower()
    # Phrase-level tokens win over generic section headings, but a method name
    # must appear at the start of the phrase/heading.  This prevents hidden LPM
    # card content ("for Instant Bank Payments", "for Klarna") from hijacking
    # the method identity of a Link, Card payments, etc. base fee.
    # Only treat the entry as tap_to_pay when the phrase itself starts with the
    # method; marketing prose in the surrounding evidence must not hijack the
    # primary card method.
    if "tap to pay" in (entry.source_text.lower() + " " + " ".join(entry.section_path).lower()):
        return "tap_to_pay"
    leading = _earliest_payment_method(entry.source_text, max_word_index=1)
    if leading:
        if leading == "terminal" and "card" in entry.source_text.lower():
            return "card"
        return leading
    # Surcharge fragments may name their method in a trailing qualifier.
    if entry.source_text.strip().startswith("+"):
        surcharge_method = _method_from_surcharge_context(entry)
        if surcharge_method:
            return surcharge_method
    # Fall back to the earliest explicit method token anywhere in the text.
    earliest = _earliest_payment_method(text)
    if earliest:
        if earliest == "terminal" and "card" in text:
            return "card"
        return earliest
    if entry.payment_method:
        if entry.payment_method == "terminal" and "card" in text:
            return "card"
        return entry.payment_method
    return None


def _infer_channel(entry: PricingEntry) -> str | None:
    if entry.channel:
        return entry.channel
    evidence = entry.source_evidence or ""
    text = " ".join(entry.section_path + [entry.source_text, evidence]).lower()
    if "terminal" in text or "in-person" in text or "in person" in text or "tap to pay" in text or "reader" in text:
        return "in_person"
    if "online" in text or "checkout" in text or "payment link" in text:
        return "online"
    # Card/payment entries with no in-person signal are online by default.
    if "card" in text or "payment" in text or "wallet" in text:
        return "online"
    return None


def _card_origin_for_region(card_region: str | None, account_country: str | None) -> str | None:
    """Map a card region to domestic/international relative to the merchant country."""
    if card_region is None:
        return "domestic" if account_country else None
    lower = card_region.lower()
    if lower == "domestic":
        return "domestic"
    if account_country and account_country.upper() in _EEA_COUNTRY_CODES and lower in {"eea", "european economic area"}:
        return "domestic"
    if account_country == "GB" and lower == "uk":
        return "domestic"
    if account_country and lower == account_country.lower():
        return "domestic"
    return "international"


def _is_card_product(product_id: str | None) -> bool:
    """Return True for products whose pricing is inherently card-based."""
    return product_id in {"payments", "terminal"}


def _is_payment_method_product(product_id: str | None) -> bool:
    """Return True when the product is a specific payment method or card family."""
    if not product_id:
        return False
    return product_id in _PAYMENT_METHOD_TOKENS or product_id in {"payments", "terminal"}


def _product_from_heading(fee_category: str | None) -> str | None:
    """Return the product id implied by a section heading, if unambiguous."""
    if not fee_category:
        return None
    lower = fee_category.lower()
    for method in _PAYMENT_METHOD_TOKENS:
        if method in {"card", "terminal"}:
            continue
        if method.replace("_", " ") in lower:
            return None

    for product, keywords in _PRODUCT_KEYWORDS:
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}", lower):
                return product
    return None


def _is_dedicated_dispute_refund_fee(entry: PricingEntry) -> bool:
    """Return True when the fee amount belongs to a dispute/refund/failure caption.

    A payment-method base fee may trail off with text-only dispute/refund/failure
    qualifiers (e.g. "... for disputed payments").  Those must keep the method
    product.  A dedicated fee has the amount immediately before the keyword
    (e.g. "$15.00 for disputed payments", "$0.50 per successful refund").
    """
    text = entry.source_text
    if not text:
        return False
    keywords = [
        "for lost disputes",
        "disputed payment",
        "failed payment",
        "per successful refund",
        "refund",
    ]
    text_lower = text.lower()
    for keyword in keywords:
        pos = text_lower.find(keyword)
        if pos == -1:
            continue
        # Find the last numeric token before the keyword.
        prefix = text[:pos]
        last_number: re.Match[str] | None = None
        for match in re.finditer(r"\d+(?:\.\d+)?", prefix):
            last_number = match
        if last_number is None:
            continue
        distance = pos - last_number.end()
        # The amount should be immediately before the keyword (a handful of words apart).
        if distance <= 40:
            return True
    return False


def _infer_product_id(entry: PricingEntry) -> str:
    """Determine the semantic product id for a pricing entry.

    Trust the extracted payment method first so that trailing qualifiers such as
    "for disputed payments" or "for refunds" cannot promote a payment-method
    base fee into the disputes/refunds product.
    """
    heading = entry.fee_category or (entry.section_path[-1] if entry.section_path else None)
    heading_product = _product_from_heading(heading)
    if heading_product:
        return heading_product

    method = _infer_payment_method(entry)
    if method:
        if method == "card":
            if _infer_channel(entry) == "in_person" or _text_has_lower(
                " ".join(p.lower() for p in entry.section_path), "terminal", "tap to pay", "in-person", "in person"
            ):
                return "terminal"
            return "payments"
        if method in {"terminal", "tap_to_pay"}:
            return "terminal"
        return method

    # No explicit payment method: infer from the section heading / fee category,
    # the source text, and the surrounding evidence.  Add-on and non-payment
    # headings can appear in the surrounding cell text (e.g. "Subscription and
    # cancellation terms apply" under a "Pay monthly" row).
    path = " ".join(p.lower() for p in entry.section_path)
    category = (entry.fee_category or "").lower()
    text = entry.source_text.lower()
    evidence = (entry.source_evidence or "").lower()
    combined = path + " " + category + " " + text + " " + evidence

    for product, keywords in _PRODUCT_KEYWORDS:
        if _text_has_lower(combined, *keywords):
            return product

    text = entry.source_text.lower()
    combined = path + " " + text
    # Generic payment processing is card-based; avoid treating generic
    # "payment methods" headings as card payments when no card or explicit
    # method token is present.
    if "card" in combined or (
        "payment" in combined and not _text_has_lower(combined, "payment methods", "payment method")
    ):
        return "payments"
    # Surcharges for regional payment-method groups (e.g. "South Korean payment
    # methods + 1.5% for international transactions") are payment fees even though
    # no individual method token is present.
    if _has_base_fee(entry) and "payment methods" in combined:
        return "payments"
    if _text_has_lower(combined, "ach"):
        return "ach_direct_debit"
    if "sepa" in combined:
        return "sepa_direct_debit"
    return "unspecified"


def _variant_id_for(
    product_id: str,
    payment_method: str | None,
    channel: str | None,
    card_region: str | None,
    card_tier: str | None,
    card_origin: str | None,
    card_type: str | None,
) -> str:
    """Build a stable variant id from the inferred dimensions."""
    if product_id == "payments":
        if card_tier == "premium":
            return "online_premium_cards" if channel != "in_person" else "in_person_premium_cards"
        prefix = "in_person" if channel == "in_person" else "online"
        origin = (
            "international"
            if (card_origin == "international" or card_region in {"international", "non-eea", "non eea", "uk"})
            else "domestic"
        )
        suffix = f"_{card_type}" if card_type else ""
        return f"{prefix}_{origin}{suffix}_cards"
    if product_id == "terminal":
        if payment_method == "tap_to_pay":
            return "tap_to_pay"
        origin = (
            "international"
            if (card_origin == "international" or card_region in {"international", "non-eea", "non eea"})
            else "domestic"
        )
        suffix = f"_{card_type}" if card_type else ""
        return f"{origin}{suffix}_cards"
    return "standard"


def _infer_pricing_plan(entry: PricingEntry) -> str | None:
    """Return custom/standard if the entry is explicitly scoped to a plan."""
    combined = " ".join(p.lower() for p in entry.section_path) + " " + entry.source_text.lower()
    # Check the longer phrase first so "custom payments pricing" wins over "custom pricing".
    return _first_match(combined, _PRICING_PLAN_TABLE)


def _infer_variant_id(entry: PricingEntry, product_id: str, account_country: str | None) -> str:
    pricing_plan = _infer_pricing_plan(entry)
    product_feature = _infer_product_feature(entry, product_id)

    # Add-on products use stable variants keyed to their pricing plan, and
    # Smart Disputes uses a won/lost variant keyed to the dispute state.
    if product_id in {"three_d_secure", "authorization_boost", "radar"} and pricing_plan:
        return f"{pricing_plan}_pricing"
    if product_id == "smart_disputes":
        lower = entry.source_text.lower()
        if re.search(r"\b(won|win)\b", lower):
            return "won_dispute"
        if "lost" in lower:
            return "lost_dispute"
        return "standard"
    if product_id == "adaptive_pricing" and pricing_plan:
        return f"{pricing_plan}_pricing"

    # Products whose headings or evidence explicitly name a tier/feature use
    # that as the variant (e.g. Tax Basic vs Tax Complete).
    if product_id in {"tax", "identity", "billing"} and product_feature:
        return product_feature

    channel = _infer_channel(entry) or "online"
    payment_method = _infer_payment_method(entry)
    card_region = _infer_card_region(entry, account_country)
    card_tier = _infer_card_tier(entry.source_text)
    card_origin = _card_origin_for_region(card_region, account_country)
    card_type = _infer_card_type(entry.source_text, account_country)
    return _variant_id_for(product_id, payment_method, channel, card_region, card_tier, card_origin, card_type)


def _infer_exactness(parsed: dict[str, Any], phrase: str) -> str:
    exactness = parsed.get("exactness") or "exact"
    lower = phrase.lower()
    if _text_has_lower(lower, "contact sales", "custom quote"):
        return "custom"
    if _text_has_lower(lower, "starting at", "starting from", "starts at", "starts from"):
        return "from"
    if re.search(r"\b(?:capped at|cap at|cap|maximum|max)\b", lower):
        return "range"
    if "up to" in lower and not re.search(r"up to \d+\s?(month|months|day|days|week|weeks|year|years)", lower):
        return "range"
    if re.search(r"\b(?:minimum|min)\b", lower):
        return "from"
    if _text_has_lower(lower, "included", "no additional fee"):
        return "included"
    if "free" in lower or "no fee" in lower:
        return "free"
    # Package/allotment pricing with overages cannot be represented as a
    # simple per-event fee, so keep it as a custom-quote row.
    if _text_has_lower(lower, "overages", "overage", "allotment", "volume pricing", "custom package"):
        return "custom"
    # A concrete fee with "custom pricing" wording is an explicitly scoped paid
    # variant, not a custom-quote exactness.
    if (
        exactness == "custom"
        and (parsed.get("percentage") or parsed.get("fixed_amount"))
        and _is_explicit_fee_phrase(phrase)
    ):
        return "exact"
    return exactness


def _infer_conditions(
    entry: PricingEntry,
    product_id: str,
    variant_id: str,
    account_country: str | None,
) -> list[FeeCondition]:
    conditions: list[FeeCondition] = []
    text = entry.source_text.lower()
    path = " ".join(p.lower() for p in entry.section_path)
    combined = path + " " + text

    # Card-region, origin and tier are only meaningful for card-based products.
    # For non-card products, international/domestic wording describes the
    # transaction region / cross-border nature of the payment, not card origin.
    card_region = _infer_card_region(entry, account_country)
    if _is_card_product(product_id):
        if card_region:
            conditions.append(FeeCondition(dimension="card_region", value=card_region))
            card_origin = _card_origin_for_region(card_region, account_country)
            if card_origin:
                conditions.append(FeeCondition(dimension="card_origin", value=card_origin))
        card_tier = _infer_card_tier(entry.source_text)
        if card_tier:
            conditions.append(FeeCondition(dimension="card_tier", value=card_tier))
        card_type = _infer_card_type(entry.source_text, account_country)
        if card_type:
            conditions.append(FeeCondition(dimension="card_type", value=card_type))
        card_network = _infer_card_network(entry)
        if card_network:
            conditions.append(
                FeeCondition(
                    dimension="card_network", value=card_network[0] if len(card_network) == 1 else sorted(card_network)
                )
            )
        if card_region == "uk":
            conditions.append(FeeCondition(dimension="customer_country", value="GB"))
        if "usd or other currency" in text or "other currency presentment" in text:
            conditions.append(FeeCondition(dimension="presentment_currency", value=["USD", "other"]))
    else:
        if card_region:
            # Surcharges explicitly scoped to "international transactions" must
            # not inherit a country-list region such as "uk" from the product name.
            if _is_international_surcharge(text):
                card_region = "international"
            if card_region == "domestic":
                conditions.append(FeeCondition(dimension="cross_border", value=False))
                conditions.append(FeeCondition(dimension="transaction_region", value="domestic"))
            else:
                conditions.append(FeeCondition(dimension="cross_border", value=True))
                conditions.append(
                    FeeCondition(
                        dimension="transaction_region",
                        value=card_region,
                    )
                )

    payment_method = _infer_payment_method(entry)
    if payment_method and _is_card_product(product_id):
        conditions.append(FeeCondition(dimension="payment_method", value=payment_method))

    channel = _infer_channel(entry)
    if channel:
        conditions.append(FeeCondition(dimension="channel", value=channel))

    if "currency conversion" in combined or "if currency conversion is required" in text:
        # A domestic-card heading followed by an international/currency-conversion
        # note (e.g. a "Show additional fees" toggle) should not inherit the
        # currency-conversion condition; it belongs to the separate international
        # variant whose rate lives on the main pricing page.
        currency_pos = text.find("currency conversion")
        if currency_pos == -1:
            currency_pos = text.find("if currency conversion is required")
        domestic_pos = text.find("domestic")
        if domestic_pos == -1 or currency_pos == -1 or domestic_pos > currency_pos:
            conditions.append(FeeCondition(dimension="currency_conversion_required", value=True))
    settlement_timing = _first_match(text, _SETTLEMENT_TIMING_TABLE)
    if settlement_timing:
        conditions.append(FeeCondition(dimension="settlement_timing", value=settlement_timing))

    pricing_plan = _infer_pricing_plan(entry)
    if pricing_plan:
        conditions.append(FeeCondition(dimension="pricing_plan", value=pricing_plan))

    product_feature = _infer_product_feature(entry, product_id)
    if product_feature:
        conditions.append(FeeCondition(dimension="product_feature", value=product_feature))

    integration_type = _infer_integration_type(entry)
    if integration_type:
        conditions.append(FeeCondition(dimension="integration_type", value=integration_type))

    method_variant = _infer_payment_method_variant(entry.source_text, payment_method)
    if method_variant:
        conditions.append(FeeCondition(dimension="payment_method_variant", value=method_variant))

    tier = _infer_pricing_tier(entry.source_text)
    if tier:
        conditions.append(FeeCondition(dimension="pricing_tier", value=tier))

    contract_length = _infer_contract_length(entry.source_text)
    if contract_length:
        conditions.append(FeeCondition(dimension="contract_length", value=contract_length))

    customer_country = _infer_customer_country(entry.source_text, account_country)
    if customer_country:
        conditions.append(FeeCondition(dimension="customer_country", value=customer_country))

    if "wire" in text and ("wire payment" in text or "wire transfer" in text):
        conditions.append(FeeCondition(dimension="transaction_type", value="wire"))
        if payment_method:
            conditions.append(FeeCondition(dimension="payment_method_variant", value="wire"))

    if product_id in {"smart_disputes", "disputes"} and _text_has_lower(combined, "smart dispute", "smart disputes"):
        conditions.append(FeeCondition(dimension="feature_enabled", value="smart_disputes"))

    if product_id == "adaptive_pricing" or _text_has_lower(combined, "adaptive pricing"):
        if "customer" in combined and ("pay" in text or "present" in text or "bear" in text):
            conditions.append(FeeCondition(dimension="payer", value="customer"))
        if "conversion" in combined:
            conditions.append(FeeCondition(dimension="fee_type", value="conversion_fee"))

    # Dedicated dispute/refund/failure fees keep their payment-method product but
    # carry a transaction_type/dispute_state condition so they do not pollute the
    # base method rule.
    if _is_dedicated_dispute_refund_fee(entry):
        if "lost disputes" in combined or "disputed payment" in combined:
            conditions.append(FeeCondition(dimension="dispute_state", value="lost"))
            conditions.append(FeeCondition(dimension="transaction_type", value="dispute"))
        elif "failed payment" in combined:
            conditions.append(FeeCondition(dimension="transaction_type", value="failed"))
        elif "refund" in combined:
            conditions.append(FeeCondition(dimension="transaction_type", value="refund"))

    # Dispute/refund/failure state is only meaningful for the dispute/refund
    # product families.  A trailing "for lost disputes" qualifier on a payment-
    # method base fee must not turn that base fee into a dispute-only rule.
    if product_id in {"disputes", "smart_disputes", "refunds"}:
        if "won disputes" in combined or ("dispute" in combined and re.search(r"\b(won|win)\b", text)):
            conditions.append(FeeCondition(dimension="dispute_state", value="won"))
        elif "lost disputes" in combined or ("dispute" in combined and re.search(r"\blost\b", text)):
            conditions.append(FeeCondition(dimension="dispute_state", value="lost"))
        elif "received" in combined and "dispute" in combined:
            conditions.append(FeeCondition(dimension="dispute_state", value="received"))
        elif "countered" in combined or ("respond" in combined and "dispute" in combined):
            conditions.append(FeeCondition(dimension="dispute_state", value="countered"))

    if product_id in {"smart_disputes", "disputes"} and not any(c.dimension == "transaction_type" for c in conditions):
        conditions.append(FeeCondition(dimension="transaction_type", value="dispute"))

    if "recurring" in path or "subscription" in path:
        conditions.append(FeeCondition(dimension="recurring", value=True))
    if "one-time" in path or "one time" in path:
        conditions.append(FeeCondition(dimension="recurring", value=False))
    if "managed payments" in path:
        conditions.append(FeeCondition(dimension="managed_payments", value=True))
    if "per successful transaction" in text or "per successful charge" in text:
        conditions.append(FeeCondition(dimension="success", value=True))

    transaction_type = _first_match(text, _TRANSACTION_TYPE_TABLE)
    if transaction_type:
        conditions.append(FeeCondition(dimension="transaction_type", value=transaction_type))
    elif product_id == "instant_payouts":
        conditions.append(FeeCondition(dimension="transaction_type", value="payout"))

    if "manually entered" in text or "manual entry" in text or "moto" in text:
        conditions.append(FeeCondition(dimension="card_entry_mode", value="manual"))

    return conditions


def _infer_unit(entry: PricingEntry, product_id: str) -> str | None:
    text = entry.source_text.lower()
    is_dedicated = _is_dedicated_dispute_refund_fee(entry)
    if product_id == "disputes" or (
        is_dedicated and ("disputed payment" in text or "lost disputes" in text or "smart dispute" in text)
    ):
        return "per_dispute"
    if is_dedicated and "failed payment" in text:
        return "per_attempt"
    unit = _first_match(text, _UNIT_KEYWORD_TABLE)
    if unit:
        return unit
    # Default unit for entries with a recognizable payment method or product.
    method = _infer_payment_method(entry)
    if method:
        return "per_transaction"
    if product_id in _DIRECT_DEBIT_DEFAULT_UNITS:
        return "per_transaction"
    if product_id in _PRODUCT_UNIT_DEFAULTS:
        return _PRODUCT_UNIT_DEFAULTS[product_id]
    if product_id in _METHOD_DEFAULT_UNITS:
        return "per_transaction"
    return None


def _entry_component_hint(entry: PricingEntry) -> str:
    """Characterise the role of this entry within its logical pricing row."""
    parsed = parse_fee_value(entry.source_text)
    # For rows that already state a concrete fee, marketing prose in the
    # surrounding cell should not override the fee's exactness.
    if _has_base_fee(entry, parsed) and _is_explicit_fee_phrase(entry.source_text):
        text = entry.source_text
    else:
        text = entry.source_text + " " + (entry.source_evidence or "")
    lower = text.lower()
    exactness = _infer_exactness(parsed, text)

    if exactness == "range" or re.search(r"\b(?:cap|capped|maximum|max)\b", lower):
        return "maximum"
    if re.search(r"\b(?:minimum|min)\b", lower):
        return "minimum"
    if exactness == "included" or "no additional" in lower:
        return "included"
    if exactness == "free":
        return "free"
    if exactness == "custom" or "contact sales" in lower:
        return "custom"
    if exactness == "from" or "starting at" in lower or "starting from" in lower:
        return "from"
    # An "uplift" is a fee itself, not a surcharge applied on top of another
    # fee (e.g. Adaptive Acceptance uplift).
    if "uplift" in lower:
        return "base"
    tokens = entry.tokens or parsed.get("tokens") or []
    is_single_surcharge_token = len(tokens) == 1 and tokens[0].operator == "+" and not _is_tap_to_pay(entry)
    if (entry.source_text.strip().startswith("+") or is_single_surcharge_token) and not _is_tap_to_pay(entry):
        return "surcharge"
    return "base"


__all__ = [
    "_infer_card_region",
    "_infer_card_tier",
    "_infer_card_type",
    "_infer_card_network",
    "_is_international_surcharge",
    "_country_name_to_code",
    "_infer_customer_country",
    "_infer_payment_method_variant",
    "_infer_pricing_tier",
    "_infer_contract_length",
    "_infer_integration_type",
    "_infer_product_feature",
    "_method_from_surcharge_context",
    "_infer_payment_method",
    "_infer_channel",
    "_card_origin_for_region",
    "_is_card_product",
    "_is_payment_method_product",
    "_product_from_heading",
    "_is_dedicated_dispute_refund_fee",
    "_infer_product_id",
    "_variant_id_for",
    "_infer_pricing_plan",
    "_infer_variant_id",
    "_infer_exactness",
    "_infer_conditions",
    "_infer_unit",
    "_entry_component_hint",
]
