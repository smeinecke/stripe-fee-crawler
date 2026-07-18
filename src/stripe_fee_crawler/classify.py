"""Classification of Stripe pricing entries into calculation-ready fee rules."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from .models import (
    CoverageSummary,
    FeeComponent,
    FeeCondition,
    FeeEvidence,
    FeeRule,
    Market,
    PricingEntry,
)
from .normalize import stable_id
from .pricing_tokens import CURRENCY_CODES, CURRENCY_SYMBOLS, currency_exponent, parse_fee_value

logger = logging.getLogger(__name__)

# Primary classification states for source pricing records and derived rules.
CALCULABLE_RULE = "calculable_rule"
NON_CALCULABLE = "non_calculable"
CUSTOM_PRICING = "custom_pricing"
INCLUDED = "included"
FREE = "free"
INFORMATIONAL = "informational"
UNSUPPORTED_SHAPE = "unsupported_fee_shape"
UNCLASSIFIED_CANDIDATE = "unclassified_fee_candidate"
IGNORED_NON_FEE = "ignored_non_fee"
REFERENCE_ONLY = "reference_only"
AMBIGUOUS = "ambiguous"

# EEA country codes. Cards issued in these regions are treated as domestic for
# merchants located in any of these countries.
_EEA_COUNTRY_CODES: set[str] = {
    "AT",
    "BE",
    "BG",
    "HR",
    "CY",
    "CZ",
    "DK",
    "EE",
    "FI",
    "FR",
    "DE",
    "GR",
    "HU",
    "IS",
    "IE",
    "IT",
    "LV",
    "LI",
    "LT",
    "LU",
    "MT",
    "NL",
    "NO",
    "PL",
    "PT",
    "RO",
    "SK",
    "SI",
    "ES",
    "SE",
}

# Normalized payment method tokens that can appear in headings or fee phrases.
_PAYMENT_METHOD_TOKENS: tuple[str, ...] = (
    "sepa_direct_debit",
    "sepa_bank_transfer",
    "ach_direct_debit",
    "bacs_direct_debit",
    "pre_authorized_debit",
    "pad",
    "bancontact",
    "bizum",
    "blik",
    "eps",
    "ideal",
    "wero",
    "przelewy24",
    "swish",
    "twint",
    "pay_by_bank",
    "mb_way",
    "pix",
    "upi",
    "klarna",
    "billie",
    "scalapay",
    "multibanco",
    "alipay",
    "mobilepay",
    "paypal",
    "revolut_pay",
    "wechat_pay",
    "amazon_pay",
    "satispay",
    "konbini",
    "apple_pay",
    "google_pay",
    "click_to_pay",
    "cash_app_pay",
    "cash_app_afterpay",
    "afterpay",
    "clearpay",
    "affirm",
    "zip",
    "sunbit",
    "tap_to_pay",
    "link",
    "paypay",
    "card",
    "terminal",
    "bank_transfer",
)

_CARD_NETWORK_TOKENS: tuple[str, ...] = (
    "mastercard",
    "visa",
    "american express",
    "american_express",
    "amex",
    "discover",
    "jcb",
    "diners",
    "diners club",
    "unionpay",
)

# Evidence vocabulary for positive fee classification and for rejecting common
# false-positive sources.
_ADDON_PRODUCTS: tuple[str, ...] = (
    "authorization_boost",
    "radar",
    "smart_disputes",
    "three_d_secure",
)

_POSITIVE_FEE_TERMS: tuple[str, ...] = (
    "fee",
    "fees",
    "cost",
    "costs",
    "charge",
    "charged",
    "starting fee",
    "platform fee",
    "uplift",
    "pricing",
    "per transaction",
    "per successful charge",
    "per successful transaction",
    "per authorization",
    "per authorisation",
    "per payout",
    "per payout paid out",
    "per dispute",
    "per dispute payment",
    "per resolution",
    "per block",
    "per lookup",
    "per deflection",
    "per invoice",
    "per paid invoice",
    "per authorised",
    "per authorized",
    "per refund",
    "for refunds",
    "per verification",
    "per charge",
    "per screened transaction",
    "per connected account",
    "per active user",
    "per user",
    "per month",
    "monthly",
    "per year",
    "yearly",
    "maximum fee",
    "minimum fee",
    "max fee",
    "min fee",
    "cap",
    "capped",
    "percentage plus fixed amount",
    "processing fee",
    "transaction fee",
    "payment fee",
    "billing volume",
    "of billing volume",
    "currency conversion",
    "foreign exchange",
    "for international",
    "for domestic",
    "for cards",
    "for uk cards",
    "payment methods",
    "cross-border",
    "cross border",
)

_MARKETING_TERMS: tuple[str, ...] = (
    "billion",
    "million",
    "trillion",
    "category leaders",
    "customers",
    "customer",
    "process more than",
    "process over",
    "api calls",
    "api requests",
    "revenue",
    "transaction volume",
    "transaction-volume",
    "ranking",
    "rankings",
    "100+",
    "100 +",
    "percent of",
    "percentage of",
    "year",
    "years",
    "annual",
    "annually",
    "share of",
    "market share",
    "most popular payment method",
    "popular payment method",
    "used in more than",
    "used by over",
    "active monthly users",
    "active global customers",
    "customers use",
    "adoption",
    "increase conversion",
    "increase acceptance",
    "return on investment",
    "roi",
    "uptime",
    "historical uptime",
)

_PROMOTIONAL_TERMS: tuple[str, ...] = (
    "may qualify",
    "temporarily reduced",
    "temporarily lower",
    "contact sales",
    "contact us",
    "custom quote",
    "starting prices may apply",
    "waive",
    "waived",
    "try for free",
)

# Phrases that indicate a percentage is a market/adoption statistic, not a fee.
_MARKET_SHARE_STATISTICS: tuple[str, ...] = (
    "share of online payments",
    "share of online transactions",
    "share of e-commerce payments",
    "market share",
    "most popular payment method",
    "used in more than",
    "used by over",
    "active monthly users",
    "active global customers",
    "customers use",
    "adoption",
    "increase conversion",
    "increase acceptance",
    "uptime",
    "historical uptime",
    "api requests",
    "api calls",
    "billion",
    "million",
    "trillion",
)

_HARDWARE_PRICE_TERMS: tuple[str, ...] = (
    "reader",
    "readers",
    "device",
    "devices",
    "purchase",
    "purchased",
    "hardware",
    "terminal",
    "tap to pay",
    "price",
    "one-time",
)

_PUBLICATION_CONFIDENCE_THRESHOLD = 0.7
_MIN_HARDWARE_MAJOR_AMOUNT = Decimal("10.0")


def _fixed_amount_minor(amount: str, currency: str) -> str | None:
    """Convert a major-unit amount to minor units for a currency."""
    try:
        dec = Decimal(amount)
        exponent = currency_exponent(currency)
        multiplier = Decimal(10) ** exponent
        minor = int(dec * multiplier)
        return str(minor)
    except Exception:
        return None


def _text_has(text: str, *terms: str) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _source_text_for_group(group: list[dict[str, Any]]) -> str:
    """Combine all source texts and section headings for the group."""
    parts: list[str] = []
    for item in group:
        entry = item["entry"]
        parts.extend(entry.section_path)
        parts.append(entry.source_text)
    return " ".join(p for p in parts if p).lower()


def _is_explicit_fee_phrase(text: str) -> bool:
    """Return True when the text contains explicit fee-calculation language."""
    return _text_has(text, *_POSITIVE_FEE_TERMS)


def _is_marketing_prose(text: str) -> bool:
    """Return True for statistics, volume claims, and other marketing copy."""
    lower = text.lower()
    # Volume/audience claims are marketing even if they also contain fee-adjacent
    # words such as "monthly" or "payments".
    if ("million" in lower or "billion" in lower or "trillion" in lower) and (
        "customers" in lower or "users" in lower or "addressable" in lower or "process" in lower
    ):
        return True
    # A phrase that already contains explicit fee language is a fee description,
    # not marketing prose (e.g. "Customers will be presented a conversion fee").
    if _is_explicit_fee_phrase(text):
        return False
    return _text_has(text, *_MARKETING_TERMS)


def _is_market_share_text(text: str) -> bool:
    """Return True when the text is a market-share/adoption statistic."""
    lower = text.lower()
    if not _text_has(lower, *_MARKET_SHARE_STATISTICS):
        return False
    # A statistic is only a false-positive fee if it actually contains a number.
    parsed = parse_fee_value(text)
    return bool(parsed["percentage"] or parsed["fixed_amount"])


def _is_promotional_language(text: str) -> bool:
    """Return True for conditional or sales-led pricing language."""
    return _text_has(text, *_PROMOTIONAL_TERMS)


def _has_hardware_context(text: str) -> bool:
    """Return True when the text describes a terminal/device purchase price."""
    return _text_has(text, *_HARDWARE_PRICE_TERMS)


def _is_terminal_hardware_price(
    entry: PricingEntry,
    product_id: str,
    fee_components: list[FeeComponent],
    unit: str | None,
) -> bool:
    """Detect terminal reader/device prices that should not be per-transaction fees."""
    if product_id != "terminal" and entry.payment_method != "terminal":
        return False
    text = (" ".join(entry.section_path + [entry.source_text])).lower()
    # A true terminal processing fee normally mentions a per-event unit.
    explicit_processing_context = _text_has(
        text,
        "per successful",
        "per transaction",
        "per charge",
        "per in-person",
        "authorization fee",
        "processing fee",
        "transaction fee",
        "fee",
        "fees",
    )
    fixed_components = [c for c in fee_components if c.type in {"fixed_amount", "fixed_surcharge"}]
    if not fixed_components:
        return False
    largest_fixed = max(
        (Decimal(c.amount) for c in fixed_components if c.amount and c.currency),
        default=Decimal("0"),
    )
    if largest_fixed < _MIN_HARDWARE_MAJOR_AMOUNT:
        return False
    if explicit_processing_context:
        return False
    return _has_hardware_context(text)


def _is_amount_from_method_name(
    entry: PricingEntry,
    fee_components: list[FeeComponent],
) -> bool:
    """Detect amounts that were parsed out of product/payment-method names."""
    text = entry.source_text
    for comp in fee_components:
        if comp.type in {"fixed_amount", "fixed_surcharge", "maximum_fee", "minimum_fee"} and comp.amount:
            # If the amount text appears right after an alphabetic payment-method
            # token (e.g. "P24" -> "24"), it is not a real fee amount.
            # Ignore amounts preceded by a currency symbol or ISO code.
            raw = comp.source_text or text or ""
            if not raw:
                continue
            for match in re.finditer(re.escape(comp.amount) + r"\b", raw):
                prefix = raw[: match.start()].rstrip()
                if not prefix:
                    continue
                token = prefix.split()[-1]
                if token.upper() in CURRENCY_CODES or token in CURRENCY_SYMBOLS:
                    continue
                if re.fullmatch(r"[A-Za-z]+", token):
                    return True
    return False


def _has_contradictory_fee_evidence(fee_components: list[FeeComponent]) -> bool:
    """Detect positive-fee evidence paired with included/free evidence."""
    component_types = {c.type for c in fee_components}
    positive_types = {
        "percentage",
        "fixed_amount",
        "percentage_surcharge",
        "fixed_surcharge",
        "maximum_fee",
        "minimum_fee",
    }
    has_positive = bool(component_types & positive_types)
    has_included_free = bool(component_types & {"included", "free"})
    return has_positive and has_included_free


def _fee_evidence_for_group(
    group: list[dict[str, Any]],
    product_id: str,
    fee_components: list[FeeComponent],
    unit: str | None,
) -> FeeEvidence:
    """Evaluate the evidence behind a would-be calculable rule."""
    base_entry = group[0]["entry"]
    combined = _source_text_for_group(group)
    # For price rows split from a surrounding cell, the noun after "per" or
    # the fee phrase may live in the surrounding evidence. Include it when the
    # source text itself is only an amount or ends with a dangling "per".
    text_for_evidence = combined
    if base_entry.source_text.rstrip().lower().endswith(" per"):
        per_noun = _per_completion_noun(base_entry)
        if per_noun:
            text_for_evidence = f"{combined} per {per_noun}"
        else:
            text_for_evidence = f"{combined} {base_entry.source_evidence or ''}".strip()
    elif not re.search(r"[a-z]{3,}", base_entry.source_text.lower()):
        text_for_evidence = f"{combined} {base_entry.source_evidence or ''}".strip()
    entry_ids = [item["entry"].entry_id for item in group]
    raw_phrases = [
        item["entry"].source_text
        for item in group
        if not _is_marketing_prose(_dedup_repeated_phrases(item["entry"].source_text))
    ]
    phrases = _ordered_unique(_dedup_repeated_phrases(p) for p in raw_phrases)

    # 1. Promotional / conditional pricing is never a concrete fee.
    if _is_promotional_language(combined):
        return FeeEvidence(
            type="promotional_language",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.1,
        )

    # 2. Marketing prose with numbers is not a fee. Use the base entry text
    # (not the combined group text) so a section heading with fee language does
    # not mask marketing copy in the body.
    if _is_marketing_prose(_dedup_repeated_phrases(base_entry.source_text)):
        return FeeEvidence(
            type="marketing_prose",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 2b. Cross-fragment alignment: the numeric fee value and the fee wording
    # must come from the same logical pricing record. If a market-share/statistic
    # fragment supplies the number while a different fragment supplies the fee
    # phrase, the rule is a false positive.
    positive_component_sources = {
        _dedup_repeated_phrases(c.source_text) if c.source_text else None
        for c in fee_components
        if c.type in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge"}
    }
    if positive_component_sources:
        value_from_marketing = all(_is_market_share_text(src) for src in positive_component_sources if src)
        fee_phrase_sources = [
            _dedup_repeated_phrases(e.source_text)
            for e in [item["entry"] for item in group]
            if _is_explicit_fee_phrase(_dedup_repeated_phrases(e.source_text))
            and not _is_market_share_text(_dedup_repeated_phrases(e.source_text))
        ]
        if value_from_marketing and fee_phrase_sources:
            return FeeEvidence(
                type="cross_fragment_fee_evidence",
                source_entry_ids=entry_ids,
                phrases=phrases,
                confidence=0.0,
            )

    # 3. Terminal hardware purchase prices.
    if _is_terminal_hardware_price(base_entry, product_id, fee_components, unit):
        return FeeEvidence(
            type="hardware_price",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 4. Amounts extracted from alphanumeric method/product names.
    if _is_amount_from_method_name(base_entry, fee_components):
        return FeeEvidence(
            type="alphanumeric_method_name",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 5. Positive fee evidence paired with included/free evidence in the same
    #    logical row is contradictory; the included/free statement wins.
    if _has_contradictory_fee_evidence(fee_components):
        return FeeEvidence(
            type="contradictory_fee_evidence",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.0,
        )

    # 6. A real fee needs explicit fee language, a trusted table heading with a
    #    fee formula, or an unconditional per-event/per-period phrase.
    if _is_explicit_fee_phrase(text_for_evidence):
        return FeeEvidence(
            type="explicit_fee_phrase",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.85,
        )

    # Trust a heading/section_path or the source text itself when it already
    # contains a percentage or amount next to a known payment method token.
    amount_patterns = r"(?:\d\s*%|[A-Za-z]?[€£$¥₹₩]\s*\d|\d\s*[€£$¥₹₩])"
    method_pattern_tokens = sorted(set(_PAYMENT_METHOD_TOKENS) | set(_CARD_NETWORK_TOKENS), key=len, reverse=True)
    method_pattern = r"\b(" + "|".join(re.escape(m.replace("_", " ")) for m in method_pattern_tokens) + r")\b"
    text_or_path = (
        " ".join(p.lower() for p in base_entry.section_path if p)
        + " "
        + base_entry.source_text.lower()
        + " "
        + (base_entry.source_evidence or "").lower()
    ).strip()
    if re.search(amount_patterns, text_or_path) and re.search(method_pattern, text_or_path):
        return FeeEvidence(
            type="pricing_table_value",
            source_entry_ids=entry_ids,
            phrases=phrases,
            confidence=0.75,
        )

    return FeeEvidence(
        type="insufficient",
        source_entry_ids=entry_ids,
        phrases=phrases,
        confidence=0.0,
    )


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
    card_context = "card" in lower or "tier" in lower or "scheme" in lower
    if "premium" in lower and card_context:
        return "premium"
    if "standard" in lower and card_context:
        return "standard"
    return None


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


# Common country-name to ISO-3166-1 alpha-2 mapping used when LPM rows list
# customer countries (e.g. "Klarna United States, Canada ...").
_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "united states": "US",
    "usa": "US",
    "canada": "CA",
    "austria": "AT",
    "belgium": "BE",
    "germany": "DE",
    "netherlands": "NL",
    "switzerland": "CH",
    "czech republic": "CZ",
    "czechia": "CZ",
    "denmark": "DK",
    "finland": "FI",
    "france": "FR",
    "greece": "GR",
    "ireland": "IE",
    "italy": "IT",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "spain": "ES",
    "sweden": "SE",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "australia": "AU",
    "new zealand": "NZ",
}


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
        if "instant bank payments" in lower:
            return "instant_bank_payments"
        if "for klarna" in lower:
            return "klarna"
    # Affirm variants: "Affirm Standard ...", "Affirm Enhanced ..."
    if "affirm standard" in lower:
        return "standard"
    if "affirm enhanced" in lower:
        return "enhanced"
    return None


def _infer_pricing_tier(text: str) -> str | None:
    """Detect standard/enhanced/premium tier qualifiers."""
    lower = text.lower()
    if "enhanced" in lower:
        return "enhanced"
    if "premium" in lower:
        return "premium"
    if "standard" in lower:
        return "standard"
    return None


def _infer_contract_length(text: str) -> str | None:
    """Distinguish monthly vs annual subscription commitments."""
    lower = text.lower()
    if re.search(r"\b1[-\s]year\s+contract\b|\bone[-\s]year\s+contract\b", lower):
        return "1_year"
    if "per month" in lower or "monthly" in lower:
        return "month_to_month"
    return None


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


def _nearest_heading_line(entry: PricingEntry) -> str | None:
    """Return the nearest preceding heading from row-level evidence.

    Many Stripe tables emit the price as a separate text node and put the
    feature name (e.g. "Visa resolution") in the surrounding container.  We
    look for the line immediately before the first price-containing line
    that is a short, non-price label.
    """
    evidence = entry.source_evidence or ""
    if not evidence:
        return None
    # Look for the heading immediately before the entry's first price token.
    first_number_match = re.search(r"\d+(?:\.\d+)?", entry.source_text)
    if not first_number_match:
        return None
    needle = first_number_match.group(0)
    lines = evidence.splitlines()
    heading: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if needle in stripped and _looks_like_price_line(stripped.lower()):
            return heading
        if not _looks_like_price_line(stripped.lower()) and len(stripped.split()) <= 8:
            heading = stripped
    return heading


_Per_UNIT_NOUNS: frozenset[str] = frozenset(
    {
        "resolution",
        "block",
        "lookup",
        "lookups",
        "deflection",
        "dispute",
        "refund",
        "transaction",
        "charge",
        "payout",
        "account",
        "user",
        "invoice",
        "verification",
    }
)


def _per_completion_noun(entry: PricingEntry) -> str | None:
    """Return a unit noun that completes a dangling 'per' in the source text."""
    source = entry.source_text.lower().rstrip()
    if not source.endswith(" per"):
        return None
    heading = _nearest_heading_line(entry)
    if not heading:
        return None
    words = heading.lower().split()
    # Prefer the last word of the heading (e.g. "Visa resolution" -> "resolution").
    for word in reversed(words):
        clean = re.sub(r"[^a-z0-9]", "", word)
        if clean in _Per_UNIT_NOUNS:
            return clean
    return None


def _is_unsupported_multi_per_shape(entry: PricingEntry, unit: str | None) -> bool:
    """Detect fee shapes with stacked per-unit dimensions that cannot be modelled."""
    text = entry.source_text.lower()
    # Capture up to three words after each "per" to distinguish repeated units
    # ("per successful charge ... per successful charge") from stacked
    # dimensions ("per institution per account holder per month").
    per_phrases = re.findall(r"\bper\b\s+((?:(?!\bper\b)[a-z0-9]+\s+){0,2}(?!\bper\b)[a-z0-9]+)", text)
    per_phrases = [p.strip() for p in per_phrases]
    if len(per_phrases) >= 3 and len(set(per_phrases)) > 1:
        return True
    if len(per_phrases) == 2 and len(set(per_phrases)) > 1:
        # Two distinct per-units are only supported when one is a recognised
        # time/fee unit (e.g. "per active user per month").
        time_units = {"month", "year", "day", "transaction", "charge", "dispute", "refund", "payout", "invoice"}
        if not any(any(t in p.split() for t in time_units) for p in per_phrases):
            return True
    return False


def _looks_like_price_line(line: str) -> bool:
    """Return True when a line is a price fragment rather than a heading."""
    if not line:
        return True
    if line in {"per", "month", "year", "transaction", "transactions", "learn more", "compare plans"}:
        return True
    stripped = line.strip()
    if re.fullmatch(r"[\d\s,.]+", stripped):
        return True
    # A heading is not just a currency code/symbol and/or an amount.
    currency_parts = sorted(set(CURRENCY_CODES) | set(CURRENCY_SYMBOLS), key=len, reverse=True)
    currency_pattern = "|".join(re.escape(c) for c in currency_parts)
    price_pattern = rf"(?i)^\s*(?:{currency_pattern})?\s*[\d\s,.]*\s*(?:{currency_pattern})?\s*(?:per)?\s*$"
    return bool(re.fullmatch(price_pattern, stripped))


def _heading_to_snake_case(heading: str) -> str:
    """Normalize a heading like 'Visa resolution' to 'visa_resolution'."""
    heading = re.sub(r"[^\w\s]", "", heading)
    return "_".join(heading.lower().split())


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


def _earliest_payment_method(text: str, max_word_index: int | None = None) -> str | None:
    """Return the earliest payment-method token in ``text``.

    When ``max_word_index`` is given, only consider matches whose first word
    starts at or before that word position.  This lets LPM card headings such
    as "Link 2.9% ..." win over later qualifiers like "for Klarna".
    """
    text_lower = text.lower()
    words = text_lower.split()
    head = " ".join(words[:6])
    best: tuple[int, int, str] | None = None
    for method in _PAYMENT_METHOD_TOKENS:
        display = method.replace("_", " ")
        for match in re.finditer(rf"\b{re.escape(display)}s?\b", head if max_word_index is not None else text_lower):
            word_index = len(head[: match.start()].split()) if max_word_index is not None else 0
            if max_word_index is not None and word_index > max_word_index:
                continue
            if best is None or (word_index, match.start()) < (best[0], best[1]):
                best = (word_index, match.start(), method)
    return best[2] if best else None


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
    # Headings that name a specific payment method are handled by method detection.
    for method in _PAYMENT_METHOD_TOKENS:
        if method in {"card", "terminal"}:
            continue
        if method.replace("_", " ") in lower:
            return None

    heading_products = [
        ("authorization boost", "authorization_boost"),
        ("authorisation boost", "authorization_boost"),
        ("smart dispute", "smart_disputes"),
        ("smart disputes", "smart_disputes"),
        ("dispute", "disputes"),
        ("chargeback", "disputes"),
        ("refund", "refunds"),
        ("3d secure", "three_d_secure"),
        ("3-d secure", "three_d_secure"),
        ("customer authentication", "three_d_secure"),
        ("instant payout", "instant_payouts"),
        ("instant pay", "instant_payouts"),
        ("custom domain", "custom_domain"),
        ("post-payment invoice", "post_payment_invoice"),
        ("post payment invoice", "post_payment_invoice"),
        ("adaptive pricing", "adaptive_pricing"),
        ("foreign exchange", "adaptive_pricing"),
        ("fx", "adaptive_pricing"),
        ("invoic", "invoicing"),
        ("subscription", "subscriptions"),
        ("recurring billing", "subscriptions"),
        ("stablecoin", "stablecoin_payments"),
        ("managed payments", "managed_payments"),
        ("radar", "radar"),
        ("platform", "platform"),
        ("marketplace", "platform"),
        ("connect", "connect"),
        ("identity", "identity"),
        ("sigma", "sigma"),
        ("billing", "billing"),
        ("payout", "payouts"),
        ("tax", "tax"),
    ]
    for keyword, product in heading_products:
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
            if _infer_channel(entry) == "in_person" or _text_has(
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

    if _text_has(combined, "smart dispute", "smart disputes"):
        return "smart_disputes"
    if _text_has(combined, "authorization boost", "authorisation boost"):
        return "authorization_boost"
    if _text_has(combined, "dispute", "chargeback"):
        return "disputes"
    if _text_has(combined, "refund"):
        return "refunds"
    if _text_has(combined, "3d secure", "3-d secure", "customer authentication"):
        return "three_d_secure"
    if _text_has(combined, "instant payout", "instant pay"):
        return "instant_payouts"
    if _text_has(combined, "custom domain"):
        return "custom_domain"
    if _text_has(combined, "post-payment invoice", "post payment invoice"):
        return "post_payment_invoice"
    if _text_has(combined, "adaptive pricing", "adaptive acceptance", "uplift") or _text_has(
        combined, "foreign exchange", "fx", "converted amount"
    ):
        return "adaptive_pricing"
    if _text_has(combined, "invoic"):
        return "invoicing"
    if _text_has(combined, "subscription", "recurring billing"):
        return "subscriptions"
    if _text_has(combined, "stablecoin"):
        return "stablecoin_payments"
    if _text_has(combined, "managed payments"):
        return "managed_payments"
    if _text_has(combined, "radar"):
        return "radar"
    if _text_has(combined, "platform", "marketplace", "connect"):
        return "platform"
    if _text_has(combined, "identity verification", "identity"):
        return "identity"
    if _text_has(combined, "sigma"):
        return "sigma"
    if _text_has(combined, "billing"):
        return "billing"
    if _text_has(combined, "payout") and not _text_has(combined, "instant payout"):
        return "payouts"
    if _text_has(combined, "tax"):
        return "tax"
    if _text_has(combined, "terminal", "tap to pay", "in-person", "in person", "reader"):
        return "terminal"

    text = entry.source_text.lower()
    combined = path + " " + text
    # Generic payment processing is card-based; avoid treating generic
    # "payment methods" headings as card payments when no card or explicit
    # method token is present.
    if "card" in combined or ("payment" in combined and not _text_has(combined, "payment methods", "payment method")):
        return "payments"
    # Surcharges for regional payment-method groups (e.g. "South Korean payment
    # methods + 1.5% for international transactions") are payment fees even though
    # no individual method token is present.
    if _has_base_fee(entry) and "payment methods" in combined:
        return "payments"
    if _text_has(combined, "ach"):
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
    if "custom payments pricing" in combined or "custom pricing" in combined:
        return "custom"
    if "standard payments pricing" in combined or "standard pricing" in combined:
        return "standard"
    return None


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
    if _text_has(lower, "contact sales", "custom quote"):
        return "custom"
    if _text_has(lower, "starting at", "starting from", "starts at", "starts from"):
        return "from"
    if re.search(r"\b(?:capped at|cap at|cap|maximum|max)\b", lower):
        return "range"
    if "up to" in lower and not re.search(r"up to \d+\s?(month|months|day|days|week|weeks|year|years)", lower):
        return "range"
    if re.search(r"\b(?:minimum|min)\b", lower):
        return "from"
    if _text_has(lower, "included", "no additional fee"):
        return "included"
    if "free" in lower or "no fee" in lower:
        return "free"
    # Package/allotment pricing with overages cannot be represented as a
    # simple per-event fee, so keep it as a custom-quote row.
    if _text_has(lower, "overages", "overage", "allotment", "volume pricing", "custom package"):
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
    if "standard settlement" in text:
        conditions.append(FeeCondition(dimension="settlement_timing", value="standard"))
    if "instant settlement" in text or "instant payout" in text:
        conditions.append(FeeCondition(dimension="settlement_timing", value="instant"))
    if "two-day settlement" in text or "two day settlement" in text:
        conditions.append(FeeCondition(dimension="settlement_timing", value="two_day"))
    if "same-day settlement" in text or "same day settlement" in text:
        conditions.append(FeeCondition(dimension="settlement_timing", value="same_day"))

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

    if product_id in {"smart_disputes", "disputes"} and _text_has(combined, "smart dispute", "smart disputes"):
        conditions.append(FeeCondition(dimension="feature_enabled", value="smart_disputes"))

    if product_id == "adaptive_pricing" or _text_has(combined, "adaptive pricing"):
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

    if "authorisation" in text or "authorization" in text:
        conditions.append(FeeCondition(dimension="transaction_type", value="authorization"))
    elif "per successful charge" in text or "per charge" in text:
        conditions.append(FeeCondition(dimension="transaction_type", value="charge"))
    elif "per payout" in text or product_id == "instant_payouts":
        conditions.append(FeeCondition(dimension="transaction_type", value="payout"))
    elif "bank account validation" in text:
        conditions.append(FeeCondition(dimension="transaction_type", value="bank_account_validation"))

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
    if "per month" in text or "monthly" in text:
        return "monthly"
    if "per year" in text or "yearly" in text:
        return "yearly"
    if "per invoice" in text:
        return "per_invoice"
    if "per payout" in text:
        return "per_payout"
    if "per refund" in text:
        return "per_refund"
    if "per verification" in text:
        return "per_verification"
    if "per active user" in text or "per user" in text:
        return "per_active_user"
    if "per connected account" in text:
        return "per_connected_account"
    if re.search(r"per\s+calculation\s+api\s+call|per\s+api\s+call|api\s+call", text):
        return "per_api_call"
    if "per screened transaction" in text or "per screened payment" in text:
        return "per_screened_transaction"
    if (
        "per authorisation" in text
        or "per authorization" in text
        or ("per" in text and re.search(r"\battempt\b", text))
    ):
        return "per_attempt"
    if "per successful charge" in text or "per charge" in text:
        return "per_charge"
    if re.search(r"\bwire\s+(?:payment|transfer)\b", text):
        return "per_charge"
    if "per successful transaction" in text or "per transaction" in text:
        return "per_transaction"
    if "per successful" in text and "transaction" in text:
        return "per_transaction"
    if "per successful" in text and "charge" in text:
        return "per_charge"
    if "per paid invoice" in text or "per invoice" in text:
        return "per_invoice"
    # Default unit for entries with a recognizable payment method or product.
    method = _infer_payment_method(entry)
    if method:
        return "per_transaction"
    if product_id in {
        "ach_direct_debit",
        "sepa_direct_debit",
        "bacs_direct_debit",
    }:
        return "per_transaction"
    # Fallback unit for common product families when the phrase omits one.
    product_unit_defaults: dict[str, str] = {
        "payments": "per_transaction",
        "instant_payouts": "per_payout",
        "payouts": "per_payout",
        "adaptive_pricing": "per_transaction",
        "post_payment_invoice": "per_invoice",
        "invoicing": "per_invoice",
        "tax": "per_transaction",
        "managed_payments": "per_transaction",
        "three_d_secure": "per_transaction",
        "radar": "per_transaction",
        "smart_disputes": "per_dispute",
        "ach_direct_debit": "per_transaction",
        "identity": "per_verification",
        "billing": "per_transaction",
        "subscriptions": "per_transaction",
        "platform": "per_transaction",
        "connect": "per_transaction",
        "stablecoin_payments": "per_transaction",
        "refunds": "per_refund",
        "sigma": "per_charge",
    }
    if product_id in product_unit_defaults:
        return product_unit_defaults[product_id]
    if product_id in {
        "ideal",
        "wero",
        "bancontact",
        "eps",
        "blik",
        "przelewy24",
        "swish",
        "twint",
        "pay_by_bank",
        "mb_way",
        "pix",
        "upi",
        "bizum",
        "alipay",
        "mobilepay",
        "paypal",
        "revolut_pay",
        "wechat_pay",
        "amazon_pay",
        "satispay",
        "konbini",
        "link",
        "klarna",
        "billie",
        "scalapay",
        "multibanco",
    }:
        return "per_transaction"
    return None


def _is_tap_to_pay(entry: PricingEntry) -> bool:
    return "tap to pay" in entry.source_text.lower() or entry.payment_method == "tap_to_pay"


def _dedup_repeated_phrases(text: str) -> str:
    """Remove consecutive duplicate word n-grams while preserving order.

    This collapses artefacts such as "per successful charge per successful charge"
    or "for domestic cards for domestic cards" that arise when a qualifier is
    repeated across merged fragments.
    """
    words = text.split()
    result: list[str] = []
    i = 0
    max_n = min(8, len(words) // 2) if len(words) > 4 else 3
    while i < len(words):
        duplicated = False
        for n in range(max_n, 1, -1):
            if i + n <= len(words) and result[-n:] == words[i : i + n]:
                i += n
                duplicated = True
                break
        if not duplicated:
            result.append(words[i])
            i += 1
    return " ".join(result)


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _has_base_fee(entry: PricingEntry, parsed: dict[str, Any] | None = None) -> bool:
    if parsed is None:
        parsed = parse_fee_value(entry.source_text)
    return bool(parsed.get("percentage") or parsed.get("fixed_amount"))


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


def _is_modifier_entry(entry: PricingEntry) -> bool:
    """Return True when an entry is a continuation of the previous base fee."""
    parsed = parse_fee_value(entry.source_text)
    hint = _entry_component_hint(entry)
    has_base = _has_base_fee(entry, parsed)

    if has_base:
        # A standalone base row can still contain a cap/max (e.g. "0.8% ... $5 cap").
        # It is not a modifier unless it only describes the modifier.
        return hint in {"maximum", "minimum"} and not any(t.kind == "percentage" for t in entry.tokens)
    # Modifiers are non-base lines that qualify an earlier fee: caps, minimums,
    # or included/free statements.  Surcharge fragments beginning with '+' and
    # standalone method variants (e.g. Tap to Pay) are base rules in their own
    # right.
    return hint in {"maximum", "minimum", "included", "free"} and not _is_tap_to_pay(entry)


def _amount_component_type(hint: str, text: str) -> str:
    lower = text.lower()
    if hint == "surcharge":
        return "fixed_surcharge"
    if hint in {"maximum"} or _text_has(lower, "cap", "capped", "maximum", "max"):
        return "maximum_fee"
    if hint in {"minimum"} or _text_has(lower, "minimum", "min"):
        return "minimum_fee"
    if hint == "included":
        return "included"
    if hint == "free":
        return "free"
    if hint == "custom":
        return "custom_pricing"
    return "fixed_amount"


def _build_components_for_entry(entry: PricingEntry, hint: str) -> list[FeeComponent]:
    """Convert an entry's tokens into typed fee components."""
    from .pricing_tokens import tokenize_fee_text

    components: list[FeeComponent] = []
    tokens = entry.tokens or tokenize_fee_text(entry.source_text)
    if not tokens:
        if hint == "included":
            components.append(
                FeeComponent(type="included", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        elif hint == "free":
            components.append(FeeComponent(type="free", source_entry_id=entry.entry_id, source_text=entry.source_text))
        elif hint == "custom":
            components.append(
                FeeComponent(type="custom_pricing", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        else:
            components.append(
                FeeComponent(type="non_calculable", source_entry_id=entry.entry_id, source_text=entry.source_text)
            )
        return components

    for token in tokens:
        if token.kind == "percentage":
            if hint == "surcharge":
                comp_type = "percentage_surcharge"
            elif hint == "from":
                comp_type = "percentage"
            else:
                comp_type = "percentage"
            components.append(
                FeeComponent(
                    type=comp_type,
                    value=token.percentage,
                    basis_points=token.basis_points,
                    operator=token.operator,
                    source_entry_id=entry.entry_id,
                    source_text=entry.source_text,
                )
            )
        elif token.kind == "amount":
            comp_type = _amount_component_type(hint, entry.source_text)
            minor = _fixed_amount_minor(token.amount, token.currency) if token.amount and token.currency else None
            components.append(
                FeeComponent(
                    type=comp_type,
                    amount=token.amount,
                    currency=token.currency,
                    minor_amount=minor,
                    operator=token.operator,
                    source_entry_id=entry.entry_id,
                    source_text=entry.source_text,
                )
            )
    return components


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

    # Determine behavior based on fee shape and wording.
    has_surcharge = any(c.type in {"percentage_surcharge", "fixed_surcharge"} for c in fee_components)
    has_alternative = bool(re.search(r"\b(or|alternative|instead of)\b", label.lower()))
    if has_surcharge:
        behavior = "additive"
    elif has_alternative:
        behavior = "alternative"
    elif any(c.type in {"included", "free"} for c in fee_components) or exactness == "custom":
        behavior = "informational"
    else:
        behavior = "conditional"

    # Default missing channel/unit for well-understood products.
    if not channel:
        if (
            product_id == "terminal"
            or payment_method == "tap_to_pay"
            or _text_has(base_entry.source_text.lower(), "in-person", "in person", "tap to pay")
        ):
            channel = "in_person"
        elif (
            payment_method
            or product_id
            in {
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
            or _text_has(base_entry.source_text.lower(), "card", "payment")
        ):
            channel = "online"
    if not unit:
        if product_id == "disputes":
            unit = "per_dispute"
        elif payment_method:
            unit = "per_transaction"

    # Determine calculability and final status using positive fee evidence.
    calculable = _has_base_fee(base_entry) or any(
        c.type in {"percentage", "fixed_amount", "percentage_surcharge", "fixed_surcharge"} for c in fee_components
    )

    fee_evidence = _fee_evidence_for_group(group, product_id, fee_components, unit)
    evidence_positive = fee_evidence.type in {
        "explicit_fee_phrase",
        "pricing_table_value",
        "structured_fee_field",
    }

    if exactness == "custom" or (not calculable and exactness == "from"):
        classification_status = CUSTOM_PRICING
    elif exactness in {"included", "free"}:
        classification_status = INCLUDED if exactness == "included" else FREE
    elif calculable and evidence_positive:
        classification_status = CALCULABLE_RULE
    else:
        classification_status = NON_CALCULABLE

    # Evidence-driven downgrades override the numeric calculability decision.
    if fee_evidence.type == "promotional_language":
        classification_status = CUSTOM_PRICING
    elif fee_evidence.type in {"marketing_prose", "hardware_price", "alphanumeric_method_name"}:
        classification_status = INFORMATIONAL
    elif fee_evidence.type in {"contradictory_fee_evidence", "cross_fragment_fee_evidence"} or (
        fee_evidence.type == "insufficient" and classification_status == CALCULABLE_RULE
    ):
        classification_status = NON_CALCULABLE

    # A rule must have a channel, unit, and behavior to be calculation-ready.
    if classification_status == CALCULABLE_RULE and (not channel or not unit or not behavior):
        classification_status = NON_CALCULABLE

    # Stacked per-unit dimensions on an otherwise unrecognised product (e.g.
    # "per institution per account holder per month") cannot be modelled
    # deterministically.
    if product_id == "unspecified" and _is_unsupported_multi_per_shape(base_entry, unit):
        classification_status = UNSUPPORTED_SHAPE

    confidence = fee_evidence.confidence
    if classification_status == CALCULABLE_RULE and (not channel or not unit or not behavior):
        confidence = 0.6
    if classification_status != CALCULABLE_RULE:
        confidence = 0.0
    if classification_status == CALCULABLE_RULE and confidence < _PUBLICATION_CONFIDENCE_THRESHOLD:
        classification_status = NON_CALCULABLE

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
    summary = CoverageSummary()
    summary = summary.model_copy(update={"source_entries": len(entries)})

    numeric_entries = _numeric_source_entries(entries)
    referenced_ids: set[str] = set()
    for rule in rules:
        for eid in rule.contributing_entry_ids:
            referenced_ids.add(eid)
    for entry in unclassified:
        referenced_ids.add(entry.entry_id)
    referenced_numeric = [e for e in numeric_entries if e.entry_id in referenced_ids]
    dropped_numeric = len(numeric_entries) - len(referenced_numeric)

    summary = summary.model_copy(
        update={
            "numeric_source_entries": len(numeric_entries),
            "referenced_numeric_entries": len(referenced_numeric),
            "dropped_numeric_entries": dropped_numeric,
        }
    )

    for rule in rules:
        if rule.classification_status == CALCULABLE_RULE:
            summary = summary.model_copy(update={"calculable_rules": summary.calculable_rules + 1})
        elif rule.classification_status == NON_CALCULABLE:
            summary = summary.model_copy(update={"non_calculable_rules": summary.non_calculable_rules + 1})
        elif rule.classification_status == "conflict":
            summary = summary.model_copy(
                update={"conflicting_rule_identities": summary.conflicting_rule_identities + 1}
            )
            if _is_fee_candidate(rule):
                summary = summary.model_copy(update={"blocking_fee_conflicts": summary.blocking_fee_conflicts + 1})
            else:
                summary = summary.model_copy(update={"informational_conflicts": summary.informational_conflicts + 1})
        elif rule.classification_status == INCLUDED:
            summary = summary.model_copy(update={"included": summary.included + 1})
        elif rule.classification_status == FREE:
            summary = summary.model_copy(update={"free": summary.free + 1})
        elif rule.classification_status == INFORMATIONAL:
            summary = summary.model_copy(update={"informational": summary.informational + 1})
        elif rule.classification_status == CUSTOM_PRICING:
            summary = summary.model_copy(update={"custom_pricing": summary.custom_pricing + 1})
        elif rule.classification_status == UNSUPPORTED_SHAPE:
            summary = summary.model_copy(update={"unsupported_fee_shapes": summary.unsupported_fee_shapes + 1})
    for entry in unclassified:
        status = entry.classification_status
        if status == UNCLASSIFIED_CANDIDATE:
            summary = summary.model_copy(
                update={"unclassified_fee_candidates": summary.unclassified_fee_candidates + 1}
            )
        if status == AMBIGUOUS:
            summary = summary.model_copy(update={"ambiguous_entries": summary.ambiguous_entries + 1})
        if status == UNSUPPORTED_SHAPE:
            summary = summary.model_copy(update={"unsupported_fee_shapes": summary.unsupported_fee_shapes + 1})
        if status == IGNORED_NON_FEE:
            summary = summary.model_copy(update={"ignored_non_fee": summary.ignored_non_fee + 1})
        if status == REFERENCE_ONLY:
            summary = summary.model_copy(update={"reference_only": summary.reference_only + 1})
        if status == INCLUDED:
            summary = summary.model_copy(update={"included": summary.included + 1})
        if status == FREE:
            summary = summary.model_copy(update={"free": summary.free + 1})
        if status == INFORMATIONAL:
            summary = summary.model_copy(update={"informational": summary.informational + 1})
        if status == CUSTOM_PRICING:
            summary = summary.model_copy(update={"custom_pricing": summary.custom_pricing + 1})
    numeric_candidates = [
        e
        for e in unclassified
        if _has_base_fee(e)
        and e.classification_status
        not in {IGNORED_NON_FEE, INFORMATIONAL, REFERENCE_ONLY, CUSTOM_PRICING, INCLUDED, FREE, UNSUPPORTED_SHAPE}
    ]
    summary = summary.model_copy(update={"numeric_fee_candidates": len(numeric_candidates)})
    return summary


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


def _account_country_from_url(url: str) -> str | None:
    """Infer the account country from a Stripe pricing URL path."""
    import re as _re

    match = _re.search(r"/(?:([a-z]{2})-([a-z]{2})|([a-z]{2}))/pricing", url.lower())
    if match:
        return (match.group(2) or match.group(3) or "").upper()
    if "/pricing" in url.lower() and "stripe.com/pricing" in url.lower():
        return "US"
    return None


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
