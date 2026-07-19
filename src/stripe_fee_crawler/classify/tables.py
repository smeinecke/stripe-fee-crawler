"""Classification constants and lookup tables."""

from __future__ import annotations

import logging
import re
from decimal import Decimal
from typing import Any

from ..payment_methods import _PAYMENT_METHOD_TOKENS

logger = logging.getLogger(__name__)
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

_PRODUCT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("authorization_boost", ("authorization boost", "authorisation boost")),
    ("smart_disputes", ("smart dispute", "smart disputes")),
    ("disputes", ("dispute", "chargeback")),
    ("refunds", ("refund",)),
    ("three_d_secure", ("3d secure", "3-d secure", "customer authentication")),
    ("instant_payouts", ("instant payout", "instant pay")),
    ("custom_domain", ("custom domain",)),
    ("post_payment_invoice", ("post-payment invoice", "post payment invoice")),
    (
        "adaptive_pricing",
        ("adaptive pricing", "adaptive acceptance", "uplift", "foreign exchange", "fx", "converted amount"),
    ),
    ("invoicing", ("invoic",)),
    ("subscriptions", ("subscription", "recurring billing")),
    ("stablecoin_payments", ("stablecoin",)),
    ("managed_payments", ("managed payments",)),
    ("radar", ("radar",)),
    ("platform", ("platform", "marketplace", "connect")),
    ("identity", ("identity verification", "identity")),
    ("sigma", ("sigma",)),
    ("billing", ("billing",)),
    ("payouts", ("payout",)),
    ("tax", ("tax",)),
    ("terminal", ("terminal", "tap to pay", "in-person", "in person", "reader")),
)

_PUBLICATION_CONFIDENCE_THRESHOLD = 0.7

_MIN_HARDWARE_MAJOR_AMOUNT = Decimal("10.0")

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


_CARD_TIER_TABLE: tuple[tuple[Any, str], ...] = (
    ("premium", "premium"),
    ("standard", "standard"),
)

_PAYMENT_METHOD_VARIANT_TABLE: tuple[tuple[Any, str], ...] = (
    ("instant bank payments", "instant_bank_payments"),
    ("for klarna", "klarna"),
    ("affirm standard", "standard"),
    ("affirm enhanced", "enhanced"),
)

_PRICING_TIER_TABLE: tuple[tuple[Any, str], ...] = (
    ("enhanced", "enhanced"),
    ("premium", "premium"),
    ("standard", "standard"),
)

_CONTRACT_LENGTH_TABLE: tuple[tuple[Any, str], ...] = (
    (re.compile(r"\b(?:1[-\s]year|one[-\s]year)\s+contract\b"), "1_year"),
    ("per month", "month_to_month"),
    ("monthly", "month_to_month"),
)

_PRICING_PLAN_TABLE: tuple[tuple[Any, str], ...] = (
    (("custom payments pricing", "custom pricing"), "custom"),
    (("standard payments pricing", "standard pricing"), "standard"),
)

_AMOUNT_COMPONENT_TABLE: tuple[tuple[Any, str], ...] = (
    (("cap", "capped", "maximum", "max"), "maximum_fee"),
    (("minimum", "min"), "minimum_fee"),
)

_UNIT_KEYWORD_TABLE: tuple[tuple[Any, str], ...] = (
    ("per month", "monthly"),
    ("monthly", "monthly"),
    ("per year", "yearly"),
    ("yearly", "yearly"),
    ("per invoice", "per_invoice"),
    ("per payout", "per_payout"),
    ("per refund", "per_refund"),
    ("per verification", "per_verification"),
    ("per active user", "per_active_user"),
    ("per user", "per_active_user"),
    ("per connected account", "per_connected_account"),
    (re.compile(r"(?:per\s+calculation\s+api\s+call|per\s+api\s+call|api\s+call)"), "per_api_call"),
    ("per screened transaction", "per_screened_transaction"),
    ("per screened payment", "per_screened_transaction"),
    (("per authorisation", "per authorization"), "per_attempt"),
    (re.compile(r"\bper\b.*\battempt\b|\battempt\b.*\bper\b"), "per_attempt"),
    ("per successful charge", "per_charge"),
    ("per charge", "per_charge"),
    (re.compile(r"\bwire\s+(?:payment|transfer)\b"), "per_charge"),
    ("per successful transaction", "per_transaction"),
    ("per transaction", "per_transaction"),
    (frozenset({"per successful", "transaction"}), "per_transaction"),
    (frozenset({"per successful", "charge"}), "per_charge"),
    ("per paid invoice", "per_invoice"),
)

_PRODUCT_UNIT_DEFAULTS: dict[str, str] = {
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

_DIRECT_DEBIT_DEFAULT_UNITS: frozenset[str] = frozenset(
    {
        "ach_direct_debit",
        "sepa_direct_debit",
        "bacs_direct_debit",
    }
)

_METHOD_DEFAULT_UNITS: frozenset[str] = frozenset(
    {
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
    }
)

_SETTLEMENT_TIMING_TABLE: tuple[tuple[Any, str], ...] = (
    ("standard settlement", "standard"),
    ("instant settlement", "instant"),
    ("instant payout", "instant"),
    ("two-day settlement", "two_day"),
    ("two day settlement", "two_day"),
    ("same-day settlement", "same_day"),
    ("same day settlement", "same_day"),
)

_TRANSACTION_TYPE_TABLE: tuple[tuple[Any, str], ...] = (
    (("authorisation", "authorization"), "authorization"),
    (("per successful charge", "per charge"), "charge"),
    ("per payout", "payout"),
    ("bank account validation", "bank_account_validation"),
)

__all__ = [
    "CALCULABLE_RULE",
    "NON_CALCULABLE",
    "CUSTOM_PRICING",
    "INCLUDED",
    "FREE",
    "INFORMATIONAL",
    "UNSUPPORTED_SHAPE",
    "UNCLASSIFIED_CANDIDATE",
    "IGNORED_NON_FEE",
    "REFERENCE_ONLY",
    "AMBIGUOUS",
    "_EEA_COUNTRY_CODES",
    "_PAYMENT_METHOD_TOKENS",
    "_CARD_NETWORK_TOKENS",
    "_POSITIVE_FEE_TERMS",
    "_MARKETING_TERMS",
    "_PROMOTIONAL_TERMS",
    "_MARKET_SHARE_STATISTICS",
    "_HARDWARE_PRICE_TERMS",
    "_PRODUCT_KEYWORDS",
    "_PUBLICATION_CONFIDENCE_THRESHOLD",
    "_MIN_HARDWARE_MAJOR_AMOUNT",
    "_COUNTRY_NAME_TO_CODE",
    "_Per_UNIT_NOUNS",
    "_CARD_TIER_TABLE",
    "_PAYMENT_METHOD_VARIANT_TABLE",
    "_PRICING_TIER_TABLE",
    "_CONTRACT_LENGTH_TABLE",
    "_PRICING_PLAN_TABLE",
    "_AMOUNT_COMPONENT_TABLE",
    "_UNIT_KEYWORD_TABLE",
    "_PRODUCT_UNIT_DEFAULTS",
    "_DIRECT_DEBIT_DEFAULT_UNITS",
    "_METHOD_DEFAULT_UNITS",
    "_SETTLEMENT_TIMING_TABLE",
    "_TRANSACTION_TYPE_TABLE",
]
