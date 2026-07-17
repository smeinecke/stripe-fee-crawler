"""Pydantic models for the Stripe fee crawler."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator

from .exceptions import ExitCode


def _require_string(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("value must be a non-empty string")
    return value.strip()


class Language(BaseModel):
    """A supported language for a market."""

    model_config = ConfigDict(frozen=True)

    code: str
    name: str | None = None

    @field_validator("code")
    @classmethod
    def _validate_code(cls, value: str) -> str:
        return _require_string(value).lower()


class Market(BaseModel):
    """A discovered Stripe market.

    Stripe uses locale-style identifiers such as ``en-de`` or ``de-de`` for
    language-country combinations. The ISO 3166-1 alpha-2 account country is
    kept separate from the locale identifier.
    """

    model_config = ConfigDict(frozen=True)

    stripe_market_code: str
    account_country: str
    country_name: str
    region: str | None = None
    locale: str
    languages: list[Language] = Field(default_factory=list)
    url_prefix: str
    preferred_language: str | None = None
    default_currency: str | None = None
    status: str = "discovered"

    @field_validator("account_country")
    @classmethod
    def _validate_account_country(cls, value: str) -> str:
        value = _require_string(value).upper()
        if len(value) != 2 or not value.isalpha():
            raise ValueError(f"account_country must be an ISO 3166-1 alpha-2 code: {value!r}")
        return value

    @field_validator("locale")
    @classmethod
    def _validate_locale(cls, value: str) -> str:
        return _require_string(value)

    @field_validator("stripe_market_code")
    @classmethod
    def _validate_stripe_market_code(cls, value: str) -> str:
        return _require_string(value).lower()

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, value: str) -> str:
        allowed = {"discovered", "supported", "unsupported", "pricing_page_unavailable", "transient_failure"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return value

    @computed_field
    @property
    def url_slug(self) -> str:
        return self.stripe_market_code.lower()


class Source(BaseModel):
    """Source metadata for a crawled fee page."""

    model_config = ConfigDict(frozen=True)

    requested_url: str
    canonical_url: str | None = None
    page_id: str | None = None
    page_title: str | None = None
    page_updated_at: str | None = None
    source_updated_at: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None
    evidence_text: str | None = None


class Link(BaseModel):
    """A hyperlink extracted from source text."""

    model_config = ConfigDict(frozen=True)

    text: str | None = None
    uri: str | None = None


class CacheStats(BaseModel):
    """HTTP cache statistics for a crawl run."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    cache_hits: int = 0
    cache_misses: int = 0
    cache_revalidations: int = 0
    cache_304_responses: int = 0
    cache_writes: int = 0
    cache_errors: int = 0
    bytes_avoided: int = 0


class FeeToken(BaseModel):
    """A normalized pricing token extracted from a fee phrase."""

    model_config = ConfigDict(frozen=True)

    raw: str
    kind: str = "text"
    value: str | None = None
    amount: str | None = None
    currency: str | None = None
    operator: str | None = None
    percentage: str | None = None
    basis_points: str | None = None
    exactness: str | None = None
    token_id: str | None = None
    note: str | None = None
    is_minor_currency: bool = False


class FeeComponent(BaseModel):
    """One calculable or descriptive component of a fee formula."""

    model_config = ConfigDict(frozen=True)

    type: str
    value: str | None = None
    amount: str | None = None
    currency: str | None = None
    minor_amount: str | None = None
    basis_points: str | None = None
    schedule_id: str | None = None
    operator: str | None = None
    source_text: str | None = None
    source_entry_id: str | None = None

    @field_validator("type")
    @classmethod
    def _type_allowed(cls, value: str) -> str:
        allowed = {
            "percentage",
            "fixed_amount",
            "maximum_fee",
            "minimum_fee",
            "percentage_surcharge",
            "fixed_surcharge",
            "included",
            "free",
            "tiered",
            "custom_pricing",
            "non_calculable",
        }
        if value not in allowed:
            raise ValueError(f"fee component type must be one of {allowed}")
        return value


class Section(BaseModel):
    """A normalized page section."""

    model_config = ConfigDict(frozen=True)

    section_id: str | None = None
    heading: str | None = None
    level: int | None = None
    body: str | None = None
    section_path: list[str] = Field(default_factory=list)
    source_order: int = 0
    links: list[Link] = Field(default_factory=list)


class PricingEntry(BaseModel):
    """A single extracted public pricing fact.

    This keeps the source evidence separate from any derived fee rule. A single
    section may yield multiple entries (e.g. domestic vs. international cards).
    """

    model_config = ConfigDict(frozen=True)

    entry_id: str
    product: str | None = None
    product_category: str | None = None
    section_path: list[str] = Field(default_factory=list)
    fee_category: str | None = None
    payment_method_family: str | None = None
    payment_method: str | None = None
    channel: str | None = None
    source_text: str
    qualifiers: list[str] = Field(default_factory=list)
    footnotes: list[str] = Field(default_factory=list)
    source_url: str
    source_evidence: str | None = None
    tokens: list[FeeToken] = Field(default_factory=list)
    links: list[Link] = Field(default_factory=list)
    source_order: int = 0
    context: dict[str, Any] = Field(default_factory=dict)
    classification_status: str = "unclassified"
    confidence: float = 0.0
    classification_evidence: list[str] = Field(default_factory=list)

    @field_validator("classification_status")
    @classmethod
    def _status_allowed(cls, value: str) -> str:
        allowed = {
            "classified",
            "unclassified",
            "non_calculable",
            "partial",
            "calculable_rule",
            "reference_only",
            "included",
            "free",
            "custom_pricing",
            "informational",
            "unsupported_fee_shape",
            "unclassified_fee_candidate",
            "ignored_non_fee",
            "ambiguous",
            "conflict",
        }
        if value not in allowed:
            raise ValueError(f"classification_status must be one of {allowed}")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))


class FeeCondition(BaseModel):
    """A single applicability condition for a fee rule."""

    model_config = ConfigDict(frozen=True)

    dimension: str
    operator: str = "eq"
    value: Any
    evidence: str | None = None


class FeeRule(BaseModel):
    """A derived calculation-ready fee rule.

    Money amounts are represented as exact decimal strings in major units
    together with a currency code. Percentage rates are represented as exact
    decimal strings and as basis points for unambiguous computation.

    A rule is identified by ``product_id + variant_id + conditions``. The
    display ``label`` and localized source text are not part of semantic
    identity.
    """

    model_config = ConfigDict(frozen=True)

    rule_id: str
    entry_id: str | None = None
    contributing_entry_ids: list[str] = Field(default_factory=list)
    product_id: str | None = None
    variant_id: str | None = None
    label: str | None = None
    name: str | None = None
    provider: str = "stripe"
    account_country: str | None = None
    channel: str | None = None
    payment_method: str | None = None
    card_origin: str | None = None
    card_region: str | None = None
    card_tier: str | None = None
    customer_country: str | None = None
    presentment_currency: str | None = None
    settlement_currency: str | None = None
    currency_conversion_required: bool | None = None
    transaction_amount_min: str | None = None
    transaction_amount_max: str | None = None
    billing_type: str | None = None
    recurring: bool | None = None
    payout_type: str | None = None
    dispute_state: str | None = None
    percentage: str | None = None
    basis_points: str | None = None
    fixed_amount: str | None = None
    fixed_amount_minor: str | None = None
    fixed_currency: str | None = None
    minimum_amount: str | None = None
    maximum_amount: str | None = None
    unit: str = "per_transaction"
    exactness: str = "exact"
    behavior: str = "additive"
    conditions: list[FeeCondition] = Field(default_factory=list)
    additional_fees: list[str] = Field(default_factory=list)
    fee_components: list[FeeComponent] = Field(default_factory=list)
    source_text: str | None = None
    source_texts: list[str] = Field(default_factory=list)
    source_url: str | None = None
    source_fragments: list[dict[str, Any]] = Field(default_factory=list)
    classification_status: str = "unclassified"
    confidence: float = 0.0
    classification_evidence: list[str] = Field(default_factory=list)

    @field_validator("exactness")
    @classmethod
    def _exactness_allowed(cls, value: str) -> str:
        allowed = {"exact", "from", "range", "tiered", "included", "free", "custom", "non_calculable"}
        if value not in allowed:
            raise ValueError(f"exactness must be one of {allowed}")
        return value

    @field_validator("unit")
    @classmethod
    def _unit_allowed(cls, value: str) -> str:
        allowed = {
            "per_transaction",
            "per_attempt",
            "per_dispute",
            "per_payout",
            "monthly",
            "yearly",
            "per_invoice",
            "per_charge",
            "informational",
        }
        if value not in allowed:
            raise ValueError(f"unit must be one of {allowed}")
        return value

    @field_validator("behavior")
    @classmethod
    def _behavior_allowed(cls, value: str) -> str:
        allowed = {"additive", "alternative", "mutually_exclusive", "conditional", "informational"}
        if value not in allowed:
            raise ValueError(f"behavior must be one of {allowed}")
        return value

    @field_validator("classification_status")
    @classmethod
    def _classification_status_allowed(cls, value: str) -> str:
        allowed = {
            "classified",
            "unclassified",
            "non_calculable",
            "partial",
            "calculable_rule",
            "reference_only",
            "included",
            "free",
            "custom_pricing",
            "informational",
            "unsupported_fee_shape",
            "unclassified_fee_candidate",
            "ignored_non_fee",
            "ambiguous",
            "conflict",
        }
        if value not in allowed:
            raise ValueError(f"classification_status must be one of {allowed}")
        return value

    @field_validator("confidence")
    @classmethod
    def _confidence_range(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @model_validator(mode="after")
    def _sync_legacy_fields(self) -> FeeRule:
        """Keep legacy flat fields in sync with fee_components when possible."""
        if not self.fee_components:
            return self
        pct = next((c for c in self.fee_components if c.type == "percentage"), None)
        fixed = next((c for c in self.fee_components if c.type == "fixed_amount"), None)
        max_fee = next((c for c in self.fee_components if c.type == "maximum_fee"), None)
        min_fee = next((c for c in self.fee_components if c.type == "minimum_fee"), None)
        updates: dict[str, Any] = {}
        if pct and not self.percentage:
            updates["percentage"] = pct.value
        if pct and not self.basis_points:
            updates["basis_points"] = pct.basis_points
        if fixed and not self.fixed_amount:
            updates["fixed_amount"] = fixed.amount
        if fixed and not self.fixed_currency:
            updates["fixed_currency"] = fixed.currency
        if max_fee and not self.maximum_amount:
            updates["maximum_amount"] = max_fee.amount
        if min_fee and not self.minimum_amount:
            updates["minimum_amount"] = min_fee.amount
        if updates:
            return self.model_copy(update=updates)
        return self


class ParserWarning(BaseModel):
    """A non-fatal parser warning."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    context: dict[str, Any] | None = None


class CoverageSummary(BaseModel):
    """Summary of how source pricing records were classified."""

    model_config = ConfigDict(frozen=True)

    source_entries: int = 0
    calculable_rules: int = 0
    non_calculable_rules: int = 0
    numeric_fee_candidates: int = 0
    unclassified_fee_candidates: int = 0
    ambiguous_entries: int = 0
    conflicting_rule_identities: int = 0
    unsupported_fee_shapes: int = 0
    ignored_non_fee: int = 0
    reference_only: int = 0
    included: int = 0
    custom_pricing: int = 0


class MarketOutput(BaseModel):
    """Per-market normalized output."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    market: Market
    sources: list[Source] = Field(default_factory=list)
    sections: list[Section] = Field(default_factory=list)
    entries: list[PricingEntry] = Field(default_factory=list)
    derived_rules: list[FeeRule] = Field(default_factory=list)
    unclassified_entries: list[PricingEntry] = Field(default_factory=list)
    warnings: list[ParserWarning] = Field(default_factory=list)
    derivation_status: str = "unclassified"
    calculator_coverage_status: str = "unclassified"
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary)
    transient_failure: bool = False
    unsupported_reason: str | None = None

    @field_validator("derivation_status")
    @classmethod
    def _derivation_status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed"}
        if value not in allowed:
            raise ValueError(f"derivation_status must be one of {allowed}")
        return value

    @field_validator("calculator_coverage_status")
    @classmethod
    def _calculator_coverage_status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed", "not_calculable"}
        if value not in allowed:
            raise ValueError(f"calculator_coverage_status must be one of {allowed}")
        return value


class MarketIndexEntry(BaseModel):
    """Compact entry in the market index."""

    model_config = ConfigDict(frozen=True)

    account_country: str
    stripe_market_code: str
    locale: str
    data_path: str
    source_urls: list[str] = Field(default_factory=list)
    source_updated_at: str | None = None
    derivation_status: str | None = None
    calculator_coverage_status: str | None = None
    content_sha256: str | None = None
    schema_version: int = 1

    @field_validator("account_country")
    @classmethod
    def _validate_account_country(cls, value: str) -> str:
        value = _require_string(value).upper()
        if len(value) != 2 or not value.isalpha():
            raise ValueError(f"Invalid ISO country code: {value!r}")
        return value


class MarketIndex(BaseModel):
    """Index of successfully processed markets."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    markets: list[MarketIndexEntry] = Field(default_factory=list)


class UnsupportedMarket(BaseModel):
    """A market without a discoverable public fee page."""

    model_config = ConfigDict(frozen=True)

    stripe_market_code: str
    account_country: str | None = None
    country_name: str | None = None
    tested_urls: list[str] = Field(default_factory=list)
    reason: str | None = None
    status: str = "unsupported"
    first_confirmed_at: str | None = None
    last_confirmed_at: str | None = None
    last_status: int | None = None
    temporary: bool = False

    @field_validator("status")
    @classmethod
    def _status_allowed(cls, value: str) -> str:
        allowed = {"unsupported", "pricing_page_unavailable", "transient_failure"}
        if value not in allowed:
            raise ValueError(f"status must be one of {allowed}")
        return value


class MarketManifest(BaseModel):
    """Discovered market manifest."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    markets: list[Market] = Field(default_factory=list)
    unsupported: list[UnsupportedMarket] = Field(default_factory=list)
    aliases: dict[str, str] = Field(default_factory=dict)
    fee_page_urls: dict[str, list[str]] = Field(default_factory=dict)
    transient_failures: list[UnsupportedMarket] = Field(default_factory=list)


class CoreFeeRule(BaseModel):
    """A compact, calculator-facing fee rule without provenance."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    product_id: str | None = None
    variant_id: str | None = None
    label: str | None = None
    provider: str = "stripe"
    account_country: str | None = None
    channel: str | None = None
    payment_method: str | None = None
    conditions: list[FeeCondition] = Field(default_factory=list)
    fee_components: list[FeeComponent] = Field(default_factory=list)
    unit: str = "per_transaction"
    behavior: str = "additive"
    classification_status: str = "unclassified"
    exactness: str = "exact"

    @field_validator("classification_status")
    @classmethod
    def _classification_status_allowed(cls, value: str) -> str:
        allowed = {
            "classified",
            "unclassified",
            "non_calculable",
            "partial",
            "calculable_rule",
            "reference_only",
            "included",
            "free",
            "custom_pricing",
            "informational",
            "unsupported_fee_shape",
            "unclassified_fee_candidate",
            "ignored_non_fee",
            "ambiguous",
            "conflict",
        }
        if value not in allowed:
            raise ValueError(f"classification_status must be one of {allowed}")
        return value

    @field_validator("exactness")
    @classmethod
    def _exactness_allowed(cls, value: str) -> str:
        allowed = {"exact", "from", "range", "tiered", "included", "free", "custom", "non_calculable"}
        if value not in allowed:
            raise ValueError(f"exactness must be one of {allowed}")
        return value


class CoreFeeEntry(BaseModel):
    """A single market's confidently derived core fees."""

    model_config = ConfigDict(frozen=True)

    account_country: str
    stripe_market_code: str
    locale: str
    derivation_status: str
    calculator_coverage_status: str = "unclassified"
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary)
    rules: list[CoreFeeRule] = Field(default_factory=list)
    unclassified_count: int = 0

    @field_validator("account_country")
    @classmethod
    def _validate_account_country(cls, value: str) -> str:
        value = _require_string(value).upper()
        if len(value) != 2 or not value.isalpha():
            raise ValueError(f"Invalid ISO country code: {value!r}")
        return value

    @field_validator("derivation_status")
    @classmethod
    def _derivation_status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed"}
        if value not in allowed:
            raise ValueError(f"derivation_status must be one of {allowed}")
        return value

    @field_validator("calculator_coverage_status")
    @classmethod
    def _calculator_coverage_status_allowed(cls, value: str) -> str:
        allowed = {"complete", "partial", "unclassified", "failed", "not_calculable"}
        if value not in allowed:
            raise ValueError(f"calculator_coverage_status must be one of {allowed}")
        return value


class CoreFees(BaseModel):
    """Consolidated core fees across all markets."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    markets: list[CoreFeeEntry] = Field(default_factory=list)


class PaymentMethodName(BaseModel):
    """Localized display name for a payment method."""

    model_config = ConfigDict(frozen=True)

    language: str
    name: str


class PaymentMethodEntry(BaseModel):
    """A normalized payment method in the cross-market catalog."""

    model_config = ConfigDict(frozen=True)

    method_id: str
    family: str
    display_name: str
    localized_names: list[PaymentMethodName] = Field(default_factory=list)
    supported_account_countries: list[str] = Field(default_factory=list)
    fee_rule_refs: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)


class PaymentMethodCatalog(BaseModel):
    """Cross-market payment-method catalog."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    methods: list[PaymentMethodEntry] = Field(default_factory=list)


class SchemaVersionInfo(BaseModel):
    """Schema version metadata."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    schema_path: str = "schemas/stripe-fees-v1.schema.json"
    schemas: list[str] = Field(
        default_factory=lambda: [
            "schemas/stripe-fees-v1.schema.json",
            "schemas/core-fees-v1.schema.json",
            "schemas/payment-methods-v1.schema.json",
            "schemas/index-v1.schema.json",
            "schemas/manifest-v1.schema.json",
        ]
    )
    description: str | None = None


class ChangeType(BaseModel):
    """A single classified change."""

    model_config = ConfigDict(frozen=True)

    kind: str
    country_code: str | None = None
    identifier: str | None = None
    before: Any | None = None
    after: Any | None = None
    message: str | None = None


class ChangeReport(BaseModel):
    """Machine-readable change report between two runs."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    generated_at: str | None = None
    changes: list[ChangeType] = Field(default_factory=list)
    has_regression: bool = False

    @model_validator(mode="before")
    @classmethod
    def _compute_has_regression(cls, data: Any) -> Any:
        regression_kinds = {
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
            "fee_component_disappeared",
            "condition_changed",
            "cap_changed",
            "classification_status_regression",
            "calculable_to_non_calculable",
            "duplicate_identifier",
            "duplicate_identity",
            "market_coverage_changed",
            "currency_changed",
            "source_url_changed",
            "large_percentage_change",
            "large_fixed_change",
            "schema_incompatible",
            "parser_output_empty",
        }
        if isinstance(data, dict):
            changes = data.get("changes", [])
            data["has_regression"] = any(
                (isinstance(change, dict) and change.get("kind") in regression_kinds)
                or (getattr(change, "kind", None) in regression_kinds)
                for change in changes
            )
        return data


class CrawlReport(BaseModel):
    """Summary of a crawl run."""

    model_config = ConfigDict(frozen=True)

    exit_code: int = 0
    changed: bool = False
    markets_processed: int = 0
    markets_failed: list[str] = Field(default_factory=list)
    markets_unsupported: list[str] = Field(default_factory=list)
    markets_reused: list[str] = Field(default_factory=list)
    warnings: list[ParserWarning] = Field(default_factory=list)
    change_report_path: str | None = None
    diagnostics_path: str | None = None
    cache_stats: CacheStats = Field(default_factory=CacheStats)
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary)

    @model_validator(mode="before")
    @classmethod
    def _exit_code_consistency(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("exit_code") == 0 and data.get("markets_failed"):
            data["exit_code"] = ExitCode.PARSER_FAILURE
        return data


class CrawlConfiguration(BaseModel):
    """Runtime crawl configuration."""

    model_config = ConfigDict(frozen=True)

    output_dir: str | None = None
    staging_dir: str | None = None
    timestamp: str | None = None
    markets: list[str] | None = None
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_workers: int = 3
    request_delay: float = 1.0
    max_retries: int = 3
    user_agent: str | None = None
    atomic: bool = True
    fail_on_regression: bool = False
    fail_on_warning: bool = False
    allow_market_drop: bool = False
    refresh_market_manifest: bool = False
    keep_diagnostics: bool = False
    verbose: bool = False
    strict: bool = False
    allow_partial: bool = False
    transient_policy: str = "preserve"
    max_response_size: int = 10 * 1024 * 1024  # 10 MB
    allowed_domains: list[str] = Field(default_factory=lambda: ["stripe.com", "www.stripe.com", "docs.stripe.com"])
    market_manifest_path: str | None = None
    offline_fixtures: dict[str, str] | None = None
    source_timestamp_override: str | None = None
    cache_dir: str | None = None
    cache_ttl_hours: float = 24.0
    no_cache: bool = False
    refresh_cache: bool = False

    @field_validator("max_workers")
    @classmethod
    def _max_workers_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_workers must be at least 1")
        return min(value, 10)

    @field_validator("timeout")
    @classmethod
    def _timeout_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout must be positive")
        return value

    @field_validator("transient_policy")
    @classmethod
    def _transient_policy_allowed(cls, value: str) -> str:
        allowed = {"preserve", "fail", "ignore"}
        if value not in allowed:
            raise ValueError(f"transient_policy must be one of {allowed}")
        return value

    @field_validator("cache_ttl_hours")
    @classmethod
    def _cache_ttl_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("cache_ttl_hours must be positive")
        return value
