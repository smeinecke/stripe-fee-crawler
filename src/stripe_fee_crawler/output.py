"""Deterministic, atomic publication of crawler output."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    ChangeReport,
    CoreFeeEntry,
    CoreFeeRule,
    CoreFees,
    FeeComponent,
    FeeRule,
    Market,
    MarketIndex,
    MarketIndexEntry,
    MarketManifest,
    MarketOutput,
    PaymentMethodCatalog,
    PaymentMethodEntry,
    SchemaVersionInfo,
    UnsupportedMarket,
)
from .normalize import normalize_method_name
from .validation import (
    generate_core_fees_schema,
    generate_index_schema,
    generate_manifest_schema,
    generate_market_output_schema,
    generate_payment_methods_schema,
    validate_all_output,
)

logger = logging.getLogger(__name__)


def _serialize(obj: Any) -> str:
    """Serialize data to deterministic, stable JSON with trailing newline."""
    text = json.dumps(
        obj,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
    )
    return text.replace("\r\n", "\n") + "\n"


def _write_json(path: Path, data: Any) -> None:
    """Write deterministic JSON to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(data), encoding="utf-8")


def _is_same_file(path: Path, content: str) -> bool:
    """Return True if the path exists and contains the given text."""
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8") == content


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class _JournalEntry:
    """One managed-path operation in the publication transaction.

    The entry is appended to the journal before the first filesystem mutation.
    This is essential: if the backup succeeds but installing the staged path
    fails, rollback still knows how to restore the original live path.
    """

    managed_name: str
    live_path: Path
    backup_path: Path
    staged_path: Path | None
    action: str
    original_existed: bool
    backup_created: bool = False
    live_installed: bool = False
    finalized: bool = False


class OutputPublisher:
    """Atomic, deterministic publisher for the stripe-fee-data repository."""

    # These are the only paths the crawler owns. The output directory itself may
    # be the root of a git repository and must never be renamed or deleted.
    MANAGED_PATHS = ("json", "meta", "schemas", "change-report.json")

    def __init__(
        self,
        output_dir: str | Path,
        staging_dir: str | Path | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.staging_dir = Path(staging_dir) if staging_dir else None
        self.timestamp = timestamp

    def _make_staging(self) -> Path:
        if self.staging_dir and self.staging_dir.is_relative_to(self.output_dir):
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return self.staging_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=str(self.output_dir)))

    def publish(
        self,
        outputs: list[MarketOutput],
        markets: list[Market],
        unsupported: list[UnsupportedMarket],
        transient_failures: list[UnsupportedMarket],
        aliases: dict[str, str] | None = None,
        fee_page_urls: dict[str, list[str]] | None = None,
        crawler_revision: str | None = None,
    ) -> tuple[bool, Path]:
        """Write all output files to a staging directory and return (changed, staging_path)."""
        staging = self._make_staging()
        json_dir = staging / "json"
        meta_dir = staging / "meta"
        schemas_dir = staging / "schemas"
        json_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir.mkdir(parents=True, exist_ok=True)

        self._write_market_files(staging, outputs)
        self._write_core_fees(staging, outputs)
        self._write_payment_methods(staging, outputs)
        self._write_manifest(staging, markets, unsupported, transient_failures, aliases, fee_page_urls)
        self._write_crawler_revision(staging, crawler_revision)
        self._write_schemas(schemas_dir)

        return staging != self.output_dir, staging

    def _write_market_files(self, staging: Path, outputs: list[MarketOutput]) -> MarketIndex:
        """Publish per-market files and build the index."""
        entries: list[MarketIndexEntry] = []
        for output in sorted(outputs, key=lambda o: o.market.account_country):
            country = output.market.account_country
            file_name = f"{country}.json"
            path = staging / "json" / file_name
            data = output.model_dump()
            _write_json(path, data)
            content_hash = _sha256_file(path)
            entries.append(
                MarketIndexEntry(
                    account_country=country,
                    stripe_market_code=output.market.stripe_market_code,
                    locale=output.market.locale,
                    data_path=f"json/{file_name}",
                    source_urls=[s.canonical_url or s.requested_url for s in output.sources],
                    source_updated_at=output.sources[0].source_updated_at if output.sources else None,
                    derivation_status=output.derivation_status,
                    content_sha256=content_hash,
                )
            )

        index = MarketIndex(
            schema_version=1,
            generated_at=self.timestamp,
            markets=entries,
        )
        _write_json(staging / "json" / "index.json", index.model_dump())
        return index

    def _write_core_fees(self, staging: Path, outputs: list[MarketOutput]) -> CoreFees:
        """Consolidate classified rules across markets into core-fees.json."""
        core_entries: list[CoreFeeEntry] = []
        for output in sorted(outputs, key=lambda o: o.market.account_country):
            country = output.market.account_country
            rules = [
                _to_core_fee_rule(r)
                for r in output.derived_rules
                if r.classification_status in {"classified", "calculable_rule"}
            ]
            core_entries.append(
                CoreFeeEntry(
                    account_country=country,
                    stripe_market_code=output.market.stripe_market_code,
                    locale=output.market.locale,
                    derivation_status=output.derivation_status,
                    calculator_coverage_status=output.calculator_coverage_status,
                    coverage_summary=output.coverage_summary,
                    rules=rules,
                    unclassified_count=len(output.unclassified_entries),
                )
            )
        core_fees = CoreFees(
            schema_version=1,
            generated_at=self.timestamp,
            markets=core_entries,
        )
        _write_json(staging / "json" / "core-fees.json", core_fees.model_dump())
        return core_fees

    def _write_payment_methods(self, staging: Path, outputs: list[MarketOutput]) -> PaymentMethodCatalog:
        """Build a normalized cross-market payment method catalog."""
        methods_by_id: dict[str, dict[str, Any]] = {}
        for output in outputs:
            country = output.market.account_country
            for rule in output.derived_rules:
                if rule.classification_status == "conflict" or not rule.payment_method:
                    continue
                method_id = normalize_method_name(rule.payment_method)
                if method_id not in methods_by_id:
                    methods_by_id[method_id] = {
                        "display_name": rule.payment_method.replace("_", " ").title(),
                        "family": _family_for_method(rule.payment_method),
                        "localized_names": [],
                        "supported_account_countries": [],
                        "fee_rule_refs": [],
                        "source_refs": [],
                    }
                method = methods_by_id[method_id]
                if country not in method["supported_account_countries"]:
                    method["supported_account_countries"].append(country)
                if rule.rule_id not in method["fee_rule_refs"]:
                    method["fee_rule_refs"].append(rule.rule_id)
                if rule.source_url and rule.source_url not in method["source_refs"]:
                    method["source_refs"].append(rule.source_url)

        entries = []
        for method_id in sorted(methods_by_id):
            data = methods_by_id[method_id]
            entries.append(
                PaymentMethodEntry(
                    method_id=method_id,
                    family=data["family"],
                    display_name=data["display_name"],
                    localized_names=data["localized_names"],
                    supported_account_countries=sorted(data["supported_account_countries"]),
                    fee_rule_refs=sorted(data["fee_rule_refs"]),
                    source_refs=sorted(data["source_refs"]),
                )
            )

        catalog = PaymentMethodCatalog(
            schema_version=1,
            generated_at=self.timestamp,
            methods=entries,
        )
        _write_json(staging / "json" / "payment-methods.json", catalog.model_dump())
        return catalog

    def _write_manifest(
        self,
        staging: Path,
        markets: list[Market],
        unsupported: list[UnsupportedMarket],
        transient_failures: list[UnsupportedMarket],
        aliases: dict[str, str] | None = None,
        fee_page_urls: dict[str, list[str]] | None = None,
    ) -> MarketManifest:
        """Publish the market manifest and metadata files."""
        manifest = MarketManifest(
            schema_version=1,
            generated_at=self.timestamp,
            markets=sorted(markets, key=lambda m: m.account_country),
            unsupported=sorted(unsupported, key=lambda u: u.stripe_market_code),
            transient_failures=sorted(transient_failures, key=lambda u: u.stripe_market_code),
            aliases=aliases or {},
            fee_page_urls=fee_page_urls or {},
        )
        _write_json(staging / "meta" / "markets.json", manifest.model_dump())
        unsupported_only = [u for u in unsupported if u.status == "unsupported"]
        _write_json(staging / "meta" / "unsupported-markets.json", unsupported_only)
        _write_json(staging / "meta" / "transient-failures.json", transient_failures)
        _write_json(
            staging / "meta" / "schema-version.json",
            SchemaVersionInfo().model_dump(),
        )
        return manifest

    def _write_crawler_revision(self, staging: Path, crawler_revision: str | None) -> None:
        """Write the crawler Git revision used to generate this data."""
        if crawler_revision:
            _write_json(
                staging / "meta" / "crawler-revision.json",
                {"crawler_revision": crawler_revision, "generated_at": self.timestamp},
            )

    def _write_schemas(self, schemas_dir: Path) -> None:
        """Generate JSON schemas from Pydantic models and write them to staging."""
        _write_json(schemas_dir / "stripe-fees-v1.schema.json", generate_market_output_schema())
        _write_json(schemas_dir / "core-fees-v1.schema.json", generate_core_fees_schema())
        _write_json(schemas_dir / "payment-methods-v1.schema.json", generate_payment_methods_schema())
        _write_json(schemas_dir / "index-v1.schema.json", generate_index_schema())
        _write_json(schemas_dir / "manifest-v1.schema.json", generate_manifest_schema())

    def publish_change_report(
        self,
        staging: Path,
        change_report: ChangeReport | None,
    ) -> None:
        """Publish the computed change report, overwriting any stale previous report."""
        if change_report is None:
            change_report = ChangeReport()
        _write_json(staging / "change-report.json", change_report.model_dump())

    def commit(self, staging: Path, validate: bool = True) -> tuple[bool, list[str]]:
        """Atomically replace only managed paths with the staged tree.

        This method is safe when ``output_dir`` is the root of a git repository:
        only ``MANAGED_PATHS`` are touched. Staging is cross-file validated
        before any live path is modified. Backups remain available until the
        live tree has also passed validation.
        """
        changed_files = self._list_changed_files(staging)
        if not changed_files and self._output_dir_exists_and_matches(staging):
            self.rollback(staging)
            return False, []

        if validate:
            errors = validate_all_output(staging, strict=False)
            if errors.get("errors"):
                self.rollback(staging)
                raise CrawlerValidationError("Staging output failed validation:\n" + "\n".join(errors["errors"]))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        journal: list[_JournalEntry] = []
        finalized = False

        try:
            for name in self.MANAGED_PATHS:
                src = staging / name
                dst = self.output_dir / name
                backup = dst.with_name(f"{dst.name}.old")
                src_exists = src.exists()
                dst_exists = dst.exists()

                if dst_exists and src_exists:
                    action = "replaced"
                elif dst_exists and not src_exists:
                    action = "removed"
                elif not dst_exists and src_exists:
                    action = "added"
                else:
                    action = "none"

                entry = _JournalEntry(
                    managed_name=name,
                    live_path=dst,
                    backup_path=backup,
                    staged_path=src if src_exists else None,
                    action=action,
                    original_existed=dst_exists,
                )
                journal.append(entry)

                if action == "none":
                    continue

                self._remove_path(backup)

                if dst_exists:
                    os.rename(dst, backup)
                    entry.backup_created = True

                if src_exists:
                    os.rename(src, dst)
                    entry.live_installed = True

            if validate:
                errors = validate_all_output(self.output_dir, strict=False)
                if errors.get("errors"):
                    raise CrawlerValidationError("Live output failed validation:\n" + "\n".join(errors["errors"]))

            finalized = True
            for entry in journal:
                entry.finalized = True

            self._cleanup_backups_best_effort(journal)
            self.rollback(staging)
            return bool(changed_files), changed_files

        except Exception as exc:
            if not finalized:
                self._rollback_live(journal)
            self.rollback(staging)
            if isinstance(exc, CrawlerValidationError):
                raise
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

    def _list_changed_files(self, staging: Path) -> list[str]:
        """Return relative paths of managed files that differ from published output."""
        changed: list[str] = []
        for name in self.MANAGED_PATHS:
            src = staging / name
            if not src.exists():
                continue
            if src.is_dir():
                for src_file in src.rglob("*"):
                    if not src_file.is_file():
                        continue
                    rel = src_file.relative_to(staging)
                    dst = self.output_dir / rel
                    content = src_file.read_text(encoding="utf-8")
                    if not _is_same_file(dst, content):
                        changed.append(str(rel))
            else:
                rel = src.relative_to(staging)
                dst = self.output_dir / rel
                content = src.read_text(encoding="utf-8")
                if not _is_same_file(dst, content):
                    changed.append(str(rel))

        if self.output_dir.exists():
            for name in self.MANAGED_PATHS:
                dst = self.output_dir / name
                if not dst.exists():
                    continue
                if dst.is_dir():
                    for dst_file in dst.rglob("*"):
                        if not dst_file.is_file():
                            continue
                        rel = dst_file.relative_to(self.output_dir)
                        if not (staging / rel).exists():
                            changed.append(str(rel))
                else:
                    rel = dst.relative_to(self.output_dir)
                    if not (staging / rel).exists():
                        changed.append(str(rel))
        return sorted(set(changed))

    def _output_dir_exists_and_matches(self, staging: Path) -> bool:
        """Return whether the output directory exists and matches staging exactly."""
        if not self.output_dir.exists():
            return False
        return not self._list_changed_files(staging)

    def rollback(self, staging: Path) -> None:
        """Clean up the staging directory on failure or when no change is published."""
        if staging.exists() and staging != self.output_dir and not self._is_managed_path(staging):
            shutil.rmtree(staging, ignore_errors=True)

    def _cleanup_backups_best_effort(self, journal: list[_JournalEntry]) -> None:
        """Remove backups after a successful, validated commit."""
        for entry in journal:
            if entry.backup_created and entry.backup_path.exists():
                try:
                    self._remove_path(entry.backup_path)
                except Exception as exc:  # pragma: no cover - platform dependent
                    logger.warning("Could not remove publication backup %s: %s", entry.backup_path, exc)

    def _rollback_live(self, journal: list[_JournalEntry]) -> None:
        """Restore managed live paths to their pre-transaction state."""
        failed: list[str] = []

        for entry in reversed(journal):
            if entry.action == "none":
                continue

            live = entry.live_path
            backup = entry.backup_path
            mutation_happened = entry.live_installed or entry.backup_created

            if mutation_happened and live.exists():
                try:
                    self._remove_path(live)
                except Exception as exc:
                    failed.append(f"Could not remove live path {live}: {exc}")
                    continue

            if entry.original_existed:
                if entry.backup_created and backup.exists():
                    try:
                        os.rename(backup, live)
                    except Exception as exc:
                        failed.append(f"Could not restore backup {backup} to {live}: {exc}")
                elif mutation_happened and not live.exists():
                    failed.append(f"Backup missing for {live}; original state cannot be restored")
            else:
                if entry.live_installed and live.exists():
                    try:
                        self._remove_path(live)
                    except Exception as exc:
                        failed.append(f"Could not remove added path {live}: {exc}")

        if failed:
            logger.error("Rollback completed with errors:\n%s", "\n".join(failed))
            raise CrawlerValidationError("Rollback completed with errors:\n" + "\n".join(failed))

    def _remove_path(self, path: Path) -> None:
        """Remove a file or directory tree, ignoring missing paths."""
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _is_managed_path(self, path: Path) -> bool:
        """Return whether the path is one of the live managed paths."""
        try:
            rel = path.relative_to(self.output_dir)
        except ValueError:
            return False
        parts = rel.parts
        if not parts:
            return False
        return parts[0] in self.MANAGED_PATHS or (len(parts) == 1 and parts[0] in self.MANAGED_PATHS)


def _to_core_fee_components(rule: FeeRule) -> list[FeeComponent]:
    """Build compact fee components from a FeeRule, falling back to legacy flat fields."""
    if rule.fee_components:
        return [
            FeeComponent(
                type=c.type,
                value=c.value,
                amount=c.amount,
                currency=c.currency,
                minor_amount=c.minor_amount,
                basis_points=c.basis_points,
                schedule_id=c.schedule_id,
                operator=c.operator,
            )
            for c in rule.fee_components
        ]
    components: list[FeeComponent] = []
    if rule.percentage:
        components.append(
            FeeComponent(
                type="percentage",
                value=rule.percentage,
                basis_points=rule.basis_points,
            )
        )
    if rule.fixed_amount:
        components.append(
            FeeComponent(
                type="fixed_amount",
                amount=rule.fixed_amount,
                currency=rule.fixed_currency,
                minor_amount=rule.fixed_amount_minor,
            )
        )
    if rule.minimum_amount:
        components.append(FeeComponent(type="minimum_fee", amount=rule.minimum_amount, currency=rule.fixed_currency))
    if rule.maximum_amount:
        components.append(FeeComponent(type="maximum_fee", amount=rule.maximum_amount, currency=rule.fixed_currency))
    return components


def _to_core_fee_rule(rule: FeeRule) -> CoreFeeRule:
    """Convert an internal FeeRule to a compact, calculator-facing CoreFeeRule."""
    return CoreFeeRule(
        rule_id=rule.rule_id,
        product_id=rule.product_id,
        variant_id=rule.variant_id,
        label=rule.label or rule.name,
        provider=rule.provider,
        account_country=rule.account_country,
        channel=rule.channel,
        payment_method=rule.payment_method,
        conditions=rule.conditions,
        fee_components=_to_core_fee_components(rule),
        unit=rule.unit,
        behavior=rule.behavior,
        exactness=rule.exactness,
        classification_status=rule.classification_status
        if rule.classification_status in {"classified", "calculable_rule"}
        else "calculable_rule",
        confidence=rule.confidence,
        fee_evidence=rule.fee_evidence,
    )


def _family_for_method(method: str) -> str:
    lower = method.lower()
    if lower in {"card", "domestic_card", "international_card", "premium_card", "standard_card"}:
        return "card"
    if lower in {"sepa_direct_debit", "sepa_bank_transfer", "ach_direct_debit", "bacs_direct_debit"}:
        return "bank_debit"
    if lower in {
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
    }:
        return "bank_redirect"
    if lower in {"alipay", "wechat_pay", "mobilepay", "paypal", "revolut_pay", "amazon_pay", "satispay"}:
        return "wallet"
    if lower in {"klarna", "billie", "scalapay"}:
        return "buy_now_pay_later"
    if lower in {"multibanco"}:
        return "cash_voucher"
    return "other"
