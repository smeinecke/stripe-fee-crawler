"""Deterministic, atomic publication of crawler output."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .models import (
    ChangeReport,
    CoreFeeEntry,
    CoreFees,
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
from .validation import validate_all_output

logger = logging.getLogger(__name__)


def _dump_json(data: Any) -> bytes:
    """Serialize data to deterministic, stable JSON with trailing newline."""
    text = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda obj: obj.model_dump() if hasattr(obj, "model_dump") else str(obj),
    )
    text = text.replace("\r\n", "\n")
    return text.encode("utf-8") + b"\n"


def _write_atomic(path: Path, content: bytes) -> None:
    """Write a file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class OutputPublisher:
    """Deterministic publisher for the stripe-fee-data repository."""

    def __init__(self, output_dir: str | Path, timestamp: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.timestamp = timestamp
        self.staging_dir = None

    def _write_json(self, path: Path, data: Any) -> Path:
        content = _dump_json(data)
        _write_atomic(path, content)
        return path

    def _init_staging(self) -> Path:
        staging = Path(tempfile.mkdtemp(prefix="stripe-fee-crawl-"))
        (staging / "json").mkdir(parents=True, exist_ok=True)
        (staging / "meta").mkdir(parents=True, exist_ok=True)
        (staging / "schemas").mkdir(parents=True, exist_ok=True)
        self.staging_dir = staging
        return staging

    def _copy_schemas(self, staging: Path, schemas_dir: str | Path | None = None) -> None:
        if schemas_dir is None:
            schemas_dir = Path(__file__).parent.parent.parent / "schemas"
        schemas_path = Path(schemas_dir)
        if schemas_path.exists():
            for schema_file in schemas_path.glob("*.json"):
                shutil.copy2(schema_file, staging / "schemas" / schema_file.name)

    def publish_markets(
        self,
        outputs: list[MarketOutput],
        schemas_dir: str | Path | None = None,
    ) -> MarketIndex:
        """Publish per-market files to the staging directory and build an index."""
        staging = self._init_staging()
        self._copy_schemas(staging, schemas_dir)

        entries: list[MarketIndexEntry] = []
        for output in sorted(outputs, key=lambda o: o.market.account_country):
            country = output.market.account_country
            file_name = f"{country}.json"
            path = staging / "json" / file_name
            data = output.model_dump()
            self._write_json(path, data)
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
        self._write_json(staging / "json" / "index.json", index.model_dump())
        return index

    def publish_core_fees(self, outputs: list[MarketOutput]) -> CoreFees:
        """Consolidate classified rules across markets into core-fees.json."""
        staging = self._ensure_staging()
        core_entries: list[CoreFeeEntry] = []
        for output in sorted(outputs, key=lambda o: o.market.account_country):
            country = output.market.account_country
            rules = [r for r in output.derived_rules if r.classification_status == "classified"]
            core_entries.append(
                CoreFeeEntry(
                    account_country=country,
                    stripe_market_code=output.market.stripe_market_code,
                    locale=output.market.locale,
                    derivation_status=output.derivation_status,
                    rules=rules,
                    unclassified_count=len(output.unclassified_entries),
                )
            )
        core_fees = CoreFees(
            schema_version=1,
            generated_at=self.timestamp,
            markets=core_entries,
        )
        self._write_json(staging / "json" / "core-fees.json", core_fees.model_dump())
        return core_fees

    def publish_payment_methods(self, outputs: list[MarketOutput]) -> PaymentMethodCatalog:
        """Build a normalized cross-market payment method catalog."""
        staging = self._ensure_staging()
        methods_by_id: dict[str, dict[str, Any]] = {}

        for output in outputs:
            country = output.market.account_country
            for rule in output.derived_rules:
                if not rule.payment_method:
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
        self._write_json(staging / "json" / "payment-methods.json", catalog.model_dump())
        return catalog

    def publish_manifest(
        self,
        markets: list[Market],
        unsupported: list[UnsupportedMarket],
        aliases: dict[str, str],
        transient_failures: list[UnsupportedMarket],
    ) -> MarketManifest:
        """Publish the market manifest and metadata files."""
        staging = self._ensure_staging()
        manifest = MarketManifest(
            schema_version=1,
            generated_at=self.timestamp,
            markets=sorted(markets, key=lambda m: m.account_country),
            unsupported=sorted(unsupported, key=lambda u: u.stripe_market_code),
            aliases=aliases,
            transient_failures=sorted(transient_failures, key=lambda u: u.stripe_market_code),
        )
        self._write_json(staging / "meta" / "markets.json", manifest.model_dump())
        unsupported_only = [u for u in unsupported if u.status == "unsupported"]
        self._write_json(staging / "meta" / "unsupported-markets.json", unsupported_only)
        self._write_json(staging / "meta" / "transient-failures.json", transient_failures)
        self._write_json(
            staging / "meta" / "schema-version.json",
            SchemaVersionInfo().model_dump(),
        )
        return manifest

    def publish_change_report(self, change_report: ChangeReport) -> None:
        """Publish the change report at the repository root."""
        staging = self._ensure_staging()
        self._write_json(staging / "change-report.json", change_report.model_dump())

    def _ensure_staging(self) -> Path:
        if self.staging_dir is None:
            return self._init_staging()
        return self.staging_dir

    def commit(self, validate: bool = True) -> Path:
        """Atomically swap the published files into the output directory.

        Only the json/, meta/, schemas/, and change-report.json paths within the
        output directory are replaced. The output directory itself is never renamed.
        """
        if self.staging_dir is None:
            raise RuntimeError("No staging directory exists; publish data first")
        staging = self.staging_dir

        if validate:
            validate_all_output(staging, strict=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("json", "meta", "schemas"):
            src = staging / subdir
            dst = self.output_dir / subdir
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)
        report_src = staging / "change-report.json"
        report_dst = self.output_dir / "change-report.json"
        if report_src.exists():
            shutil.copy2(report_src, report_dst)

        return self.output_dir

    def rollback(self) -> None:
        """Discard the staging directory."""
        if self.staging_dir and self.staging_dir.exists():
            shutil.rmtree(self.staging_dir)
            self.staging_dir = None


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
