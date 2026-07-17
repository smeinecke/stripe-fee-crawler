"""Crawler orchestration: discovery, extraction, classification, and output."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

from .classify import derive_market_fees
from .discovery import (
    MarketDiscoveryError,
    UnsupportedMarketError,
    build_market_from_code,
    discover_fee_pages,
    discover_markets,
    get_bootstrap_markets,
)
from .exceptions import (
    AccessChallengeError,
    FeePageError,
    NetworkError,
    ParserError,
)
from .extract import extract_page_source, extract_pricing_entries
from .http import HttpClient, HttpResponse
from .models import (
    ChangeReport,
    CoverageSummary,
    CrawlConfiguration,
    CrawlReport,
    Market,
    MarketOutput,
    ParserWarning,
    Source,
    UnsupportedMarket,
)
from .output import OutputPublisher
from .regression import check_regression
from .validation import validate_all_output

logger = logging.getLogger(__name__)


def _load_fixture(path: str | None) -> str | None:
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return None


def _crawler_revision() -> str | None:
    """Return the current crawler Git revision, or None if not available."""
    try:
        result = subprocess.run(  # nosec
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


class StripeCrawler:
    """High-level crawler for Stripe public pricing pages."""

    def __init__(self, config: CrawlConfiguration | None = None) -> None:
        self.config = config or CrawlConfiguration()
        self.http_client = HttpClient(config=self.config)
        self.warnings: list[ParserWarning] = []
        self.aliases: dict[str, str] = {}
        self.fee_page_urls: dict[str, list[str]] = {}

    async def discover(self) -> list[Market]:
        """Discover Stripe markets dynamically or from the bootstrap list."""
        if self.config.markets:
            self.aliases = {}
            return [build_market_from_code(code, status="supported") for code in self.config.markets]
        try:
            markets, aliases = await discover_markets(self.http_client, self.config)
            self.aliases = aliases
        except MarketDiscoveryError as exc:
            logger.warning("Dynamic market discovery failed: %s; using bootstrap list", exc)
            markets = get_bootstrap_markets()
            self.aliases = {}
        return markets

    async def crawl_market(self, market: Market) -> MarketOutput:
        """Crawl a single market and return normalized output."""
        sources: list[Source] = []
        entries: list[Any] = []
        sections: list[Any] = []
        warnings: list[ParserWarning] = []
        transient = False

        # Determine URLs.
        if market.status == "pricing_page_unavailable":
            return MarketOutput(
                market=market,
                sources=[],
                derivation_status="unclassified",
                transient_failure=False,
                unsupported_reason="pricing_page_unavailable",
            )

        try:
            pricing_url, payment_methods_url = await discover_fee_pages(self.http_client, market, self.config)
            self.fee_page_urls[market.stripe_market_code] = [u for u in [pricing_url, payment_methods_url] if u]
        except UnsupportedMarketError as exc:
            return MarketOutput(
                market=market,
                sources=[],
                derivation_status="unclassified",
                transient_failure=False,
                unsupported_reason=exc.args[0] if exc.args else "unsupported",
            )
        except (FeePageError, NetworkError, AccessChallengeError) as exc:
            transient = True
            warnings.append(
                ParserWarning(
                    code="transient_failure",
                    message=f"Transient failure crawling {market.account_country}: {exc}",
                )
            )
            return MarketOutput(
                market=market,
                sources=[],
                derivation_status="failed",
                transient_failure=True,
                warnings=warnings,
            )

        # Crawl main pricing page.
        try:
            main_html, main_source = await self._fetch_page(pricing_url, market)
            if main_html is not None and main_source is not None:
                sources.append(main_source)
                page_entries, page_sections = extract_pricing_entries(
                    main_html, main_source.canonical_url or pricing_url, "pricing"
                )
                entries.extend(page_entries)
                sections.extend(page_sections)
        except Exception as exc:
            warnings.append(
                ParserWarning(
                    code="pricing_page_extraction_failure",
                    message=f"Failed to extract main pricing page for {market.account_country}: {exc}",
                )
            )

        # Crawl local payment methods page when available.
        if payment_methods_url and payment_methods_url != pricing_url:
            try:
                lpm_html, lpm_source = await self._fetch_page(payment_methods_url, market)
                if lpm_html is not None and lpm_source is not None:
                    sources.append(lpm_source)
                    page_entries, page_sections = extract_pricing_entries(
                        lpm_html, lpm_source.canonical_url or payment_methods_url, "local-payment-methods"
                    )
                    entries.extend(page_entries)
                    sections.extend(page_sections)
            except Exception as exc:
                warnings.append(
                    ParserWarning(
                        code="payment_methods_extraction_failure",
                        message=f"Failed to extract payment methods page for {market.account_country}: {exc}",
                    )
                )

        rules, unclassified, derivation_status, coverage_summary, calculator_coverage_status = derive_market_fees(
            entries, market=market
        )
        timestamp = self.config.timestamp or self.config.source_timestamp_override

        return MarketOutput(
            schema_version=1,
            generated_at=timestamp,
            market=market,
            sources=sources,
            sections=sections,
            entries=entries,
            derived_rules=rules,
            unclassified_entries=unclassified,
            warnings=warnings,
            derivation_status=derivation_status,
            calculator_coverage_status=calculator_coverage_status,
            coverage_summary=coverage_summary,
            transient_failure=transient,
        )

    async def _fetch_page(self, url: str, market: Market) -> tuple[str | None, Source | None]:
        if self.config.offline_fixtures and url in self.config.offline_fixtures:
            fixture_path = self.config.offline_fixtures[url]
            html_text = _load_fixture(fixture_path)
            if html_text is None:
                raise ParserError(f"Offline fixture not found: {fixture_path}")
            response = HttpResponse(
                url=url,
                requested_url=url,
                status_code=200,
                content=html_text.encode("utf-8"),
                text=html_text,
                headers={"content-type": "text/html"},
            )
        else:
            response = await self.http_client.get(url, market=market.account_country, locale=market.locale)

        source = extract_page_source(response)
        if source.detected_market and source.detected_market.upper() != market.account_country.upper():
            raise FeePageError(
                f"Requested {market.account_country} but page served {source.detected_market} "
                f"(effective_url={source.effective_url}, requested_url={source.requested_url})"
            )
        return response.text, source

    async def crawl_all(
        self, markets: list[Market] | None = None
    ) -> tuple[list[MarketOutput], list[UnsupportedMarket]]:
        """Crawl all markets concurrently and return outputs plus unsupported list."""
        if markets is None:
            markets = await self.discover()

        unsupported: list[UnsupportedMarket] = []
        outputs: list[MarketOutput] = []
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def _crawl_one(market: Market) -> MarketOutput | None:
            async with semaphore:
                try:
                    return await self.crawl_market(market)
                except UnsupportedMarketError as exc:
                    unsupported.append(
                        UnsupportedMarket(
                            stripe_market_code=market.stripe_market_code,
                            account_country=market.account_country,
                            country_name=market.country_name,
                            tested_urls=exc.tested_urls,
                            reason=exc.args[0] if exc.args else "unsupported",
                        )
                    )
                    return None
                except Exception as exc:
                    logger.warning("Unexpected failure for %s: %s", market.account_country, exc)
                    return MarketOutput(
                        market=market,
                        sources=[],
                        derivation_status="failed",
                        transient_failure=True,
                        warnings=[
                            ParserWarning(
                                code="unexpected_failure",
                                message=f"Unexpected failure for {market.account_country}: {exc}",
                            )
                        ],
                    )

        tasks = [asyncio.create_task(_crawl_one(m)) for m in markets]
        results = await asyncio.gather(*tasks)
        outputs = [r for r in results if r is not None]
        return outputs, unsupported

    async def publish(
        self,
        outputs: list[MarketOutput],
        unsupported: list[UnsupportedMarket],
        output_dir: str | Path,
        atomic: bool = True,
        fail_on_regression: bool = False,
    ) -> CrawlReport:
        """Publish outputs to the data repository."""
        publisher = OutputPublisher(output_dir, timestamp=self.config.timestamp)
        staging: Path | None = None
        try:
            markets = [o.market for o in outputs]
            transient_failures = [
                UnsupportedMarket(
                    stripe_market_code=o.market.stripe_market_code,
                    account_country=o.market.account_country,
                    country_name=o.market.country_name,
                    status="transient_failure",
                    reason="transient failure during crawl",
                )
                for o in outputs
                if o.transient_failure
            ]

            crawler_revision = _crawler_revision()
            _, staging = publisher.publish(
                outputs,
                markets,
                unsupported,
                transient_failures,
                aliases=self.aliases,
                fee_page_urls=self.fee_page_urls,
                crawler_revision=crawler_revision,
            )

            # Regression check against previous published data if it exists.
            # If the previous change report already marked a regression, the old
            # dataset is stale and should be overwritten with a clean report.
            old_dir = Path(output_dir)
            old_change_report_path = old_dir / "change-report.json"
            stale_baseline = False
            if old_change_report_path.exists():
                try:
                    with open(old_change_report_path, encoding="utf-8") as fh:
                        stale_baseline = json.load(fh).get("has_regression", False)
                except Exception:
                    stale_baseline = False

            if stale_baseline:
                change_report = ChangeReport()
            elif (old_dir / "json" / "index.json").exists():
                change_report = check_regression(old_dir, staging)
            else:
                change_report = ChangeReport()
            publisher.publish_change_report(staging, change_report)

            if fail_on_regression and change_report.has_regression:
                from .exceptions import RegressionError

                raise RegressionError("Regression detected; publication aborted")

            if atomic:
                changed, _ = publisher.commit(staging, validate=True)
            else:
                validate_all_output(staging, strict=True)
                changed = _staging_changed(staging, old_dir)

            coverage_summary = _aggregate_coverage(outputs)
            report = CrawlReport(
                exit_code=0,
                changed=changed,
                markets_processed=len(outputs),
                markets_failed=[
                    o.market.account_country for o in outputs if o.transient_failure or o.derivation_status == "failed"
                ],
                markets_unsupported=[u.account_country for u in unsupported if u.account_country],
                warnings=self.warnings,
                change_report_path=str(Path(output_dir) / "change-report.json"),
                cache_stats=self.http_client.cache_stats,
                coverage_summary=coverage_summary,
            )
            return report
        except Exception:
            if staging is not None:
                publisher.rollback(staging)
            raise

    async def close(self) -> None:
        await self.http_client.close()

    async def __aenter__(self) -> StripeCrawler:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()


def _staging_changed(staging_dir: Path, output_dir: Path) -> bool:
    """Return True if any published file differs from the previous output."""
    for subdir in ("json", "meta", "schemas"):
        for src in (staging_dir / subdir).rglob("*"):
            if not src.is_file():
                continue
            dst = output_dir / subdir / src.relative_to(staging_dir / subdir)
            if not dst.exists():
                return True
            if not _files_equal(src, dst):
                return True
    return False


def _aggregate_coverage(outputs: list[MarketOutput]) -> CoverageSummary:
    """Sum per-market coverage summaries into a crawl-level summary."""
    totals: dict[str, int] = {}
    for output in outputs:
        for field, value in output.coverage_summary.model_dump().items():
            totals[field] = totals.get(field, 0) + int(value)
    return CoverageSummary(**totals)


def _files_equal(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        return fa.read() == fb.read()
