"""Command-line interface for the Stripe fee crawler."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import click

from .crawler import StripeCrawler
from .exceptions import (
    AccessChallengeError,
    ConfigurationError,
    CrawlerError,
    ExitCode,
    NetworkError,
    ParserError,
    RegressionError,
    UnsupportedMarketError,
    ValidationError,
)
from .extract import extract_pricing_entries
from .models import CrawlConfiguration, Market
from .regression import check_regression
from .validation import validate_data_repository


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--strict", is_flag=True, help="Fail on warnings and non-exact data.")
@click.pass_context
def main(ctx: click.Context, verbose: bool, strict: bool) -> None:
    """Stripe fee crawler CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["strict"] = strict
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _config_from_options(
    markets: tuple[str, ...] | None = None,
    max_workers: int = 3,
    timeout: float = 30.0,
    retries: int = 3,
    request_delay: float = 1.0,
    strict: bool = False,
    allow_partial: bool = False,
    source_timestamp: str | None = None,
) -> CrawlConfiguration:
    return CrawlConfiguration(
        markets=list(markets) if markets else None,
        max_workers=max_workers,
        timeout=timeout,
        max_retries=retries,
        request_delay=request_delay,
        strict=strict,
        allow_partial=allow_partial,
        source_timestamp_override=source_timestamp,
    )


@main.command(name="discover-markets")
@click.pass_context
def discover_markets_cmd(ctx: click.Context) -> None:
    """Discover Stripe markets from public pages."""
    config = _config_from_options()
    asyncio.run(_discover_markets(ctx, config))


async def _discover_markets(ctx: click.Context, config: CrawlConfiguration) -> None:
    async with StripeCrawler(config) as crawler:
        markets = await crawler.discover()
    click.echo(json.dumps([m.model_dump() for m in markets], indent=2, ensure_ascii=False))


@main.command(name="crawl-market")
@click.argument("country_code")
@click.option("--output-format", default="json", type=click.Choice(["json", "summary"]))
@click.option("--fixture-pricing", type=click.Path(exists=True))
@click.option("--fixture-lpm", type=click.Path(exists=True))
@click.option("--max-workers", default=3, type=int)
@click.option("--timeout", default=30.0, type=float)
@click.option("--retries", default=3, type=int)
@click.option("--request-delay", default=1.0, type=float)
@click.option("--source-timestamp", default=None)
@click.pass_context
def crawl_market_cmd(
    ctx: click.Context,
    country_code: str,
    output_format: str,
    fixture_pricing: str | None,
    fixture_lpm: str | None,
    max_workers: int,
    timeout: float,
    retries: int,
    request_delay: float,
    source_timestamp: str | None,
) -> None:
    """Crawl a single market."""
    from .discovery import _payment_methods_url_for, _pricing_url_for, build_market_from_code

    fixtures: dict[str, str] = {}
    if fixture_pricing:
        fixtures["pricing"] = fixture_pricing
    if fixture_lpm:
        fixtures["lpm"] = fixture_lpm
    market = build_market_from_code(country_code, status="supported")
    offline_fixtures: dict[str, str] = {}
    if fixtures.get("pricing"):
        offline_fixtures[_pricing_url_for(market)] = fixtures["pricing"]
    if fixtures.get("lpm"):
        lpm_url = _payment_methods_url_for(market)
        if lpm_url:
            offline_fixtures[lpm_url] = fixtures["lpm"]
    config = _config_from_options(
        markets=(country_code,),
        max_workers=max_workers,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
        strict=ctx.obj.get("strict", False),
        source_timestamp=source_timestamp,
    )
    config = config.model_copy(update={"offline_fixtures": offline_fixtures})
    asyncio.run(_crawl_market(ctx, config, output_format, market))


async def _crawl_market(
    ctx: click.Context, config: CrawlConfiguration, output_format: str, market: Market | None = None
) -> None:
    from .discovery import build_market_from_code

    if market is None:
        country_code = config.markets[0] if config.markets else "US"
        market = build_market_from_code(country_code, status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    if output_format == "json":
        click.echo(json.dumps(output.model_dump(), indent=2, ensure_ascii=False))
    else:
        click.echo(f"Market: {output.market.account_country}")
        click.echo(f"Sources: {len(output.sources)}")
        click.echo(f"Entries: {len(output.entries)}")
        click.echo(f"Derived rules: {len(output.derived_rules)}")
        click.echo(f"Unclassified: {len(output.unclassified_entries)}")
        click.echo(f"Warnings: {len(output.warnings)}")
        if output.transient_failure:
            click.echo("Transient failure: yes")


@main.command(name="crawl")
@click.option("--output", required=True, type=click.Path())
@click.option("--atomic", is_flag=True, default=True)
@click.option("--fail-on-regression", is_flag=True, default=False)
@click.option("--market", "markets", multiple=True)
@click.option("--max-workers", default=3, type=int)
@click.option("--timeout", default=30.0, type=float)
@click.option("--retries", default=3, type=int)
@click.option("--request-delay", default=1.0, type=float)
@click.option("--source-timestamp", default=None)
@click.option("--allow-partial", is_flag=True)
@click.option("--report", type=click.Path(), help="Write machine-readable JSON report to this path.")
@click.pass_context
def crawl_cmd(
    ctx: click.Context,
    output: str,
    atomic: bool,
    fail_on_regression: bool,
    markets: tuple[str, ...],
    max_workers: int,
    timeout: float,
    retries: int,
    request_delay: float,
    source_timestamp: str | None,
    allow_partial: bool,
    report: str | None,
) -> None:
    """Crawl all markets and publish to the data repository."""
    config = _config_from_options(
        markets=markets,
        max_workers=max_workers,
        timeout=timeout,
        retries=retries,
        request_delay=request_delay,
        strict=ctx.obj.get("strict", False),
        allow_partial=allow_partial,
        source_timestamp=source_timestamp,
    )
    asyncio.run(_crawl_all(ctx, config, output, atomic, fail_on_regression, report))


async def _crawl_all(
    ctx: click.Context,
    config: CrawlConfiguration,
    output_dir: str,
    atomic: bool,
    fail_on_regression: bool,
    report: str | None = None,
) -> None:
    async with StripeCrawler(config) as crawler:
        outputs, unsupported = await crawler.crawl_all()
        report_obj = await crawler.publish(
            outputs,
            unsupported,
            output_dir,
            atomic=atomic,
            fail_on_regression=fail_on_regression,
        )
    report_text = json.dumps(report_obj.model_dump(), indent=2, ensure_ascii=False)
    click.echo(report_text)
    if report:
        Path(report).write_text(report_text + "\n", encoding="utf-8")
    if report_obj.exit_code != 0:
        sys.exit(report_obj.exit_code)


@main.command(name="validate")
@click.argument("data_dir", type=click.Path(exists=True))
@click.pass_context
def validate_cmd(ctx: click.Context, data_dir: str) -> None:
    """Validate a stripe-fee-data repository."""
    try:
        result = validate_data_repository(data_dir)
    except ValidationError as exc:
        click.echo(f"Validation failed: {exc}", err=True)
        sys.exit(ExitCode.VALIDATION_FAILURE)
    if result["success"]:
        click.echo("Validation passed.")
    else:
        for error in result["errors"]:
            click.echo(f"Error: {error}", err=True)
        sys.exit(ExitCode.VALIDATION_FAILURE)


@main.command(name="inspect")
@click.argument("fixture_path", type=click.Path(exists=True))
@click.option("--page-kind", default="pricing", type=click.Choice(["pricing", "local-payment-methods"]))
@click.option("--base-url", default="https://stripe.com/pricing")
@click.pass_context
def inspect_cmd(
    ctx: click.Context,
    fixture_path: str,
    page_kind: str,
    base_url: str,
) -> None:
    """Inspect a local HTML fixture and print extracted entries."""
    with open(fixture_path, encoding="utf-8") as fh:
        html_text = fh.read()
    entries, sections = extract_pricing_entries(html_text, base_url, page_kind=page_kind)
    click.echo(
        json.dumps(
            {
                "sections": [s.model_dump() for s in sections],
                "entries": [e.model_dump() for e in entries],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


@main.command(name="diff")
@click.argument("old_data")
@click.argument("new_data")
@click.pass_context
def diff_cmd(ctx: click.Context, old_data: str, new_data: str) -> None:
    """Compare two published datasets and print a change report."""
    report = check_regression(old_data, new_data)
    click.echo(json.dumps(report.model_dump(), indent=2, ensure_ascii=False))
    if report.has_regression:
        sys.exit(ExitCode.REGRESSION_FAILURE)


def _exit_code_for_error(exc: Exception) -> int:
    if isinstance(exc, NetworkError):
        return ExitCode.NETWORK_FAILURE
    if isinstance(exc, AccessChallengeError):
        return ExitCode.ACCESS_CHALLENGE
    if isinstance(exc, ParserError):
        return ExitCode.PARSER_FAILURE
    if isinstance(exc, ValidationError):
        return ExitCode.VALIDATION_FAILURE
    if isinstance(exc, RegressionError):
        return ExitCode.REGRESSION_FAILURE
    if isinstance(exc, UnsupportedMarketError):
        return ExitCode.UNSUPPORTED_MARKET
    if isinstance(exc, ConfigurationError):
        return ExitCode.CONFIGURATION_ERROR
    return ExitCode.UNEXPECTED_ERROR


if __name__ == "__main__":
    try:
        main()
    except CrawlerError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(_exit_code_for_error(exc))
    except Exception as exc:
        click.echo(f"Unexpected error: {exc}", err=True)
        sys.exit(ExitCode.UNEXPECTED_ERROR)
