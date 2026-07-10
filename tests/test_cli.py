"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from stripe_fee_crawler.cli import main


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "discover-markets" in result.output


def test_cli_inspect_fixture(fixtures_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "inspect",
            str(fixtures_dir / "de-pricing.html"),
            "--page-kind",
            "pricing",
            "--base-url",
            "https://stripe.com/en-de/pricing",
        ],
    )
    assert result.exit_code == 0
    assert "Domestic card payments" in result.output or "Payments" in result.output


def test_cli_validate(tmp_path: Path) -> None:
    from stripe_fee_crawler.models import Market, MarketOutput, Source
    from stripe_fee_crawler.output import OutputPublisher

    market = Market(
        stripe_market_code="de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    publisher.publish_markets([output])
    publisher.commit(validate=False)

    runner = CliRunner()
    result = runner.invoke(main, ["validate", str(tmp_path)])
    assert result.exit_code == 0
    assert "Validation passed" in result.output


def test_cli_crawl_market(fixtures_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "crawl-market",
            "DE",
            "--output-format",
            "summary",
            "--fixture-pricing",
            str(fixtures_dir / "de-pricing.html"),
            "--fixture-lpm",
            str(fixtures_dir / "de-lpm.html"),
        ],
    )
    assert result.exit_code == 0
    assert "Market: DE" in result.output


def test_cli_diff(tmp_path: Path) -> None:
    from stripe_fee_crawler.models import Market, MarketOutput, Source
    from stripe_fee_crawler.output import OutputPublisher

    market = Market(
        stripe_market_code="de",
        account_country="DE",
        country_name="Germany",
        locale="en-de",
        url_prefix="https://stripe.com/en-de",
        status="supported",
    )
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derivation_status="partial",
    )
    old = tmp_path / "old"
    new = tmp_path / "new"
    publisher_old = OutputPublisher(old, timestamp=None)
    publisher_old.publish_markets([output])
    publisher_old.commit(validate=False)
    publisher_new = OutputPublisher(new, timestamp=None)
    publisher_new.publish_markets([output])
    publisher_new.commit(validate=False)

    runner = CliRunner()
    result = runner.invoke(main, ["diff", str(old), str(new)])
    assert result.exit_code == 0
    assert "has_regression" in result.output
