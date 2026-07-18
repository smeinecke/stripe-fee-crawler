"""Tests for the crawler orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from stripe_fee_crawler.crawler import StripeCrawler
from stripe_fee_crawler.discovery import build_market_from_code
from stripe_fee_crawler.models import CrawlConfiguration, Market
from stripe_fee_crawler.output import OutputPublisher


def _minimal_market_output(market: Market) -> dict:
    from stripe_fee_crawler.models import MarketOutput, Source

    return MarketOutput(
        market=market,
        sources=[Source(requested_url=f"{market.url_prefix}/pricing")],
        derivation_status="partial",
    ).model_dump()


async def test_crawl_market_with_offline_fixture() -> None:
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/en-de/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-de/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
        request_delay=0.0,
    )
    market = build_market_from_code("DE", status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    assert output.market.account_country == "DE"
    assert len(output.entries) > 0
    assert output.derivation_status in {"complete", "partial"}


async def test_crawl_all_publishes(tmp_path: Path) -> None:
    config = CrawlConfiguration(
        markets=["DE"],
        offline_fixtures={
            "https://stripe.com/en-de/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-de/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
        request_delay=0.0,
    )
    async with StripeCrawler(config) as crawler:
        outputs, unsupported = await crawler.crawl_all()
        report = await crawler.publish(outputs, unsupported, tmp_path, atomic=True, fail_on_regression=False)
    assert report.exit_code == 0
    assert (tmp_path / "json" / "DE.json").exists()
    assert (tmp_path / "json" / "index.json").exists()
    assert (tmp_path / "change-report.json").exists()
    assert (tmp_path / "meta" / "crawl-report.json").exists()
    assert report.changed is True


async def test_crawl_all_second_run_unchanged(tmp_path: Path) -> None:
    config = CrawlConfiguration(
        markets=["DE"],
        offline_fixtures={
            "https://stripe.com/en-de/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-de/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
        request_delay=0.0,
    )
    async with StripeCrawler(config) as crawler:
        outputs, unsupported = await crawler.crawl_all()
        report1 = await crawler.publish(outputs, unsupported, tmp_path, atomic=True, fail_on_regression=False)
        report2 = await crawler.publish(outputs, unsupported, tmp_path, atomic=True, fail_on_regression=False)
    assert report1.changed is True
    assert report2.changed is False
    crawl_report_path = tmp_path / "meta" / "crawl-report.json"
    assert crawl_report_path.exists()
    crawl_report = json.loads(crawl_report_path.read_text(encoding="utf-8"))
    assert crawl_report["changed"] is False


async def test_crawl_market_unsupported() -> None:
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/en-us/pricing": "tests/fixtures/unsupported.html",
            "https://stripe.com/en-us/pricing/local-payment-methods": "tests/fixtures/unsupported.html",
        },
        request_delay=0.0,
    )
    market = build_market_from_code("US", status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    assert output.unsupported_reason is not None


async def test_publish_with_regression(tmp_path: Path) -> None:
    from stripe_fee_crawler.models import FeeRule, MarketOutput, Source

    market = build_market_from_code("DE", status="supported")
    output = MarketOutput(
        market=market,
        sources=[Source(requested_url="https://stripe.com/en-de/pricing")],
        derived_rules=[
            FeeRule(
                rule_id="de-card",
                name="card_payment",
                provider="stripe",
                unit="per_transaction",
                exactness="exact",
                behavior="additive",
                classification_status="classified",
                confidence=0.9,
                percentage="2.9",
            )
        ],
        derivation_status="complete",
    )
    publisher = OutputPublisher(tmp_path, timestamp=None)
    _, staging = publisher.publish([output], [market], [], [])
    publisher.commit(staging, validate=False)

    # Second publish with a large percentage change.
    output2 = output.model_copy(
        deep=True,
        update={
            "derived_rules": [
                FeeRule(
                    rule_id="de-card",
                    name="card_payment",
                    provider="stripe",
                    unit="per_transaction",
                    exactness="exact",
                    behavior="additive",
                    classification_status="classified",
                    confidence=0.9,
                    percentage="10.0",
                )
            ]
        },
    )
    publisher2 = OutputPublisher(tmp_path / "new", timestamp=None)
    _, staging2 = publisher2.publish([output2], [market], [], [])
    publisher2.commit(staging2, validate=False)

    from stripe_fee_crawler.regression import check_regression

    report = check_regression(tmp_path, tmp_path / "new")
    assert report.has_regression
    assert any(c.kind == "large_percentage_change" for c in report.changes)


async def test_crawl_market_rejects_cross_market_response() -> None:
    """A US request that is answered with a DE page must be rejected."""
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/en-us/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-us/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
        request_delay=0.0,
    )
    market = build_market_from_code("US", status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    assert output.transient_failure
    assert output.derivation_status == "failed"
    assert len(output.entries) == 0


async def test_crawl_market_us_rates() -> None:
    """US fixtures must produce US online card and ACH direct debit rates."""
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/en-us/pricing": "tests/fixtures/us-pricing.html",
            "https://stripe.com/en-us/pricing/local-payment-methods": "tests/fixtures/us-lpm.html",
        },
        request_delay=0.0,
    )
    market = build_market_from_code("US", status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    assert output.market.account_country == "US"
    assert not output.transient_failure

    online_card = [
        r
        for r in output.derived_rules
        if r.product_id == "payments" and r.channel == "online" and r.percentage == "2.9"
    ]
    assert online_card, "expected US online domestic card fee 2.9%"
    assert online_card[0].fixed_currency == "USD"

    ach = [r for r in output.derived_rules if r.product_id == "ach_direct_debit"]
    assert ach, "expected ACH Direct Debit rule"
    assert ach[0].percentage == "0.8"
    assert any(c.type == "maximum_fee" and c.amount == "5.00" and c.currency == "USD" for c in ach[0].fee_components)


async def test_crawl_market_de_rates_eur() -> None:
    """DE fixtures must produce EUR-denominated domestic card fees."""
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/en-de/pricing": "tests/fixtures/de-pricing.html",
            "https://stripe.com/en-de/pricing/local-payment-methods": "tests/fixtures/de-lpm.html",
        },
        request_delay=0.0,
    )
    market = build_market_from_code("DE", status="supported")
    async with StripeCrawler(config) as crawler:
        output = await crawler.crawl_market(market)
    domestic = [r for r in output.derived_rules if r.product_id == "payments" and r.fixed_currency]
    assert domestic
    assert all(r.fixed_currency == "EUR" for r in domestic), "DE card fees must use EUR"
