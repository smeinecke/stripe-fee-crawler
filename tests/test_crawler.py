"""Tests for the crawler orchestration."""

from __future__ import annotations

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


async def test_crawl_market_unsupported() -> None:
    config = CrawlConfiguration(
        offline_fixtures={
            "https://stripe.com/pricing": "tests/fixtures/unsupported.html",
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
