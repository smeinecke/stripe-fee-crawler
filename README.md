# Stripe Fee Crawler

A Python crawler that collects public Stripe merchant fee pages and publishes deterministic, schema-validated JSON artifacts. It feeds the separate `stripe-fee-data` repository.

## Highlights

- No browser or JavaScript execution: uses regular HTTP requests and `lxml` parsing.
- Discovers Stripe markets dynamically from the country selector on public pricing pages with a reviewed bootstrap fallback.
- Extracts pricing as a section tree, preserving headings, source text, and evidence.
- Renders rich-text fee phrases and normalizes localized percentages, money, and qualifiers.
- Derives calculation-ready fee rules conservatively; marks uncertain data as `unclassified` or `non_calculable`.
- Atomic, deterministic publication with regression guards and change reports.
- Comprehensive offline test suite plus optional live integration tests.

## Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

## Installation

```bash
uv sync
uv run stripe-fee-crawler --help
```

## Usage

```bash
# Discover all Stripe markets
uv run stripe-fee-crawler discover-markets

# Crawl a single market
uv run stripe-fee-crawler crawl-market DE

# Crawl all markets and publish to a data repository
uv run stripe-fee-crawler crawl \
  --output ../stripe-fee-data \
  --atomic \
  --fail-on-regression

# Validate generated JSON
uv run stripe-fee-crawler validate ../stripe-fee-data

# Inspect a local HTML fixture
uv run stripe-fee-crawler inspect tests/fixtures/de-pricing.html --page-kind pricing

# Compare two published datasets
uv run stripe-fee-crawler diff ../stripe-fee-data-old ../stripe-fee-data
```

## Development

```bash
uv run pytest
uv run make validate
```

Live integration tests are disabled by default. Run them with:

```bash
STRIPE_LIVE_TESTS=1 uv run pytest -m live
```

## Architecture

The crawler is split into small, testable modules under `src/stripe_fee_crawler/`:

- `http.py` — safe HTTP client with retries, backoff, domain allowlist, and conditional requests.
- `discovery.py` — market discovery from the country selector and fee-page URL validation.
- `extract.py` — high-level page extraction and source metadata.
- `components.py` — section tree extraction from headings and component markers.
- `rich_text.py` — text cleaning and link extraction.
- `pricing_tokens.py` — localized percentage, money, and qualifier tokenization.
- `normalize.py` — stable identifiers and normalization helpers.
- `classify.py` — conservative derivation of fee rules from extracted entries.
- `validation.py` — schema and Pydantic validation.
- `regression.py` — deterministic change reports and regression guards.
- `output.py` — deterministic, atomic publication.
- `crawler.py` — orchestration.
- `cli.py` — command-line interface.

## Output schema

Published files follow these schemas in the data repository:

- `json/{market}.json` — complete normalized and derived data for one market.
- `json/index.json` — index of published market files with content hashes.
- `json/core-fees.json` — consolidated calculation-ready rules across markets.
- `json/payment-methods.json` — cross-market payment method catalog.
- `meta/markets.json` — discovered and supported market manifest.
- `meta/unsupported-markets.json` — markets without public pricing pages.
- `meta/transient-failures.json` — transient failures from the latest run.
- `meta/schema-version.json` — schema version metadata.
- `schemas/*.schema.json` — JSON schemas for the generated files.

## Deterministic output

Canonical output is byte-for-byte stable when source content has not changed:

- Sorted keys and arrays at every level.
- Stable SHA-256 identifiers derived from content.
- Normalized Unicode and whitespace.
- Two-space JSON indentation with a trailing newline.
- `generated_at` is `null` unless an explicit reproducible timestamp is supplied.

## Security and rate limiting

- Redirects are restricted to Stripe domains.
- Response sizes, timeouts, and concurrency are limited.
- No credentials, cookies, or JavaScript execution are used.
- Logs are sanitized to exclude sensitive values.

## License and disclaimer

MIT License. See [LICENSE](LICENSE).

This project is unofficial and not affiliated with, maintained by, sponsored by, or endorsed by Stripe, Inc. "Stripe" is a trademark of Stripe, Inc. Source data remains subject to Stripe's terms and policies. Consumers are responsible for verifying fees applicable to their own accounts and contracts.
