# Makefile for stripe-fee-crawler

CACHE_DIR ?= $(if $(XDG_CACHE_HOME),$(XDG_CACHE_HOME),$(HOME)/.cache)/stripe-fee-crawler/http
CACHE_TTL_HOURS ?= 24

.PHONY: all format check validate test test-unit test-live regenerate regenerate-strict regenerate-refresh help

all: validate test-unit

format:
	uv run ruff format --check --diff .

reformat-ruff:
	uv run ruff format .

check:
	uv run ruff check .

fix-ruff:
	uv run ruff check . --fix

fix: reformat-ruff fix-ruff
	@echo "Updated code."

pyright:
	uv run pyright

bandit:
	uv run bandit -c pyproject.toml -r src

test:
	uv run pytest

test-unit:
	uv run pytest tests/ -m "not live"

test-live:
	uv run pytest tests/ -m live

regenerate:
	uv run stripe-fee-crawler crawl --output .. --max-workers 8 --timeout 20 --cache-dir "$(CACHE_DIR)" --cache-ttl-hours "$(CACHE_TTL_HOURS)"

regenerate-strict:
	uv run stripe-fee-crawler crawl --output .. --max-workers 8 --timeout 20 --fail-on-regression --cache-dir "$(CACHE_DIR)" --cache-ttl-hours "$(CACHE_TTL_HOURS)"

regenerate-refresh:
	uv run stripe-fee-crawler crawl --output .. --max-workers 8 --timeout 20 --fail-on-regression --cache-dir "$(CACHE_DIR)" --cache-ttl-hours "$(CACHE_TTL_HOURS)" --refresh-cache

validate: format check pyright bandit
	@echo "Validation passed."

help:
	@echo "Available targets:"
	@echo "  all           - Run validation and unit tests (default)"
	@echo "  format        - Check code formatting with ruff"
	@echo "  reformat-ruff - Format code with ruff"
	@echo "  check         - Run ruff linting"
	@echo "  fix-ruff      - Auto-fix ruff issues"
	@echo "  fix           - Run reformat-ruff and fix-ruff"
	@echo "  pyright       - Run type checking"
	@echo "  bandit        - Run security analysis"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests (no live network)"
	@echo "  test-live     - Run live integration tests"
	@echo "  regenerate    - Regenerate all market data"
	@echo "  regenerate-strict - Regenerate all market data and fail on regression"
	@echo "  regenerate-refresh - Regenerate all market data, force cache refresh"
	@echo "  validate      - Run all validation checks"
	@echo "  help          - Show this help message"
