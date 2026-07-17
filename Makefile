# Makefile for stripe-fee-crawler

.PHONY: all format check validate test test-unit test-live regenerate regenerate-strict help

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
	uv run stripe-fee-crawler crawl --output .. --max-workers 8 --timeout 20

regenerate-strict:
	uv run stripe-fee-crawler crawl --output .. --max-workers 8 --timeout 20 --fail-on-regression

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
	@echo "  validate      - Run all validation checks"
	@echo "  help          - Show this help message"
