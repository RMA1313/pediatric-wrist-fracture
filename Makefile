PYTHON ?= python

.PHONY: setup audit test lint format dataset-info dataset-download-dry-run

setup:
\tuv sync

audit:
\tuv run python scripts/audit_environment.py

test:
\tuv run pytest

lint:
\tuv run ruff check .

format:
\tuv run ruff format .

dataset-info:
\tuv run python scripts/download_dataset.py --metadata-only

dataset-download-dry-run:
\tuv run python scripts/download_dataset.py --dry-run

