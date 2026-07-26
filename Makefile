PYTHON ?= python

.PHONY: setup audit test lint format \
	dataset-info dataset-download dataset-verify dataset-extract \
	dataset-inspect dataset-convert dataset-split dataset-validate \
	dataset-figures dataset-smoke dataset-prepare

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

dataset-download:
\tuv run python scripts/prepare_dataset.py download

dataset-verify:
\tuv run python scripts/download_dataset.py --metadata-only

dataset-extract:
\tuv run python scripts/prepare_dataset.py extract

dataset-inspect:
\tuv run python scripts/prepare_dataset.py inspect --progress

dataset-convert:
\tuv run python scripts/prepare_dataset.py convert --workers 8 --io-workers 16 --batch-size 32 --progress

dataset-split:
\tuv run python scripts/prepare_dataset.py split

dataset-validate:
\tuv run python scripts/prepare_dataset.py validate --io-workers 16 --batch-size 64 --progress

dataset-figures:
\tuv run python scripts/prepare_dataset.py figures --progress

dataset-smoke:
\tuv run python scripts/prepare_dataset.py smoke --max-batches 2

dataset-prepare:
\tuv run python scripts/prepare_dataset.py prepare
