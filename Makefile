PYTHON ?= python

.PHONY: setup audit test lint format \
	experiment-check experiment-validate \
	train-dry-run-yolov8n train-dry-run-yolov9t train-dry-run-yolo26n \
	evaluate-dry-run benchmark-dry-run gpu-preflight transfer-manifest transfer-verify \
	dataset-info dataset-download dataset-verify dataset-extract \
	dataset-inspect dataset-convert dataset-split dataset-validate \
	dataset-figures dataset-smoke dataset-prepare

setup:
	uv sync

audit:
	uv run python scripts/audit_environment.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

experiment-check:
	uv run python scripts/train.py --config configs/experiment.yaml --dry-run

experiment-validate:
	uv run python scripts/train.py --config configs/experiment.yaml --preflight

train-dry-run-yolov8n:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov8.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run-yolov9t:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolov9.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run-yolo26n:
	uv run python scripts/train.py --config configs/experiment.yaml --model-config configs/models/yolo26.yaml --hardware-config configs/hardware/cpu-dev.yaml --run-config configs/runs/smoke.yaml --dry-run --smoke

train-dry-run:
	uv run python scripts/train.py --config configs/experiment.yaml --dry-run

evaluate-dry-run:
	uv run python scripts/evaluate.py --config configs/experiment.yaml --checkpoint data/processed/yolo/dataset.yaml --dry-run

benchmark-dry-run:
	uv run python scripts/benchmark.py --config configs/experiment.yaml --checkpoint data/processed/yolo/dataset.yaml --dry-run

gpu-preflight:
	uv run python scripts/gpu_preflight.py

transfer-manifest:
	uv run python scripts/transfer_manifest.py

transfer-verify:
	uv run python scripts/transfer_manifest.py --verify

dataset-info:
	uv run python scripts/download_dataset.py --metadata-only

dataset-download:
	uv run python scripts/prepare_dataset.py download

dataset-verify:
	uv run python scripts/download_dataset.py --metadata-only

dataset-extract:
	uv run python scripts/prepare_dataset.py extract

dataset-inspect:
	uv run python scripts/prepare_dataset.py inspect --progress

dataset-convert:
	uv run python scripts/prepare_dataset.py convert --workers 8 --io-workers 16 --batch-size 32 --progress

dataset-split:
	uv run python scripts/prepare_dataset.py split

dataset-validate:
	uv run python scripts/prepare_dataset.py validate --io-workers 16 --batch-size 64 --progress

dataset-figures:
	uv run python scripts/prepare_dataset.py figures --progress

dataset-smoke:
	uv run python scripts/prepare_dataset.py smoke --max-batches 2

dataset-prepare:
	uv run python scripts/prepare_dataset.py prepare
