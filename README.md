# Detection and Localization of Pediatric Wrist Fractures in Radiographs Using YOLO26

This repository contains the research infrastructure for a bachelor’s thesis on single-class wrist fracture detection in pediatric radiographs.

## Status

- Phase 2 dataset preparation is complete.
- Phase 3 infrastructure is being finalized.
- Real validation and benchmark execution are supported for the completed smoke checkpoints, but the repository still treats those results as pipeline verification rather than final science.

## Core controls

- Dataset: GRAZPEDWRI-DX derived YOLO dataset under `data/processed/yolo/`
- Splits: immutable patient-level train/val/test splits
- Primary models: YOLOv8n, YOLOv9t, YOLO26n
- Default image size: 640
- Seed: 42
- Validation metric: `metrics/mAP50-95(B)`
- Smoke-suite dry-run: `uv run python scripts/run_smoke_suite.py --dry-run`
- Smoke-suite execute: `uv run python scripts/run_smoke_suite.py --execute`
- Full-suite dry-run: `uv run python scripts/run_full_experiment_suite.py --dry-run`
- Full-suite execute: `uv run python scripts/run_full_experiment_suite.py --execute`
- Validation/benchmark suite dry-run: `uv run python scripts/run_validation_benchmark_suite.py --source-suite <path> --dry-run`
- Validation/benchmark suite execute: `uv run python scripts/run_validation_benchmark_suite.py --source-suite <path> --execute`
- Smoke metrics are pipeline checks only and are not scientific results.

## Safe commands

- `make experiment-validate`
- `make train-dry-run-yolov8n`
- `make train-dry-run-yolov9t`
- `make train-dry-run-yolo26n`
- `make smoke-suite-dry-run`
- `make smoke-suite`
- `make full-suite-dry-run`
- `make full-suite`
- `make gpu-preflight`
- `make transfer-manifest`
- `make transfer-verify`
- `make evaluate-dry-run`
- `make benchmark-dry-run`
- `make validation-benchmark-dry-run SOURCE_SUITE=<path>`
- `make validation-benchmark-suite SOURCE_SUITE=<path>`
- `make test`
- `make lint`
- `make format`

## Documentation

- [Experiment protocol](docs/experiment_protocol.md)
- [Decisions](docs/decisions.md)
- [References](docs/references.md)
- [RTX 4090 setup](docs/rtx4090_setup.md)

## Safety

- Full training requires explicit `--execute`.
- CPU execution is limited to development smoke paths.
- Test-set evaluation requires `--allow-test`.
- Model checkpoints are required explicitly for evaluation and benchmarking.
- Phase 5 full training requires validated calibration evidence and the frozen `configs/runs/full.yaml` protocol before execution.
