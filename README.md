# Detection and Localization of Pediatric Wrist Fractures in Radiographs Using YOLO26

This repository contains the research infrastructure for a bachelor’s thesis on single-class wrist fracture detection in pediatric radiographs.

## Status

- Phase 2 dataset preparation is complete.
- Phase 3 infrastructure is being finalized.
- No training, inference, or benchmark run is executed in this repository phase.

## Core controls

- Dataset: GRAZPEDWRI-DX derived YOLO dataset under `data/processed/yolo/`
- Splits: immutable patient-level train/val/test splits
- Primary models: YOLOv8n, YOLOv9t, YOLO26n
- Default image size: 640
- Seed: 42
- Validation metric: `metrics/mAP50-95(B)`

## Safe commands

- `make experiment-validate`
- `make train-dry-run-yolov8n`
- `make train-dry-run-yolov9t`
- `make train-dry-run-yolo26n`
- `make gpu-preflight`
- `make transfer-manifest`
- `make transfer-verify`
- `make evaluate-dry-run`
- `make benchmark-dry-run`
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
