# Detection and Localization of Pediatric Wrist Fractures in Radiographs Using YOLO26

This repository is the research foundation for a bachelor's thesis project on single-class object detection of pediatric wrist fractures in radiographs.

## Purpose

The project will evaluate YOLO26 for fracture detection and localization on the public GRAZPEDWRI-DX dataset and compare it with YOLOv8 and YOLOv9 using reproducible experimental protocols.

## Medical disclaimer

This project is a research decision-support prototype only.
It must not be treated as a clinical system, medical device, treatment recommender, or substitute for radiologist judgment.

## Current phase

Phase 2 is focused on dataset preparation, auditability, and repository cleanup.
No full model training has been started.

## Planned models

- YOLO26
- YOLOv8
- YOLOv9

The exact YOLOv9 implementation remains unresolved until the official source choice is finalized.

## Dataset

Primary dataset source: GRAZPEDWRI-DX on Figshare.

- Official record: <https://doi.org/10.6084/m9.figshare.14825193.v2>

## Installation

1. Install `uv`.
2. Run `make setup`.

## Commands

- `make setup`
- `make audit`
- `make test`
- `make lint`
- `make dataset-info`
- `make dataset-download-dry-run`

## Repository structure

See `docs/decisions.md` and the directory tree in this repository.

## Next phases

1. Run controlled model training only after the final dataset audit passes.
2. Evaluate the held-out test split with the finalized experiment protocol.
3. Compare model families under the same data and validation conditions.

## Known limitations

- No training has been run yet.
- Dataset download is intentionally conservative until storage and metadata are verified.
- Some implementation details are intentionally left unresolved pending source verification.
