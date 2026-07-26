# Data Layout

This directory is reserved for dataset-related files that must not be committed.

Planned structure:

- `data/raw/`: original downloaded archives and immutable source files.
- `data/interim/`: temporary files created while validating or converting the dataset.
- `data/processed/`: cleaned and normalized artifacts derived from the raw data.
- `data/splits/`: train, validation, and test split definitions and manifests.
- `data/downloads/`: temporary download cache and partial transfers.

Large images, annotations, caches, and derived training artifacts should stay outside version control.

For this project, the processed YOLO dataset is materialized under
`data/processed/yolo/` using hard links when the filesystem supports them.
Split manifests are stored in `data/splits/` as CSV files.
