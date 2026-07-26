# Decisions Log

## 2026-07-25

- Adopted Python 3.11 as the preferred environment target.
- Chose `uv` for dependency management and reproducible environments.
- Selected a simple YAML-based configuration layer instead of a custom framework.
- Deferred all training until dataset metadata and split strategy are verified.
- Kept YOLOv9 implementation choice unresolved pending primary-source verification.
- Scoped the thesis as single-class object detection with class `fracture`.
- Adopted Pascal VOC XML as the authoritative annotation source for conversion,
  with Supervisely JSON retained for audit comparison only.
- Split the verified 20,327-image dataset at patient level with seed `42` and a
  fixed approximate 70/15/15 train/validation/test ratio.
- Materialized the processed YOLO dataset with hard links when supported by the
  filesystem, falling back to safe copying otherwise.
