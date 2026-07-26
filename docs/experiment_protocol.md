# Experiment Protocol

## Preliminary plan

1. Verify dataset structure and label format.
2. Create reproducible train/validation/test splits.
3. Fine-tune pretrained YOLO26, YOLOv8, and YOLOv9 implementations only after
   the final dataset audit passes.
4. Evaluate on the held-out test split using detection metrics and complexity
   estimates.

## Unresolved items

- Final YOLO26 size variants
- Exact YOLOv9 implementation
- Final augmentation values
- Final split ratios are fixed at approximately 70/15/15 after the verified
  patient-level split
- Full-training hardware constraints

## Baseline constraints

- Single-class detection only
- No architecture reimplementation from scratch
- No training until dataset preparation is verified
