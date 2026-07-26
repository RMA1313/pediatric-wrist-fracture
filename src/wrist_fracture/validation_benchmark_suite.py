from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from wrist_fracture.config import ConfigError, load_config_bundle, validate_experiment_config
from wrist_fracture.provenance import (
    collect_environment_report,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)

SCHEMA_VERSION = 1
MODEL_ORDER = ("yolov8", "yolov9", "yolo26")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _json_safe(value: Any) -> Any:
    from scripts.train import _normalize_json_value

    return _normalize_json_value(value)


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        parsed = float(value)
        return None if math.isnan(parsed) or math.isinf(parsed) else parsed
    except Exception:
        return None


def _sha256_if_exists(path: Path | None) -> str | None:
    return sha256_file(path) if path and path.exists() else None


def _is_finite_number(value: Any) -> bool:
    parsed = _to_float(value)
    return parsed is not None and math.isfinite(parsed)


def _count_params(model: Any) -> dict[str, int | None]:
    total = trainable = None
    try:
        params = list(model.parameters())
        total = int(sum(p.numel() for p in params))
        trainable = int(sum(p.numel() for p in params if p.requires_grad))
    except Exception:
        pass
    return {"total": total, "trainable": trainable}


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = (len(ordered) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)


def _summary_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        keys = ["mean", "median", "std", "min", "max", "p90", "p95", "p99", "throughput"]
        return {k: None for k in keys}
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    total = sum(values)
    return {
        "mean": mean,
        "median": _percentile(values, 0.5),
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "p90": _percentile(values, 0.9),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "throughput": len(values) / total if total > 0 else None,
    }


def deterministic_sample_manifest(images: list[Path], samples: int) -> list[Path]:
    ordered = sorted(images, key=lambda path: path.as_posix().lower())
    return ordered[: min(samples, len(ordered))]


def validate_required_numeric_fields(payload: dict[str, Any], fields: Iterable[str]) -> list[str]:
    issues: list[str] = []
    for field in fields:
        value = payload.get(field)
        if value is not None and not _is_finite_number(value):
            issues.append(f"{field} must be finite or null")
    return issues


def collect_optional_artifacts(root: Path, names: Iterable[str]) -> dict[str, str | None]:
    collected: dict[str, str | None] = {}
    for name in names:
        matches = list(root.glob(name))
        collected[name] = str(matches[0]) if matches else None
    return collected


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _complexity_from_model(model: Any, checkpoint: Path, imgsz: int) -> dict[str, Any]:
    complexity = {
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.exists() else None,
        "input_image_size": imgsz,
        "total_parameters": None,
        "trainable_parameters": None,
        "flops_g": None,
        "flops_reason": None,
        "model_memory_bytes": None,
        "official_checkpoint_name": checkpoint.name,
    }
    try:
        info = getattr(model, "model", None)
        if info is not None:
            counts = _count_params(info)
            complexity["total_parameters"] = counts["total"]
            complexity["trainable_parameters"] = counts["trainable"]
            if hasattr(info, "info"):
                try:
                    complexity["flops_g"] = _to_float(info.info(verbose=False, imgsz=imgsz)[2])
                except Exception as exc:
                    complexity["flops_reason"] = str(exc)
    except Exception as exc:
        complexity["flops_reason"] = str(exc)
    return complexity


def _safe_json_write(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(_json_safe(payload), indent=2, sort_keys=True))


@dataclass
class SuiteContext:
    root: Path
    suite_id: str
    source_suite: Path
    out_root: Path


def discover_source_runs(source_suite: Path) -> list[dict[str, Any]]:
    if not source_suite.exists():
        raise ConfigError(f"source suite not found: {source_suite}")
    summary_path = source_suite / "suite_summary.json"
    if not summary_path.exists():
        raise ConfigError(f"missing suite summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    runs = payload.get("models", [])
    if not isinstance(runs, list) or not runs:
        raise ConfigError("source suite has no model runs")
    return runs


def select_runs(runs: list[dict[str, Any]], models: Iterable[str]) -> list[dict[str, Any]]:
    requested = set(models)
    selected = [run for run in runs if run.get("model_family") in requested]
    if len(selected) != len(requested):
        found = {run.get("model_family") for run in selected}
        missing = sorted(requested - found)
        raise ConfigError(f"missing requested model runs: {', '.join(missing)}")
    return sorted(selected, key=lambda run: MODEL_ORDER.index(run["model_family"]))


def evaluate_checkpoint(
    *,
    checkpoint: Path,
    cfg_path: Path,
    model_cfg: Path | None,
    hardware_cfg: Path | None,
    run_cfg: Path | None,
    split: str,
    execute: bool,
    out_dir: Path,
    allow_test: bool = False,
) -> dict[str, Any]:
    from scripts.train import _normalize_device
    from ultralytics import YOLO

    if not execute:
        return {"planned": True, "checkpoint": str(checkpoint), "split": split}
    cfg = load_config_bundle(
        cfg_path,
        model_path=model_cfg,
        hardware_path=hardware_cfg,
        run_path=run_cfg,
    )
    errors = validate_experiment_config(cfg, dry_run=False)
    if errors:
        raise ConfigError("; ".join(errors))
    if split == "test" and not allow_test:
        raise ConfigError("test evaluation requires --allow-test")
    if split not in {"val", "test"}:
        raise ConfigError(f"unsupported split: {split}")
    if cfg.model.family not in checkpoint.as_posix():
        raise ConfigError("checkpoint/config model-family mismatch")
    out_dir.mkdir(parents=True, exist_ok=False)
    resolved = {
        "dataset_yaml": str(cfg.dataset_yaml),
        "dataset_yaml_sha256": _sha256_if_exists(cfg.dataset_yaml),
        "split": split,
        "imgsz": cfg.image_size,
        "batch": cfg.batch_size,
        "workers": cfg.hardware.workers,
        "device": cfg.hardware.device,
        "amp": cfg.hardware.amp,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "plots": True,
        "save_json": True,
        "seed": cfg.seed,
    }
    _safe_json_write(out_dir / "resolved_config.yaml", resolved)
    _safe_json_write(
        out_dir / "environment.json",
        to_jsonable(collect_environment_report(Path.cwd())),
    )
    _safe_json_write(
        out_dir / "provenance.json",
        {
            "git_commit": git_commit(Path.cwd()),
            "git_dirty": git_dirty(Path.cwd()),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "command": " ".join(os.sys.argv),
        },
    )
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "curves").mkdir(exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "raw").mkdir(exist_ok=True)
    model = YOLO(str(checkpoint))
    eval_start = time.perf_counter()
    results = model.val(
        data=str(cfg.dataset_yaml),
        split=split,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        workers=cfg.hardware.workers,
        device=_normalize_device(cfg.hardware.device),
        amp=cfg.hardware.amp,
        conf=resolved["conf"],
        iou=resolved["iou"],
        max_det=resolved["max_det"],
        plots=True,
        save_json=True,
        project=str(out_dir / "raw"),
        name="val",
        exist_ok=True,
        verbose=False,
        seed=cfg.seed,
    )
    eval_duration = perf_counter() - eval_start
    raw_dir = out_dir / "raw" / "val"
    artifacts = collect_optional_artifacts(
        raw_dir,
        [
            "PR_curve.png",
            "F1_curve.png",
            "P_curve.png",
            "R_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
            "predictions.json",
        ],
    )
    metrics_dict = getattr(results, "results_dict", {}) or {}
    speed = getattr(results, "speed", {}) or {}
    precision = _to_float(metrics_dict.get("metrics/precision(B)") or metrics_dict.get("precision"))
    recall = _to_float(metrics_dict.get("metrics/recall(B)") or metrics_dict.get("recall"))
    map50 = _to_float(metrics_dict.get("metrics/mAP50(B)") or metrics_dict.get("map50"))
    map50_95 = _to_float(metrics_dict.get("metrics/mAP50-95(B)") or metrics_dict.get("map"))
    f1 = None
    if precision is not None and recall is not None and (precision + recall):
        f1 = 2 * precision * recall / (precision + recall)
    per_class_ap = _json_safe(getattr(results, "maps", None))
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "model_family": cfg.model.family,
        "split": split,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_yaml": str(cfg.dataset_yaml),
        "dataset_yaml_sha256": _sha256_if_exists(cfg.dataset_yaml),
        "imgsz": cfg.image_size,
        "batch_size": cfg.batch_size,
        "workers": cfg.hardware.workers,
        "device": cfg.hardware.device,
        "amp": cfg.hardware.amp,
        "conf": resolved["conf"],
        "iou": resolved["iou"],
        "max_det": resolved["max_det"],
        "ultralytics_version": __import__("ultralytics").__version__,
        "image_count": len(getattr(results, "files", []) or []),
        "positive_count": _to_float(metrics_dict.get("nt_per_class", [None])[0])
        if isinstance(metrics_dict.get("nt_per_class"), list)
        else None,
        "background_count": None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_derived": f1 is not None,
        "ap_fracture": _to_float(per_class_ap[0])
        if isinstance(per_class_ap, list) and per_class_ap
        else None,
        "map50": map50,
        "map50_95": map50_95,
        "fitness": _to_float(getattr(results, "fitness", None)),
        "preprocess_time_ms": _to_float(speed.get("preprocess")),
        "inference_time_ms": _to_float(speed.get("inference")),
        "loss_time_ms": _to_float(speed.get("loss")),
        "postprocess_time_ms": _to_float(speed.get("postprocess")),
        "validation_duration_seconds": eval_duration,
        "artifacts": artifacts,
        "raw_results": _json_safe(results),
    }
    metrics["validation_duration_seconds"] = eval_duration
    metrics["required_field_issues"] = validate_required_numeric_fields(
        metrics,
        [
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95",
            "fitness",
            "validation_duration_seconds",
        ],
    )
    _safe_json_write(out_dir / "metrics.json", metrics)
    _copy_tree(raw_dir, out_dir / "raw")
    (out_dir / "completed.marker").write_text(_now(), encoding="utf-8")
    return {"metrics": metrics, "raw_dir": str(raw_dir), "duration_seconds": eval_duration}


def benchmark_checkpoint(
    *,
    checkpoint: Path,
    images: list[Path],
    cfg_path: Path,
    model_cfg: Path | None,
    hardware_cfg: Path | None,
    run_cfg: Path | None,
    warmup: int,
    samples: int,
    batch_size: int,
    execute: bool,
    out_dir: Path,
) -> dict[str, Any]:
    if not execute:
        return {"planned": True, "checkpoint": str(checkpoint), "samples": samples}
    cfg = load_config_bundle(
        cfg_path,
        model_path=model_cfg,
        hardware_path=hardware_cfg,
        run_path=run_cfg,
    )
    if batch_size != 1:
        raise ConfigError("benchmark batch size must be 1 for controlled protocol")
    import cv2
    import torch
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    complexity = _complexity_from_model(model, checkpoint, cfg.image_size)
    timings = {"end_to_end": [], "inference": []}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        _ = model.predict(
            str(images[0]),
            imgsz=cfg.image_size,
            device="0",
            verbose=False,
            stream=False,
        )
    torch.cuda.synchronize()
    start_mem = torch.cuda.max_memory_allocated()
    selected = deterministic_sample_manifest(images, samples)
    for img in selected:
        t0 = perf_counter()
        frame = cv2.imread(str(img))
        t1 = perf_counter()
        _ = model.predict(frame, imgsz=cfg.image_size, device="0", verbose=False, stream=False)
        torch.cuda.synchronize()
        t2 = perf_counter()
        timings["end_to_end"].append(t2 - t0)
        timings["inference"].append(t2 - t1)
    stats = _summary_stats(timings["end_to_end"])
    reserve = int(torch.cuda.max_memory_reserved())
    alloc = int(torch.cuda.max_memory_allocated())
    _safe_json_write(
        out_dir / "benchmark.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "sample_manifest": [str(path) for path in selected],
            "samples": len(timings["end_to_end"]),
            "warmup": warmup,
            "batch_size": batch_size,
            "latency_seconds": stats,
            "end_to_end_latency_seconds": timings["end_to_end"],
            "inference_only_latency_seconds": timings["inference"],
            "peak_allocated_bytes": alloc,
            "peak_reserved_bytes": reserve,
            "steady_allocated_bytes": int(start_mem),
            "architecture": {
                "nms_based": not cfg.model.nms_free_default,
                "native_end_to_end": cfg.model.nms_free_default,
                "ultralytics_applied_nms": not cfg.model.nms_free_default,
            },
            "complexity": complexity,
        },
    )
    (out_dir / "completed.marker").write_text(_now(), encoding="utf-8")
    return {
        "latency": stats,
        "complexity": complexity,
        "selected_images": [str(p) for p in selected],
    }
