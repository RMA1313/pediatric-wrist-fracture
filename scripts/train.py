from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from wrist_fracture.config import (
    ConfigError,
    ExperimentConfig,
    config_to_dict,
    load_config_bundle,
    validate_experiment_config,
)
from wrist_fracture.models.registry import describe_model_spec, resolve_model_spec
from wrist_fracture.provenance import (
    collect_environment_report,
    dependency_lock_hash,
    git_commit,
    git_dirty,
    sha256_file,
    to_jsonable,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def build_run_id(cfg: ExperimentConfig) -> str:
    return (
        cfg.run.run_id
        or f"{cfg.run.name}-{_now().replace(':', '').replace('-', '')}-{uuid.uuid4().hex[:8]}"
    )


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config_bundle(
        args.config,
        model_path=args.model_config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )


def run_root(cfg: ExperimentConfig, run_id: str) -> Path:
    return cfg.run.output_root / cfg.model.family / run_id


def ensure_unique_run_dir(path: Path, *, resume: bool = False) -> None:
    if path.exists():
        if resume and (path / "completed.marker").exists():
            raise ConfigError(f"cannot resume completed run: {path}")
        if not resume:
            raise ConfigError(f"run directory already exists: {path}")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def persist_run_metadata(root: Path, cfg: ExperimentConfig, args: argparse.Namespace) -> None:
    env = collect_environment_report(Path.cwd())
    spec = resolve_model_spec(cfg.model)
    payload = {
        "timestamp_utc": _now(),
        "git_commit": git_commit(Path.cwd()),
        "git_dirty": git_dirty(Path.cwd()),
        "dependency_lock_sha256": dependency_lock_hash(Path.cwd()),
        "dataset_yaml_sha256": sha256_file(cfg.dataset_yaml) if cfg.dataset_yaml.exists() else None,
        "dataset_split_yaml_sha256": sha256_file(cfg.dataset_split_yaml)
        if cfg.dataset_split_yaml and cfg.dataset_split_yaml.exists()
        else None,
        "command": sys.argv,
        "model": describe_model_spec(spec, imgsz=cfg.image_size),
        "environment": to_jsonable(env),
        "config": config_to_dict(cfg),
    }
    write_atomic(
        root / "resolved_config.yaml",
        json.dumps(config_to_dict(cfg), indent=2, sort_keys=True),
    )
    write_atomic(root / "environment.json", json.dumps(to_jsonable(env), indent=2, sort_keys=True))
    write_atomic(root / "provenance.json", json.dumps(payload, indent=2, sort_keys=True))
    write_atomic(root / "command.txt", " ".join(sys.argv))
    for folder in ["logs", "checkpoints", "raw", "metrics", "figures", "benchmarks"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "started.marker").write_text(_now(), encoding="utf-8")


def finalize_run(root: Path, success: bool) -> None:
    marker = root / ("completed.marker" if success else "interrupted.marker")
    marker.write_text(_now(), encoding="utf-8")


def dry_plan(cfg: ExperimentConfig, run_id: str) -> dict[str, object]:
    spec = resolve_model_spec(cfg.model)
    return {
        "run_id": run_id,
        "run_root": str(run_root(cfg, run_id)),
        "model": describe_model_spec(spec, imgsz=cfg.image_size),
        "config": config_to_dict(cfg),
        "python": sys.version,
    }


def _cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _cuda_device_exists(device: str) -> bool:
    if not device.startswith("cuda"):
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        if device == "cuda":
            return torch.cuda.device_count() > 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        else:
            index = 0
        return index < torch.cuda.device_count()
    except Exception:
        return False


def _normalize_device(device: str) -> str:
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    if device == "cuda":
        return "0"
    return device


def _maybe_copy_or_link(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)
    return dst


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _collect_checkpoint_paths(save_dir: Path) -> dict[str, Path | None]:
    weights = save_dir / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    return {
        "best": best if best.exists() else None,
        "last": last if last.exists() else None,
    }


def _normalize_history(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            parsed = _to_float(value)
            item[key] = parsed if parsed is not None else value
        normalized.append(item)
    return normalized


def _metrics_from_history(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    final = rows[-1]
    best = max(
        rows,
        key=lambda row: _to_float(row.get("metrics/mAP50-95(B)")) or float("-inf"),
    )
    return {
        "final_precision": _to_float(final.get("metrics/precision(B)")),
        "final_recall": _to_float(final.get("metrics/recall(B)")),
        "final_map50": _to_float(final.get("metrics/mAP50(B)")),
        "final_map50_95": _to_float(final.get("metrics/mAP50-95(B)")),
        "best_epoch": int(_to_float(best.get("epoch")) or 0),
        "best_map50_95": _to_float(best.get("metrics/mAP50-95(B)")),
    }


def _write_validation_json(root: Path, payload: dict[str, Any]) -> None:
    write_atomic(
        root / "metrics" / "validation.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )


def _write_run_summary(
    root: Path,
    *,
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    started_at: str,
    ended_at: str,
    duration_seconds: float,
    save_dir: Path,
    history_rows: list[dict[str, Any]],
    checkpoints: dict[str, Path | None],
    gpu_peak_memory_bytes: int | None,
) -> None:
    metrics = _metrics_from_history(history_rows)
    payload = {
        "run_status": "completed",
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
        "gpu_peak_memory_bytes": gpu_peak_memory_bytes,
        "checkpoint_paths": {
            "best": str(checkpoints["best"]) if checkpoints["best"] else None,
            "last": str(checkpoints["last"]) if checkpoints["last"] else None,
        },
        "ultralytics_save_dir": str(save_dir),
        "config": config_to_dict(cfg),
        "execute": args.execute,
        "smoke": args.smoke,
        **metrics,
    }
    write_atomic(
        root / "metrics" / "run_summary.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )


def _execute_training(cfg: ExperimentConfig, root: Path) -> None:
    raise NotImplementedError("use _execute_training_with_args")


def _execute_training_with_args(
    cfg: ExperimentConfig, root: Path, args: argparse.Namespace
) -> None:
    from ultralytics import YOLO

    started_at = _now()
    start_perf = perf_counter()
    gpu_peak_memory_bytes: int | None = None
    device = _normalize_device(cfg.hardware.device)
    spec = resolve_model_spec(cfg.model)
    model = YOLO(spec.checkpoint)
    train_kwargs: dict[str, Any] = {
        "data": str(cfg.dataset_yaml),
        "imgsz": cfg.image_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "batch": cfg.batch_size,
        "workers": cfg.hardware.workers,
        "device": device,
        "amp": cfg.hardware.amp,
        "seed": cfg.seed,
        "deterministic": cfg.hardware.deterministic,
        "optimizer": cfg.optimizer,
        "lr0": cfg.lr0,
        "lrf": cfg.lrf,
        "weight_decay": cfg.weight_decay,
        "cache": cfg.hardware.cache,
        "save_period": cfg.run.save_period,
        "project": str(root / "raw"),
        "name": "train",
        "exist_ok": True,
        "pretrained": cfg.pretrained,
        "plots": True,
        "val": True,
        "save_json": cfg.save_json,
        "resume": bool(cfg.run.resume),
    }
    train_kwargs.update(cfg.augmentation)
    if cfg.resume_checkpoint is not None:
        train_kwargs["resume"] = str(cfg.resume_checkpoint)
    if cfg.run.validation_split != "val":
        raise ConfigError("validation split must remain val during training")
    if cfg.run.test_split == "val":
        raise ConfigError("test split must not be used during training")
    if cfg.hardware.device == "cpu" and not cfg.hardware.allow_cpu_training:
        raise ConfigError("CPU full training is disabled")
    if cfg.run.resume and cfg.resume_checkpoint is None:
        raise ConfigError("safe resume validation failed")
    if cfg.run.resume and not cfg.resume_checkpoint.exists():
        raise ConfigError("safe resume validation failed")
    if device == "cpu" and not cfg.hardware.allow_cpu_training:
        raise ConfigError("no CPU full training")
    if device == "cpu" and cfg.epochs > 1:
        raise ConfigError("no CPU full training")
    try:
        results = model.train(**train_kwargs)
        trainer = getattr(model, "trainer", None)
        save_dir = Path(getattr(trainer, "save_dir", root / "raw" / "train"))
        history_src = save_dir / "results.csv"
        history_rows = (
            _normalize_history(_read_csv_rows(history_src)) if history_src.exists() else []
        )
        _write_csv_rows(root / "metrics" / "history.csv", history_rows)
        metrics = _metrics_from_history(history_rows)
        checkpoints = _collect_checkpoint_paths(save_dir)
        best_src = checkpoints["best"]
        last_src = checkpoints["last"]
        best_dst = root / "checkpoints" / "best.pt"
        last_dst = root / "checkpoints" / "last.pt"
        actual_best = _maybe_copy_or_link(best_src, best_dst) if best_src else None
        actual_last = _maybe_copy_or_link(last_src, last_dst) if last_src else None
        if actual_best is None or actual_last is None:
            raise ConfigError("missing expected Ultralytics checkpoints")
        gpu_peak_memory_bytes = None
        try:
            import torch

            if torch.cuda.is_available() and device != "cpu":
                gpu_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            gpu_peak_memory_bytes = None
        validation_payload = {
            "precision": metrics.get("final_precision"),
            "recall": metrics.get("final_recall"),
            "map50": metrics.get("final_map50"),
            "map50_95": metrics.get("final_map50_95"),
            "best_epoch": metrics.get("best_epoch"),
            "best_map50_95": metrics.get("best_map50_95"),
            "raw_results": to_jsonable(getattr(results, "__dict__", {})),
        }
        _write_validation_json(root, validation_payload)
        ended_at = _now()
        duration_seconds = perf_counter() - start_perf
        _write_run_summary(
            root,
            cfg=cfg,
            args=args,
            started_at=started_at,
            ended_at=ended_at,
            duration_seconds=duration_seconds,
            save_dir=save_dir,
            history_rows=history_rows,
            checkpoints={"best": actual_best, "last": actual_last},
            gpu_peak_memory_bytes=gpu_peak_memory_bytes,
        )
        (root / "completed.marker").write_text(_now(), encoding="utf-8")
    except Exception:
        (root / "interrupted.marker").write_text(_now(), encoding="utf-8")
        raise


SMOKE_SAFETY_CAPS = {
    "image_size": 320,
    "epochs": 1,
    "batch_size": 4,
    "patience": 1,
    "run.repeated_runs": 1,
}


def _validate_smoke_caps(cfg: ExperimentConfig) -> list[str]:
    errors: list[str] = []
    if cfg.image_size > SMOKE_SAFETY_CAPS["image_size"]:
        errors.append("smoke image_size exceeds safety cap")
    if cfg.epochs > SMOKE_SAFETY_CAPS["epochs"]:
        errors.append("smoke epochs exceeds safety cap")
    if cfg.batch_size > SMOKE_SAFETY_CAPS["batch_size"]:
        errors.append("smoke batch_size exceeds safety cap")
    if cfg.patience > SMOKE_SAFETY_CAPS["patience"]:
        errors.append("smoke patience exceeds safety cap")
    if cfg.run.repeated_runs > SMOKE_SAFETY_CAPS["run.repeated_runs"]:
        errors.append("smoke repeated_runs exceeds safety cap")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-cpu-smoke", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    args = parser.parse_args()

    cfg = resolve_config(args)
    if args.smoke:
        cfg = ExperimentConfig(
            dataset_yaml=cfg.dataset_yaml,
            dataset_split_yaml=cfg.dataset_split_yaml,
            model=cfg.model,
            hardware=cfg.hardware,
            run=cfg.run,
            image_size=cfg.image_size,
            epochs=cfg.epochs,
            patience=cfg.patience,
            seed=cfg.seed,
            pretrained=cfg.pretrained,
            optimizer=cfg.optimizer,
            lr0=cfg.lr0,
            lrf=cfg.lrf,
            weight_decay=cfg.weight_decay,
            augmentation=cfg.augmentation,
            resume_checkpoint=cfg.resume_checkpoint,
            save_json=cfg.save_json,
            batch_size=cfg.batch_size,
            extra=cfg.extra,
        )
    if not args.execute and not (args.preflight or args.dry_run):
        raise ConfigError("full training requires explicit --execute")
    gpu_ready = _cuda_is_available() if args.execute else None
    if args.print_resolved_config:
        print(json.dumps(config_to_dict(cfg), indent=2, sort_keys=True))
    errors = validate_experiment_config(
        cfg,
        dry_run=not args.execute,
        allow_cpu_smoke=args.allow_cpu_smoke and args.smoke,
    )
    if args.smoke:
        errors.extend(_validate_smoke_caps(cfg))
    if errors:
        raise ConfigError("; ".join(errors))
    run_id = build_run_id(cfg)
    root = run_root(cfg, run_id)
    if args.resume and not root.exists():
        raise ConfigError("resume requested but run directory does not exist")
    ensure_unique_run_dir(root, resume=args.resume)
    if cfg.hardware.device.startswith("cuda") and args.execute:
        if not gpu_ready:
            raise ConfigError("torch.cuda.is_available() is false")
        if not _cuda_device_exists(cfg.hardware.device):
            raise ConfigError("requested CUDA device does not exist")
    if args.preflight or args.dry_run or not args.execute:
        print(json.dumps(dry_plan(cfg, run_id), indent=2, sort_keys=True))
        return
    persist_run_metadata(root, cfg, args)
    try:
        _execute_training_with_args(cfg, root, args)
    except Exception:
        finalize_run(root, success=False)
        raise


if __name__ == "__main__":
    main()
