from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

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
        "model": describe_model_spec(resolve_model_spec(cfg.model)),
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
        "model": describe_model_spec(spec),
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


def _execute_training(cfg: ExperimentConfig, root: Path) -> None:
    raise NotImplementedError(
        "Training execution is wired but intentionally not run in this phase."
    )


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
        _execute_training(cfg, root)
    except Exception:
        finalize_run(root, success=False)
        raise


if __name__ == "__main__":
    main()
