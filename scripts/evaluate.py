from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrist_fracture.config import ConfigError, load_config_bundle, validate_experiment_config
from wrist_fracture.models.registry import resolve_model_spec
from wrist_fracture.provenance import collect_environment_report, git_commit, git_dirty, to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_config_bundle(
        args.config,
        model_path=args.model_config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )
    if args.split == "test" and not args.allow_test:
        raise ConfigError("test evaluation requires --allow-test")
    if not Path(args.checkpoint).exists():
        raise ConfigError(f"checkpoint not found: {args.checkpoint}")
    if cfg.run.validation_split == "test":
        raise ConfigError("test split misuse for validation")
    errors = validate_experiment_config(cfg, dry_run=not args.execute, gpu_ready=False)
    if errors:
        raise ConfigError("; ".join(errors))
    spec = resolve_model_spec(cfg.model)
    env = collect_environment_report(Path.cwd())
    payload = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "model": {"family": spec.family, "checkpoint": spec.checkpoint, "scale": spec.scale},
        "dry_run": args.dry_run,
        "execute": args.execute,
        "environment": to_jsonable(env),
        "git_commit": git_commit(Path.cwd()),
        "git_dirty": git_dirty(Path.cwd()),
        "metrics": ["Precision", "Recall", "F1", "AP", "mAP@0.5", "mAP@0.5:0.95"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.execute and not args.dry_run:
        raise NotImplementedError("Evaluation execution is intentionally disabled in this phase.")


if __name__ == "__main__":
    main()
