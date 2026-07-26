from __future__ import annotations

import argparse
import json
from pathlib import Path

from wrist_fracture.config import ConfigError, load_config_bundle, validate_experiment_config
from wrist_fracture.models.registry import resolve_model_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config_bundle(
        args.config,
        model_path=args.model_config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )
    if not Path(args.checkpoint).exists():
        raise ConfigError(f"checkpoint not found: {args.checkpoint}")
    errors = validate_experiment_config(cfg, dry_run=not args.execute, gpu_ready=False)
    if errors:
        raise ConfigError("; ".join(errors))
    spec = resolve_model_spec(cfg.model)
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model": {"family": spec.family, "checkpoint": spec.checkpoint, "scale": spec.scale},
        "dry_run": args.dry_run,
        "execute": args.execute,
        "protocol": {
            "batch_size": 1,
            "warmup_iterations": args.warmup,
            "measured_samples": args.samples,
            "metrics": [
                "preprocess_latency",
                "model_latency",
                "postprocess_latency",
                "total_latency",
                "mean",
                "median",
                "std",
                "p95",
                "throughput",
                "peak_gpu_memory",
            ],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.execute and not args.dry_run:
        raise NotImplementedError("Benchmark execution is intentionally disabled in this phase.")


if __name__ == "__main__":
    main()
