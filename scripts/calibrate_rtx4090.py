from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from wrist_fracture.calibration import (
    CALIBRATED_MODEL_ORDER,
    build_calibration_report,
    build_command,
    build_hardware_profile,
    build_recommended_full_config,
    build_resume_state,
    candidate_batches,
    flatten_candidate_rows,
    now_utc,
    update_run_config_batch,
    write_reports,
    write_resume_state,
)
from wrist_fracture.calibration_probe import run_bounded_training_probe
from wrist_fracture.config import ConfigError, load_config_bundle
from wrist_fracture.provenance import collect_environment_report, git_commit, git_dirty, to_jsonable


def _default_output_dir(root: Path) -> Path:
    return root / "outputs" / "calibration"


def _load_resume_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_candidate_batches(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    parsed = [int(item.strip()) for item in raw.split(",") if item.strip()]
    return candidate_batches(candidates=parsed)


def _resume_state_for_model(path: Path, model_family: str) -> dict[str, Any] | None:
    data = _load_resume_state(path)
    if not data:
        return None
    if data.get("model_family") in {None, model_family, "all"}:
        return data
    return None


def _candidate_plan(
    *,
    cfg,
    model_family: str,
    batches: list[int],
    out_dir: Path,
    resume_state: dict[str, Any] | None,
    force: bool,
) -> list[dict[str, Any]]:
    completed = {int(row["batch_size"]) for row in (resume_state or {}).get("results", [])}
    rows: list[dict[str, Any]] = []
    seen_oom = False
    for batch in batches:
        if seen_oom:
            rows.append(
                {
                    "model_family": model_family,
                    "batch_size": batch,
                    "status": "skipped_after_oom",
                    "status_detail": "skipped after OOM on smaller batch",
                }
            )
            continue
        if batch in completed and not force:
            rows.append(
                {
                    "model_family": model_family,
                    "batch_size": batch,
                    "status": "skipped",
                    "status_detail": "already completed",
                }
            )
            continue
        probe_cfg = dataclasses.replace(cfg, batch_size=batch)
        result = run_bounded_training_probe(probe_cfg, out_dir=out_dir, iterations=30)
        rows.append(result)
        if result.get("status") == "oom":
            seen_oom = True
    return rows


def _selected_common_batch(
    results_by_model: dict[str, list[dict[str, Any]]], fallback: int
) -> int | None:
    largest_stables = []
    for rows in results_by_model.values():
        stable = [int(row["batch_size"]) for row in rows if row.get("status") == "stable"]
        if not stable:
            return None
        largest_stables.append(max(stable))
    return min(largest_stables) if largest_stables else None


def run(args: argparse.Namespace) -> int:
    root = Path.cwd()
    base_cfg = load_config_bundle(
        args.config,
        hardware_path=args.hardware_config,
        run_path=args.run_config,
    )
    out_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    resume_state_path = out_dir / "resume_state.json"
    batches = (
        _parse_candidate_batches(getattr(args, "candidate_batches", None)) or candidate_batches()
    )
    target_models = [base_cfg.model.family] if args.model_config else list(CALIBRATED_MODEL_ORDER)
    per_model_rows: dict[str, list[dict[str, Any]]] = {}
    evidence_ready = False
    if args.execute:
        for model_family in target_models:
            model_cfg = load_config_bundle(
                args.config,
                model_path=Path(args.model_config)
                if args.model_config
                else Path(f"configs/models/{model_family}.yaml"),
                hardware_path=args.hardware_config,
                run_path=args.run_config,
            )
            model_resume = (
                _resume_state_for_model(resume_state_path, model_family) if args.resume else None
            )
            candidate_rows = _candidate_plan(
                cfg=model_cfg,
                model_family=model_family,
                batches=batches,
                out_dir=out_dir,
                resume_state=model_resume,
                force=bool(args.force),
            )
            per_model_rows[model_family] = candidate_rows
        recommended_batch = _selected_common_batch(per_model_rows, base_cfg.batch_size)
        evidence_ready = all(
            any(row["status"] == "stable" for row in rows) for rows in per_model_rows.values()
        )
        candidate_rows = flatten_candidate_rows(per_model_rows)
    else:
        for model_family in target_models:
            load_config_bundle(
                args.config,
                model_path=Path(args.model_config)
                if args.model_config
                else Path(f"configs/models/{model_family}.yaml"),
                hardware_path=args.hardware_config,
                run_path=args.run_config,
            )
            candidate_rows = [
                {
                    "model_family": model_family,
                    "batch_size": batch,
                    "status": "planned",
                    "status_detail": "dry-run only",
                }
                for batch in batches
            ]
            per_model_rows[model_family] = candidate_rows
        recommended_batch = None
        candidate_rows = flatten_candidate_rows(per_model_rows)
    report = build_calibration_report(
        cfg=base_cfg,
        model_family="all" if len(target_models) > 1 else target_models[0],
        candidates=candidate_rows,
        recommended_batch_size=recommended_batch,
        applied=bool(args.apply),
        resume_state=build_resume_state(
            model_family="all" if len(target_models) > 1 else target_models[0],
            candidate_order=batches,
            results=candidate_rows,
        ),
    )
    hardware_profile = build_hardware_profile(base_cfg)
    stage_timings = {
        "planning_seconds": 0.0,
        "selection_seconds": 0.0,
        "reporting_seconds": 0.0,
    }
    environment = {
        "timestamp_utc": now_utc(),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "environment": to_jsonable(collect_environment_report(root)),
    }
    commands = [
        build_command(
            model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
            hardware_path=Path(args.hardware_config or "configs/hardware/rtx4090.yaml"),
            run_path=Path(args.run_config or "configs/runs/full.yaml"),
            execute=False,
        ),
    ]
    recommended_config = build_recommended_full_config(
        base_cfg, batch_size=recommended_batch or base_cfg.batch_size
    )
    write_reports(
        out_dir,
        report=report,
        hardware_profile=hardware_profile,
        stage_timings=stage_timings,
        environment=environment,
        commands=commands,
        recommended_config=recommended_config,
    )
    write_resume_state(
        out_dir,
        build_resume_state(
            model_family=("all" if len(target_models) > 1 else target_models[0]),
            candidate_order=batches,
            results=candidate_rows,
        ),
    )
    (out_dir / "completed.marker").write_text(now_utc(), encoding="utf-8")
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))
    if args.apply:
        if not evidence_ready:
            raise ConfigError("apply requires complete real calibration evidence for the model")
        update_run_config_batch(
            root / "configs" / "runs" / "full.yaml", batch_size=recommended_batch
        )
        applied_cfg = load_config_bundle(
            args.config,
            model_path=Path(args.model_config or f"configs/models/{target_models[0]}.yaml"),
            hardware_path=args.hardware_config,
            run_path=root / "configs" / "runs" / "full.yaml",
        )
        if applied_cfg.batch_size != recommended_batch:
            raise ConfigError("applied calibration batch did not propagate into ExperimentConfig")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--hardware-config")
    parser.add_argument("--run-config")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--print-report", action="store_true")
    parser.add_argument("--candidate-batches")
    args = parser.parse_args()
    if args.execute and args.dry_run:
        raise ConfigError("--execute and --dry-run are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
