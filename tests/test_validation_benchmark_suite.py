from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import benchmark, evaluate
from scripts import run_validation_benchmark_suite as suite

from wrist_fracture import runtime
from wrist_fracture.config import ConfigError
from wrist_fracture.validation_benchmark_suite import (
    _percentile,
    _summary_stats,
    resolve_checkpoint_path,
    select_runs,
)


def test_percentile_and_summary_stats():
    values = [1.0, 2.0, 3.0, 4.0]
    assert _percentile(values, 0.5) == 2.5
    stats = _summary_stats(values)
    assert stats["mean"] == 2.5
    assert stats["p95"] is not None
    assert stats["throughput"] == pytest.approx(1 / 2.5)


def test_select_runs_preserves_requested_order():
    runs = [
        {"model_family": "yolo26"},
        {"model_family": "yolov8"},
        {"model_family": "yolov9"},
    ]
    selected = select_runs(runs, ["yolov8", "yolov9", "yolo26"])
    assert [run["model_family"] for run in selected] == ["yolov8", "yolov9", "yolo26"]


def test_evaluate_refuses_test_without_allow_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        evaluate.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config="configs/models/yolov8.yaml",
            hardware_config="configs/hardware/rtx4090.yaml",
            run_config="configs/runs/smoke.yaml",
            checkpoint=str(checkpoint),
            split="test",
            allow_test=False,
            dry_run=False,
            execute=True,
            evaluation_id=None,
            output_dir=str(tmp_path / "out"),
            preflight=False,
        ),
    )
    with pytest.raises(ConfigError, match="test evaluation requires --allow-test"):
        evaluate.main()


def test_benchmark_requires_explicit_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    checkpoint = tmp_path / "yolov8n.pt"
    checkpoint.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        benchmark.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            config="configs/experiment.yaml",
            model_config=None,
            hardware_config=None,
            run_config=None,
            checkpoint=str(checkpoint),
            dry_run=False,
            execute=False,
            warmup=30,
            samples=300,
            batch_size=1,
            device="cuda:0",
            benchmark_id=None,
            output_dir=str(tmp_path / "bench"),
        ),
    )
    with pytest.raises(ConfigError, match="benchmark requires explicit --execute"):
        benchmark.main()


def test_suite_dry_run_refuses_execute_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "suite_summary.json").write_text(
        json.dumps(
            {
                "models": [
                    {"model_family": "yolov8"},
                    {"model_family": "yolov9"},
                    {"model_family": "yolo26"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            source_suite=str(source),
            dry_run=True,
            execute=False,
            suite_id="suite",
            models=None,
            skip_completed=False,
            continue_on_error=False,
            resume=False,
            force=False,
            warmup=30,
            samples=300,
            benchmark_batch_size=1,
            io_workers=4,
            print_commands=False,
        ),
    )
    assert suite.main() == 0


def test_suite_missing_model_run_rejected(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "suite_summary.json").write_text(
        json.dumps({"models": [{"model_family": "yolov8"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="missing requested model runs"):
        select_runs([{"model_family": "yolov8"}], ["yolov8", "yolov9"])


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", "cpu"),
        ("cuda", "0"),
        ("cuda:0", "0"),
        ("cuda:1", "1"),
        (0, "0"),
        (1, "1"),
    ],
)
def test_normalize_device_cases(device, expected):
    assert runtime.normalize_device(device) == expected


def test_package_modules_import_without_scripts_hack():
    __import__("wrist_fracture.validation_benchmark_suite")
    __import__("wrist_fracture.smoke_suite")
    __import__("wrist_fracture.runtime")


def test_validation_benchmark_module_has_no_scripts_dependency():
    import wrist_fracture.validation_benchmark_suite as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "from scripts" not in source
    assert "import scripts" not in source


def test_resolve_checkpoint_accepts_direct_pt(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(checkpoint)
    assert resolved.selected == str(checkpoint.resolve())
    assert resolved.candidates == (str(checkpoint),)


def test_resolve_checkpoint_accepts_project_run_root(tmp_path: Path):
    checkpoint = tmp_path / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(tmp_path)
    assert resolved.selected == str(checkpoint.resolve())


def test_resolve_checkpoint_accepts_raw_ultralytics_run_root(tmp_path: Path):
    checkpoint = tmp_path / "raw" / "train" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(tmp_path)
    assert resolved.selected == str(checkpoint.resolve())


def test_resolve_checkpoint_missing(tmp_path: Path):
    with pytest.raises(ConfigError, match="checkpoint missing or invalid"):
        resolve_checkpoint_path(tmp_path / "missing")


def test_resolve_checkpoint_rejects_zero_byte_file(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"")
    with pytest.raises(ConfigError, match="checkpoint missing or invalid"):
        resolve_checkpoint_path(checkpoint)


def test_resolve_checkpoint_does_not_append_to_existing_pt(tmp_path: Path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"model")
    resolved = resolve_checkpoint_path(checkpoint)
    assert "weights/best.pt" not in resolved.selected
    assert "best.pt/weights/best.pt" not in resolved.selected


def test_smoke_suite_model_records_preserve_checkpoint_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source"
    source.mkdir()
    run_root = tmp_path / "runs"
    run_root.mkdir()
    records = []
    for model in ["yolov8", "yolov9", "yolo26"]:
        checkpoint = tmp_path / model / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(model.encode("utf-8"))
        records.append(
            {
                "model_family": model,
                "run_path": str(run_root / model),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": f"sha-{model}",
            }
        )
    (source / "suite_summary.json").write_text(json.dumps({"models": records}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            source_suite=str(source),
            dry_run=False,
            execute=True,
            suite_id="suite",
            models=None,
            skip_completed=False,
            continue_on_error=True,
            resume=False,
            force=True,
            warmup=0,
            samples=1,
            benchmark_batch_size=1,
            io_workers=0,
            print_commands=False,
        ),
    )
    monkeypatch.setattr(
        suite,
        "evaluate_checkpoint",
        lambda **kwargs: {"metrics": {"checkpoint_sha256": kwargs["checkpoint"].name}},
    )
    monkeypatch.setattr(
        suite,
        "benchmark_checkpoint",
        lambda **kwargs: {"latency": {"mean": 1.0}, "complexity": {}},
    )
    monkeypatch.setattr(suite, "deterministic_sample_manifest", lambda images, samples: [])
    monkeypatch.setattr(suite, "discover_source_runs", lambda _: records)
    monkeypatch.setattr(suite, "collect_environment_report", lambda _: {})
    monkeypatch.setattr(suite, "git_commit", lambda _: None)
    monkeypatch.setattr(suite, "git_dirty", lambda _: False)
    monkeypatch.setattr(suite, "to_jsonable", lambda value: value)
    monkeypatch.setattr(suite, "_finish_marker", lambda *args, **kwargs: None)
    assert suite.main() == 0
    summary = json.loads(
        (tmp_path / "outputs/validation_benchmark_suites/suite/suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["checkpoint"] for row in summary["models"]] == [
        rec["checkpoint"] for rec in records
    ]
    assert [row["checkpoint_sha256"] for row in summary["models"]] == [
        rec["checkpoint_sha256"] for rec in records
    ]
