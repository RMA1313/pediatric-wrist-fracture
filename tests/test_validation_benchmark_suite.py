from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import benchmark, evaluate
from scripts import run_validation_benchmark_suite as suite

from wrist_fracture.config import ConfigError
from wrist_fracture.validation_benchmark_suite import _percentile, _summary_stats, select_runs


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
