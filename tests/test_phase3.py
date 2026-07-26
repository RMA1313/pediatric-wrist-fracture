from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from scripts import benchmark, gpu_preflight, train, transfer_manifest

from wrist_fracture.config import (
    ConfigError,
    ExperimentConfig,
    HardwareConfig,
    ModelConfig,
    RunConfig,
    config_to_dict,
    load_config_bundle,
    validate_experiment_config,
)
from wrist_fracture.models.registry import resolve_model_spec


def _write_bundle(tmp_path: Path, *, device: str = "cpu", batch_size: int = 1) -> Path:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "path: data\ntrain: train\nval: val\ntest: test\nnames: {0: fracture}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
experiment:
  dataset:
    yaml: {dataset.as_posix()}
  model:
    family: yolo26
    checkpoint: yolo26n.pt
    scale: n
  hardware:
    device: {device}
    amp: false
    workers: 0
    cache: ram
    deterministic: true
    allow_cpu_training: true
    require_gpu: false
  run:
    name: smoke
    output_root: {tmp_path.as_posix()}/outputs
    resume: false
    save_period: 1
    validation_split: val
    test_split: test
    allow_test_evaluation: false
    selection_metric: metrics/mAP50-95(B)
    repeated_runs: 1
    batch_size_policy: fixed
  image_size: 320
  epochs: 1
  patience: 1
  seed: 42
  batch_size: {batch_size}
""",
        encoding="utf-8",
    )
    return experiment


def _write_composed_bundle(
    tmp_path: Path,
    *,
    hardware_device: str = "cpu",
    run_name: str = "smoke",
    model_family: str = "yolo26",
) -> tuple[Path, Path, Path, Path]:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text(
        "path: data\ntrain: train\nval: val\ntest: test\nnames: {0: fracture}\n",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
experiment:
  dataset:
    yaml: {dataset.as_posix()}
  model:
    family: yolo26
    checkpoint: yolo26n.pt
    scale: n
  hardware:
    device: cpu
    amp: false
    workers: 0
    cache: ram
    deterministic: true
    allow_cpu_training: false
    require_gpu: false
  run:
    name: base
    output_root: {tmp_path.as_posix()}/outputs
    resume: false
    save_period: 1
    validation_split: val
    test_split: test
    allow_test_evaluation: false
    selection_metric: metrics/mAP50-95(B)
    repeated_runs: 1
    batch_size_policy: fixed
""",
        encoding="utf-8",
    )
    model = tmp_path / "model.yaml"
    model.write_text(
        f"""
model:
  family: {model_family}
  checkpoint: yolov8n.pt
  scale: n
""",
        encoding="utf-8",
    )
    hardware = tmp_path / "hardware.yaml"
    hardware.write_text(
        f"""
hardware:
  device: {hardware_device}
  amp: true
  workers: 8
  cache: disk
  deterministic: false
  allow_cpu_training: false
  require_gpu: true
""",
        encoding="utf-8",
    )
    run = tmp_path / "run.yaml"
    run.write_text(
        f"""
run:
  name: {run_name}
  output_root: {tmp_path.as_posix()}/outputs
  resume: false
  save_period: 1
  validation_split: val
  test_split: test
  allow_test_evaluation: false
  selection_metric: metrics/mAP50-95(B)
  repeated_runs: 1
  batch_size_policy: fixed
""",
        encoding="utf-8",
    )
    return experiment, model, hardware, run


def _write_smoke_overlay_bundle(
    tmp_path: Path,
    *,
    image_size: int = 320,
    epochs: int = 1,
    patience: int = 1,
    batch_size: int = 4,
    repeated_runs: int = 1,
) -> tuple[Path, Path]:
    experiment = _write_bundle(tmp_path)
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text(
        f"""
image_size: {image_size}
epochs: {epochs}
patience: {patience}
batch_size: {batch_size}
run:
  repeated_runs: {repeated_runs}
  name: smoke
  output_root: {tmp_path.as_posix()}/outputs
  resume: false
  save_period: 1
  validation_split: val
  test_split: test
  allow_test_evaluation: false
  selection_metric: metrics/mAP50-95(B)
  batch_size_policy: fixed
""",
        encoding="utf-8",
    )
    return experiment, smoke


def test_config_composition(tmp_path: Path):
    exp = _write_bundle(tmp_path)
    cfg = load_config_bundle(exp)
    assert cfg.model.family == "yolo26"
    assert cfg.hardware.device == "cpu"
    assert cfg.run.name == "smoke"


def test_resolved_overlay_precedence_is_deterministic(tmp_path: Path):
    experiment, model, hardware, run = _write_composed_bundle(
        tmp_path, hardware_device="cuda:0", model_family="yolov8", run_name="smoke"
    )
    cfg = load_config_bundle(
        experiment,
        model_path=model,
        hardware_path=hardware,
        run_path=run,
    )
    assert cfg.model.family == "yolov8"
    assert cfg.hardware.device == "cuda:0"
    assert cfg.run.name == "smoke"


def test_rtx4090_overlay_resolves_gpu_device(tmp_path: Path):
    cfg = load_config_bundle(
        Path("configs/experiment.yaml"),
        model_path=Path("configs/models/yolov8.yaml"),
        hardware_path=Path("configs/hardware/rtx4090.yaml"),
        run_path=Path("configs/runs/smoke.yaml"),
    )
    assert cfg.hardware.device == "cuda:0"
    assert cfg.hardware.require_gpu is True


def test_smoke_overlay_resolves_bounded_values(tmp_path: Path):
    experiment, smoke = _write_smoke_overlay_bundle(tmp_path)
    cfg = load_config_bundle(experiment, run_path=smoke)
    assert cfg.image_size == 320
    assert cfg.epochs == 1
    assert cfg.patience == 1
    assert cfg.batch_size == 4
    assert cfg.run.repeated_runs == 1


def test_full_overlay_still_resolves_full_protocol(tmp_path: Path):
    cfg = load_config_bundle(
        Path("configs/experiment.yaml"),
        model_path=Path("configs/models/yolov8.yaml"),
        hardware_path=Path("configs/hardware/cpu-dev.yaml"),
        run_path=Path("configs/runs/full.yaml"),
    )
    assert cfg.image_size == 640
    assert cfg.epochs == 100
    assert cfg.patience == 20
    assert cfg.batch_size == 1
    assert cfg.run.repeated_runs == 1


def test_smoke_execute_refuses_configs_above_caps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    experiment, smoke = _write_smoke_overlay_bundle(
        tmp_path, image_size=640, epochs=100, patience=20, batch_size=8, repeated_runs=3
    )
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (object,),
            {
                "config": str(experiment),
                "model_config": None,
                "hardware_config": None,
                "run_config": str(smoke),
                "dry_run": False,
                "preflight": False,
                "smoke": True,
                "execute": True,
                "resume": False,
                "allow_cpu_smoke": True,
                "print_resolved_config": False,
            },
        )(),
    )
    with pytest.raises(ConfigError, match="smoke .* exceeds safety cap"):
        train.main()


def test_validate_invalid_values(tmp_path: Path):
    cfg = ExperimentConfig(
        dataset_yaml=tmp_path / "missing.yaml",
        dataset_split_yaml=None,
        model=ModelConfig("bogus", "x.pt", "n"),
        hardware=HardwareConfig(device="cuda:0", workers=-1, require_gpu=True),
        run=RunConfig(
            name="r",
            output_root=tmp_path / "out",
            validation_split="test",
            test_split="test",
        ),
        image_size=0,
        epochs=0,
        batch_size=0,
    )
    errors = validate_experiment_config(cfg, dry_run=True)
    assert any("unknown model family" in e for e in errors)
    assert any("invalid batch size" in e for e in errors)
    assert any("invalid image size" in e for e in errors)
    assert any("invalid epoch count" in e for e in errors)


def test_config_to_dict_serializes_paths(tmp_path: Path):
    exp = _write_bundle(tmp_path)
    cfg = load_config_bundle(exp)
    data = config_to_dict(cfg)
    assert isinstance(data["dataset_yaml"], str)
    assert isinstance(data["run"]["output_root"], str)


def test_train_refuses_full_without_execute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exp = _write_bundle(tmp_path)
    monkeypatch.setattr(train, "resolve_config", lambda args: load_config_bundle(exp))
    monkeypatch.setattr(train, "validate_experiment_config", lambda *a, **k: [])
    monkeypatch.setattr(train, "build_run_id", lambda cfg: "run1")
    monkeypatch.setattr(train, "ensure_unique_run_dir", lambda *a, **k: None)
    monkeypatch.setattr(train, "persist_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "finalize_run", lambda *a, **k: None)
    monkeypatch.setattr(train.sys, "argv", ["train.py", "--config", str(exp)])
    with pytest.raises(ConfigError, match="full training requires explicit --execute"):
        train.main()


def test_train_dry_run_gpu_overlay_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    exp = _write_composed_bundle(tmp_path, hardware_device="cuda:0", model_family="yolov8")
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (object,),
            {
                "config": str(exp[0]),
                "model_config": str(exp[1]),
                "hardware_config": str(exp[2]),
                "run_config": str(exp[3]),
                "dry_run": True,
                "preflight": False,
                "smoke": False,
                "execute": False,
                "resume": False,
                "allow_cpu_smoke": False,
                "print_resolved_config": False,
            },
        )(),
    )
    train.main()
    out = capsys.readouterr().out
    assert '"device": "cuda:0"' in out


def test_train_execute_gpu_fails_when_cuda_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = _write_composed_bundle(tmp_path, hardware_device="cuda:0", model_family="yolov8")
    monkeypatch.setattr(train, "_cuda_is_available", lambda: False)
    monkeypatch.setattr(train, "_cuda_device_exists", lambda device: True)
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (object,),
            {
                "config": str(exp[0]),
                "model_config": str(exp[1]),
                "hardware_config": str(exp[2]),
                "run_config": str(exp[3]),
                "dry_run": False,
                "preflight": False,
                "smoke": False,
                "execute": True,
                "resume": False,
                "allow_cpu_smoke": False,
                "print_resolved_config": False,
            },
        )(),
    )
    with pytest.raises(ConfigError, match="torch.cuda.is_available\\(\\) is false"):
        train.main()


def test_train_execute_gpu_succeeds_with_mocked_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    exp = _write_composed_bundle(tmp_path, hardware_device="cuda:0", model_family="yolov8")
    monkeypatch.setattr(train, "_cuda_is_available", lambda: True)
    monkeypatch.setattr(train, "_cuda_device_exists", lambda device: True)
    monkeypatch.setattr(train, "_execute_training_with_args", lambda cfg, root, args: None)
    monkeypatch.setattr(train, "persist_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "finalize_run", lambda *a, **k: None)
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (object,),
            {
                "config": str(exp[0]),
                "model_config": str(exp[1]),
                "hardware_config": str(exp[2]),
                "run_config": str(exp[3]),
                "dry_run": False,
                "preflight": False,
                "smoke": False,
                "execute": True,
                "resume": False,
                "allow_cpu_smoke": False,
                "print_resolved_config": False,
            },
        )(),
    )
    assert train.main() is None


def test_device_normalization():
    assert train._normalize_device("cuda:0") == "0"
    assert train._normalize_device("cuda") == "0"
    assert train._normalize_device("cpu") == "cpu"


def test_ultralytics_argument_mapping_preserves_smoke_values(monkeypatch: pytest.MonkeyPatch):
    cfg = ExperimentConfig(
        dataset_yaml=Path("data/dataset.yaml"),
        dataset_split_yaml=None,
        model=ModelConfig("yolov8", "yolov8n.pt", "n"),
        hardware=HardwareConfig(
            device="cuda:0",
            amp=True,
            workers=8,
            cache="disk",
            deterministic=False,
        ),
        run=RunConfig(name="smoke", output_root=Path("outputs"), save_period=3),
        image_size=320,
        epochs=1,
        patience=1,
        seed=42,
        optimizer="SGD",
        lr0=0.02,
        lrf=0.1,
        weight_decay=0.001,
        augmentation={"mosaic": 0.0, "mixup": 0.0},
        batch_size=4,
    )
    root = Path("C:/tmp/run")
    root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, checkpoint: str):
            captured["checkpoint"] = checkpoint
            self.trainer = SimpleNamespace(save_dir=root / "raw" / "train")

        def train(self, **kwargs):
            captured["kwargs"] = kwargs
            return {"ok": True}

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(train.sys.modules, "ultralytics", fake_ultralytics)
    monkeypatch.setattr(train, "_read_csv_rows", lambda path: [])
    monkeypatch.setattr(
        train, "_collect_checkpoint_paths", lambda save_dir: {"best": None, "last": None}
    )
    monkeypatch.setattr(train, "_maybe_copy_or_link", lambda src, dst: dst)
    monkeypatch.setattr(train, "_write_csv_rows", lambda *a, **k: None)
    monkeypatch.setattr(train, "_write_validation_json", lambda *a, **k: None)
    monkeypatch.setattr(train, "_write_run_summary", lambda *a, **k: None)
    monkeypatch.setattr(train, "write_atomic", lambda *a, **k: None)
    monkeypatch.setattr(
        train, "resolve_model_spec", lambda model: SimpleNamespace(checkpoint="yolov8n.pt")
    )
    with pytest.raises(ConfigError, match="missing expected Ultralytics checkpoints"):
        train._execute_training_with_args(cfg, root, SimpleNamespace(execute=True, smoke=True))
    assert captured["checkpoint"] == "yolov8n.pt"
    assert captured["kwargs"]["imgsz"] == 320
    assert captured["kwargs"]["epochs"] == 1
    assert captured["kwargs"]["batch"] == 4
    assert captured["kwargs"]["workers"] == 8
    assert captured["kwargs"]["device"] == "0"
    assert captured["kwargs"]["amp"] is True
    assert captured["kwargs"]["seed"] == 42
    assert captured["kwargs"]["deterministic"] is False
    assert captured["kwargs"]["optimizer"] == "SGD"
    assert captured["kwargs"]["lr0"] == 0.02
    assert captured["kwargs"]["lrf"] == 0.1
    assert captured["kwargs"]["weight_decay"] == 0.001
    assert captured["kwargs"]["cache"] == "disk"
    assert captured["kwargs"]["save_period"] == 3
    assert captured["kwargs"]["mosaic"] == 0.0
    assert captured["kwargs"]["mixup"] == 0.0


def test_checkpoint_collection_and_metric_normalization(tmp_path: Path):
    save_dir = tmp_path / "raw" / "train"
    weights = save_dir / "weights"
    weights.mkdir(parents=True)
    (weights / "best.pt").write_text("best", encoding="utf-8")
    (weights / "last.pt").write_text("last", encoding="utf-8")
    (save_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.1,0.2,0.3,0.4\n"
        "1,0.5,0.6,0.7,0.8\n",
        encoding="utf-8",
    )
    checkpoints = train._collect_checkpoint_paths(save_dir)
    assert checkpoints["best"] == weights / "best.pt"
    assert checkpoints["last"] == weights / "last.pt"
    rows = train._normalize_history(train._read_csv_rows(save_dir / "results.csv"))
    metrics = train._metrics_from_history(rows)
    assert metrics["best_epoch"] == 1
    assert metrics["best_map50_95"] == 0.8
    assert metrics["final_precision"] == 0.5
    assert metrics["final_recall"] == 0.6
    assert metrics["final_map50"] == 0.7
    assert metrics["final_map50_95"] == 0.8


def test_interrupt_and_completion_markers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = load_config_bundle(_write_bundle(tmp_path))
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(
        train,
        "_execute_training_with_args",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(train, "persist_run_metadata", lambda *a, **k: None)
    monkeypatch.setattr(train, "ensure_unique_run_dir", lambda *a, **k: None)
    monkeypatch.setattr(train, "build_run_id", lambda cfg: "run1")
    monkeypatch.setattr(train, "run_root", lambda cfg, run_id: root)
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "config": str(tmp_path / "experiment.yaml"),
                "model_config": None,
                "hardware_config": None,
                "run_config": None,
                "dry_run": False,
                "preflight": False,
                "smoke": False,
                "execute": True,
                "resume": False,
                "allow_cpu_smoke": True,
                "print_resolved_config": False,
            },
        )(),
    )
    monkeypatch.setattr(train, "resolve_config", lambda args: cfg)
    with pytest.raises(RuntimeError, match="boom"):
        train.main()
    assert (root / "interrupted.marker").exists()


def test_dry_run_reports_resolved_smoke_imgsz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    exp, smoke = _write_smoke_overlay_bundle(tmp_path)
    monkeypatch.setattr(
        train.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "config": str(exp),
                "model_config": None,
                "hardware_config": None,
                "run_config": str(smoke),
                "dry_run": True,
                "preflight": False,
                "smoke": True,
                "execute": False,
                "resume": False,
                "allow_cpu_smoke": False,
                "print_resolved_config": False,
            },
        )(),
    )
    train.main()
    out = capsys.readouterr().out
    assert '"imgsz": 320' in out


def test_gpu_preflight_cpu_mode(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        gpu_preflight,
        "collect_environment_report",
        lambda root: type(
            "X",
            (),
            {
                "nvidia_smi": None,
                "torch_cuda": None,
                "ultralytics_version": None,
            },
        )(),
    )
    monkeypatch.setattr(
        gpu_preflight.psutil, "disk_usage", lambda path: type("D", (), {"free": 1 << 30})()
    )
    monkeypatch.setattr(
        gpu_preflight.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": ""})()
    )
    monkeypatch.setattr(
        gpu_preflight.argparse.ArgumentParser,
        "parse_args",
        lambda self: type(
            "Args",
            (),
            {
                "require_gpu": False,
                "dataset_yaml": "data/processed/yolo/dataset.yaml",
                "output_dir": "outputs",
            },
        )(),
    )
    gpu_preflight.main()
    out = capsys.readouterr().out
    assert "torch_version" in out


def test_gpu_preflight_require_gpu_fails(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        gpu_preflight,
        "collect_environment_report",
        lambda root: type(
            "X",
            (),
            {
                "nvidia_smi": None,
                "torch_cuda": None,
                "ultralytics_version": None,
            },
        )(),
    )
    with pytest.raises(SystemExit, match="GPU required"):
        monkeypatch.setattr(
            gpu_preflight.argparse.ArgumentParser,
            "parse_args",
            lambda self: type(
                "Args",
                (),
                {
                    "require_gpu": True,
                    "dataset_yaml": "data/processed/yolo/dataset.yaml",
                    "output_dir": "outputs",
                },
            )(),
        )
        gpu_preflight.main()


def test_transfer_manifest_roundtrip(tmp_path: Path):
    root = tmp_path
    (root / "uv.lock").write_text("lock", encoding="utf-8")
    (root / "data/processed/yolo").mkdir(parents=True, exist_ok=True)
    dataset = root / "data/processed/yolo/dataset.yaml"
    dataset.write_text("a: b\n", encoding="utf-8")
    (root / "data/splits").mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (root / f"data/splits/{split}_patients.csv").write_text(
            "patient_id\n1\n2\n",
            encoding="utf-8",
        )
    (root / "outputs/dataset_reports").mkdir(parents=True, exist_ok=True)
    (root / "outputs/dataset_reports/final_dataset_audit.json").write_text("{}", encoding="utf-8")
    manifest = transfer_manifest.build_manifest(root, dataset)
    assert "git_commit" in manifest
    errors = transfer_manifest.verify_manifest(manifest, root, dataset)
    assert errors == []


def test_benchmark_requires_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    exp = _write_bundle(tmp_path)
    monkeypatch.setattr(benchmark, "load_config_bundle", lambda *a, **k: load_config_bundle(exp))
    monkeypatch.setattr(benchmark, "validate_experiment_config", lambda *a, **k: [])
    with pytest.raises(ConfigError, match="checkpoint not found"):
        monkeypatch.setattr(
            benchmark.argparse.ArgumentParser,
            "parse_args",
            lambda self: type(
                "Args",
                (),
                {
                    "config": str(exp),
                    "model_config": None,
                    "hardware_config": None,
                    "run_config": None,
                    "checkpoint": "missing.pt",
                    "dry_run": True,
                    "preflight": False,
                    "execute": False,
                    "warmup": 10,
                    "samples": 100,
                },
            )(),
        )
        benchmark.main()


def test_run_directory_policies(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    with pytest.raises(ConfigError):
        train.ensure_unique_run_dir(root)
    (root / "completed.marker").write_text("done", encoding="utf-8")
    with pytest.raises(ConfigError):
        train.ensure_unique_run_dir(root, resume=True)


def test_resume_conflict_rejected(tmp_path: Path):
    cfg = ExperimentConfig(
        dataset_yaml=tmp_path / "dataset.yaml",
        dataset_split_yaml=None,
        model=ModelConfig("yolo26", "yolo26n.pt", "n"),
        hardware=HardwareConfig(device="cpu", allow_cpu_training=True),
        run=RunConfig(
            name="r",
            output_root=tmp_path / "out",
            resume=True,
            validation_split="val",
            test_split="test",
        ),
        resume_checkpoint=tmp_path / "missing.pt",
        image_size=320,
        epochs=1,
        batch_size=1,
    )
    cfg.dataset_yaml.write_text("x: y\n", encoding="utf-8")
    errors = validate_experiment_config(cfg, dry_run=True)
    assert any("resume checkpoint missing" in e for e in errors)


def test_merge_precedence_is_deterministic(tmp_path: Path):
    experiment, model, hardware, run = _write_composed_bundle(
        tmp_path, hardware_device="cuda:0", model_family="yolov9", run_name="overlay"
    )
    cfg = load_config_bundle(
        experiment,
        model_path=model,
        hardware_path=hardware,
        run_path=run,
    )
    assert cfg.model.family == "yolov9"
    assert cfg.hardware.device == "cuda:0"
    assert cfg.run.name == "overlay"
    assert cfg.run.output_root == tmp_path / "outputs"


def test_model_registry_resolution_without_downloads():
    spec = resolve_model_spec(ModelConfig("yolo26", "yolo26n.pt", "n"))
    assert spec.checkpoint == "yolo26n.pt"
