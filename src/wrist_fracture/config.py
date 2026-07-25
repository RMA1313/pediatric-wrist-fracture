from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    seed: int
    output_dir: Path
    data_dir: Path
    log_level: str = "INFO"


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    figshare_doi: str
    expected_root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    splits_dir: Path
    dry_run: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    checkpoint: str | None
    image_size: int
    batch_size: int
    epochs: int
    device: str
    train_ratio: float
    val_ratio: float
    test_ratio: float


def _require_mapping(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{source} must contain a mapping at its root")
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return _require_mapping(data, str(config_path))


def load_project_config(path: str | Path) -> ProjectConfig:
    data = load_yaml_config(path).get("project")
    if not isinstance(data, dict):
        raise ConfigError("project section is missing or invalid")
    return ProjectConfig(
        name=str(data["name"]),
        seed=int(data["seed"]),
        output_dir=Path(data["output_dir"]),
        data_dir=Path(data["data_dir"]),
        log_level=str(data.get("log_level", "INFO")),
    )


def load_dataset_config(path: str | Path) -> DatasetConfig:
    data = load_yaml_config(path).get("dataset")
    if not isinstance(data, dict):
        raise ConfigError("dataset section is missing or invalid")
    return DatasetConfig(
        source=str(data["source"]),
        figshare_doi=str(data["figshare_doi"]),
        expected_root=Path(data["expected_root"]),
        raw_dir=Path(data["raw_dir"]),
        interim_dir=Path(data["interim_dir"]),
        processed_dir=Path(data["processed_dir"]),
        splits_dir=Path(data["splits_dir"]),
        dry_run=bool(data.get("dry_run", True)),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml_config(path).get("experiment")
    if not isinstance(data, dict):
        raise ConfigError("experiment section is missing or invalid")
    return ExperimentConfig(
        model=str(data["model"]),
        checkpoint=data.get("checkpoint"),
        image_size=int(data["image_size"]),
        batch_size=int(data["batch_size"]),
        epochs=int(data["epochs"]),
        device=str(data["device"]),
        train_ratio=float(data["train_ratio"]),
        val_ratio=float(data["val_ratio"]),
        test_ratio=float(data["test_ratio"]),
    )
