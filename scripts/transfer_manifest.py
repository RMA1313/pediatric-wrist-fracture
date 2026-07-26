from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from wrist_fracture.provenance import dependency_lock_hash, git_commit, git_dirty, sha256_file


def _file_hash(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def _tree_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = json.dumps(
        sorted(
            (
                str(item.relative_to(path)).replace("\\", "/"),
                sha256_file(item),
            )
            for item in path.rglob("*")
            if item.is_file()
        ),
        separators=(",", ":"),
    )
    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def build_manifest(root: Path, dataset_yaml: Path) -> dict[str, object]:
    return {
        "git_commit": git_commit(root),
        "git_dirty": git_dirty(root),
        "uv_lock_sha256": dependency_lock_hash(root),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": _file_hash(dataset_yaml),
        "patient_split_hashes": {
            "train": _file_hash(root / "data/splits/train_patients.json"),
            "val": _file_hash(root / "data/splits/val_patients.json"),
            "test": _file_hash(root / "data/splits/test_patients.json"),
        },
        "image_split_hashes": {
            "train": _tree_hash(root / "data/processed/yolo/images/train"),
            "val": _tree_hash(root / "data/processed/yolo/images/val"),
            "test": _tree_hash(root / "data/processed/yolo/images/test"),
        },
        "final_dataset_audit_sha256": _file_hash(
            root / "outputs/dataset_reports/final_dataset_audit.json"
        ),
        "expected_counts": {
            "images": None,
            "labels": None,
            "patients": None,
        },
        "required_local_paths": [
            "data/processed/yolo/dataset.yaml",
            "outputs/dataset_reports/final_dataset_audit.json",
            "data/splits",
            "data/processed/yolo/images",
            "data/processed/yolo/labels",
        ],
        "approximate_disk_requirement_gb": None,
    }


def verify_manifest(manifest: dict[str, object], root: Path, dataset_yaml: Path) -> list[str]:
    errors: list[str] = []
    if manifest.get("git_commit") != git_commit(root):
        errors.append("git commit mismatch")
    if manifest.get("git_dirty") != git_dirty(root):
        errors.append("dirty working-tree flag mismatch")
    if manifest.get("uv_lock_sha256") != dependency_lock_hash(root):
        errors.append("uv.lock hash mismatch")
    if manifest.get("dataset_yaml_sha256") != _file_hash(dataset_yaml):
        errors.append("dataset YAML hash mismatch")
    if manifest.get("final_dataset_audit_sha256") != _file_hash(
        root / "outputs/dataset_reports/final_dataset_audit.json"
    ):
        errors.append("dataset audit hash mismatch")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/transfer_manifest.json")
    parser.add_argument("--dataset-yaml", default="data/processed/yolo/dataset.yaml")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    dataset_yaml = root / args.dataset_yaml
    manifest = build_manifest(root, dataset_yaml)
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if args.verify:
        errors = verify_manifest(manifest, root, dataset_yaml)
        if errors:
            raise SystemExit("; ".join(errors))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
