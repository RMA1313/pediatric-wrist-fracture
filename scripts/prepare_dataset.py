# ruff: noqa: E402
from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.download_dataset import download_file, inspect_dataset_record
from wrist_fracture.data.preparation import (
    AnnotationBox,
    ImageRecord,
    build_patient_split,
    classify_label,
    dedupe_boxes,
    ensure_idempotent_remove,
    parse_dataset_csv,
    parse_pascalvoc,
    parse_supervisely,
    render_histogram,
    safe_extract_zip,
    save_json,
    write_csv,
)
from wrist_fracture.paths import get_paths


def load_dataset_assets(raw_dir: Path) -> dict[str, Path]:
    archives = raw_dir / "archives"
    extracted = raw_dir / "extracted"
    return {
        "archives": archives,
        "extracted": extracted,
        "dataset_csv": next(archives.rglob("dataset.csv")),
        "folder_structure": next(archives.rglob("folder_structure.zip")),
    }


def extract_archives(raw_dir: Path, force: bool = False) -> dict[str, Any]:
    assets = load_dataset_assets(raw_dir)
    extracted = assets["extracted"]
    extracted.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for zip_path in sorted((raw_dir / "archives").glob("*.zip")):
        target = extracted / zip_path.stem
        if force and target.exists():
            shutil.rmtree(target)
        if target.exists() and any(target.rglob("*")):
            manifest.append({"archive": zip_path.name, "target": str(target), "skipped": True})
            continue
        safe_extract_zip(zip_path, target)
        manifest.append({"archive": zip_path.name, "target": str(target), "skipped": False})
    return {"manifest": manifest, "extracted": extracted}


def locate_annotation_root(extracted: Path) -> tuple[Path | None, Path | None, Path | None]:
    pascal = next(iter(sorted(extracted.rglob("*.xml"))), None)
    sup = next(iter(sorted(extracted.rglob("*.json"))), None)
    images = next(iter(sorted(extracted.rglob("*.png"))), None)
    return (
        pascal.parent if pascal else None,
        sup.parent if sup else None,
        images.parent if images else None,
    )


def build_records(raw_dir: Path) -> tuple[list[ImageRecord], dict[str, Any]]:
    assets = load_dataset_assets(raw_dir)
    csv_df = parse_dataset_csv(assets["dataset_csv"])
    extracted = assets["extracted"]
    records: list[ImageRecord] = []
    summary = {
        "missing_images": 0,
        "missing_annotations": 0,
        "invalid_images": 0,
        "formats": Counter(),
        "labels": Counter(),
    }
    for _, row in csv_df.iterrows():
        stem = str(row.get("filestem") or row.get("filename") or row.get("image"))
        patient_id = str(row.get("patient_id")) if "patient_id" in row else None
        study_id = str(row.get("study_id")) if "study_id" in row else None
        image_path = next(iter(sorted(extracted.rglob(f"{stem}.png"))), None)
        if image_path is None:
            summary["missing_images"] += 1
            continue
        info = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if info is None:
            summary["invalid_images"] += 1
            records.append(
                ImageRecord(
                    stem,
                    image_path,
                    None,
                    None,
                    patient_id,
                    study_id,
                    0,
                    0,
                    0,
                    "",
                    [],
                    [],
                    [],
                    unreadable=True,
                )
            )
            continue
        width, height = int(info.shape[1]), int(info.shape[0])
        channels = 1 if info.ndim == 2 else int(info.shape[2])
        dtype = str(info.dtype)
        xml = next(iter(sorted(extracted.rglob(f"{stem}.xml"))), None)
        js = next(iter(sorted(extracted.rglob(f"{stem}.json"))), None)
        boxes: list[AnnotationBox] = []
        fmt = None
        if xml and xml.exists():
            _, _, _, boxes = parse_pascalvoc(xml)
            fmt = "pascalvoc"
        elif js and js.exists():
            _, _, boxes = parse_supervisely(js)
            fmt = "supervisely"
        if not boxes:
            summary["missing_annotations"] += 1
        boxes = dedupe_boxes(boxes)
        fracture_boxes = [b for b in boxes if classify_label(b.label) == "fracture"]
        summary["formats"][fmt or "missing"] += 1
        for b in boxes:
            summary["labels"][b.label] += 1
        records.append(
            ImageRecord(
                stem,
                image_path,
                xml or js,
                fmt,
                patient_id,
                study_id,
                width,
                height,
                channels,
                dtype,
                fracture_boxes,
                boxes,
                sorted({b.label for b in boxes}),
            )
        )
    return records, summary


def compare_annotation_formats(raw_dir: Path) -> dict[str, Any]:
    extracted = load_dataset_assets(raw_dir)["extracted"]
    xmls = {p.stem: p for p in extracted.rglob("*.xml")}
    jsons = {p.stem: p for p in extracted.rglob("*.json")}
    paired = sorted(set(xmls) & set(jsons))
    sample = paired[:500]
    match = 0
    mismatch = 0
    for stem in sample:
        _, _, _, x = parse_pascalvoc(xmls[stem])
        _, _, j = parse_supervisely(jsons[stem])
        if len(x) == len(j) and all(
            a.key() == b.key()
            for a, b in zip(dedupe_boxes(x), dedupe_boxes(j), strict=False)
        ):
            match += 1
        else:
            mismatch += 1
    return {
        "paired_images": len(paired),
        "sample_compared": len(sample),
        "matching": match,
        "mismatching": mismatch,
        "xml_only": len(xmls) - len(paired),
        "json_only": len(jsons) - len(paired),
    }


def convert_to_yolo(
    records: list[ImageRecord], processed_dir: Path, negative_empty: bool = True
) -> dict[str, Any]:
    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"
    ensure_idempotent_remove(images_root)
    ensure_idempotent_remove(labels_root)
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    counts = Counter()
    for rec in records:
        img_dst = images_root / f"{rec.stem}.png"
        if not img_dst.exists():
            shutil.copy2(rec.image_path, img_dst)
        label_path = labels_root / f"{rec.stem}.txt"
        lines = []
        invalid = 0
        for box in rec.fracture_boxes:
            try:
                cls, cx, cy, bw, bh = box.to_yolo(rec.width, rec.height)
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                counts["fracture_boxes"] += 1
            except ValueError:
                invalid += 1
        if lines or negative_empty:
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        counts["images"] += 1
        counts["positive_images" if lines else "negative_images"] += 1
        manifest.append(
            {
                "stem": rec.stem,
                "image_path": str(img_dst),
                "label_path": str(label_path),
                "patient_id": rec.patient_id,
                "study_id": rec.study_id,
                "fracture_boxes": len(lines),
                "invalid_boxes": invalid,
                "source_annotation": str(rec.annotation_path) if rec.annotation_path else None,
                "source_format": rec.annotation_format,
            }
        )
    write_csv(processed_dir / "manifests" / "conversion_manifest.csv", manifest)
    return {
        "counts": dict(counts),
        "manifest_rows": len(manifest),
        "images_dir": str(images_root),
        "labels_dir": str(labels_root),
    }


def make_splits(records: list[ImageRecord], seed: int = 42) -> dict[str, list[ImageRecord]]:
    by_patient: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        by_patient[str(rec.patient_id)].append(rec)
    split_patients = build_patient_split(list(by_patient.keys()), seed=seed)
    return {k: [rec for pid in v for rec in by_patient[pid]] for k, v in split_patients.items()}


def save_split_files(splits: dict[str, list[ImageRecord]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, rows in splits.items():
        patients = sorted({r.patient_id for r in rows if r.patient_id is not None})
        write_csv(out_dir / f"{split}_patients.csv", [{"patient_id": p} for p in patients])
        write_csv(
            out_dir / f"{split}_images.csv",
            [
                {
                    "stem": r.stem,
                    "patient_id": r.patient_id,
                    "study_id": r.study_id,
                    "image_path": str(r.image_path),
                }
                for r in rows
            ],
        )
        result[split] = {
            "patients": len(patients),
            "images": len(rows),
            "positive_images": sum(bool(r.fracture_boxes) for r in rows),
            "negative_images": sum(not r.fracture_boxes for r in rows),
            "fracture_boxes": sum(len(r.fracture_boxes) for r in rows),
        }
    return result


def validate_records(
    records: list[ImageRecord], split_records: dict[str, list[ImageRecord]] | None = None
) -> dict[str, Any]:
    errors: list[str] = []
    box_errors = 0
    for rec in records:
        for box in rec.fracture_boxes:
            try:
                box.to_yolo(rec.width, rec.height)
            except Exception:
                box_errors += 1
    if split_records:
        patient_sets = {k: {r.patient_id for r in v} for k, v in split_records.items()}
        if (
            patient_sets["train"] & patient_sets["val"]
            or patient_sets["train"] & patient_sets["test"]
            or patient_sets["val"] & patient_sets["test"]
        ):
            errors.append("patient leakage detected")
    return {"errors": errors, "invalid_boxes": box_errors, "image_count": len(records)}


def generate_dataset_figures(
    records: list[ImageRecord], figure_dir: Path, seed: int = 42
) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_labels = Counter(b.label for rec in records for b in rec.all_boxes)
    pos = [1 if rec.fracture_boxes else 0 for rec in records]
    render_histogram(
        list(source_labels.values()),
        figure_dir / "source_label_distribution.png",
        "Source Label Distribution",
        "Count per label",
    )
    render_histogram(
        [rec.width for rec in records],
        figure_dir / "image_width_distribution.png",
        "Image Width Distribution",
        "Width (px)",
    )
    render_histogram(
        [rec.height for rec in records],
        figure_dir / "image_height_distribution.png",
        "Image Height Distribution",
        "Height (px)",
    )
    render_histogram(
        [rec.width / rec.height for rec in records if rec.height],
        figure_dir / "aspect_ratio_distribution.png",
        "Aspect Ratio Distribution",
        "Width / Height",
    )
    render_histogram(
        [len(rec.fracture_boxes) for rec in records],
        figure_dir / "fracture_boxes_per_image.png",
        "Fracture Boxes per Image",
        "Boxes",
    )
    render_histogram(
        [sum((b.xmax - b.xmin) * (b.ymax - b.ymin) for b in rec.fracture_boxes) for rec in records],
        figure_dir / "fracture_bbox_area_distribution.png",
        "Fracture Bounding Box Area Distribution",
        "Pixel area",
    )
    render_histogram(
        pos,
        figure_dir / "positive_negative_distribution.png",
        "Positive vs Negative Images",
        "Binary label",
    )
    return [str(p) for p in sorted(figure_dir.glob("*.png"))]


def write_dataset_yaml(processed_dir: Path) -> Path:
    yaml_path = processed_dir / "dataset.yaml"
    payload = {
        "path": str(processed_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": ["fracture"],
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def build_final_dataset(
    records: list[ImageRecord],
    processed_dir: Path,
    splits: dict[str, list[ImageRecord]],
    force: bool = False,
) -> Path:
    yolo_dir = processed_dir / "yolo"
    if force and yolo_dir.exists():
        shutil.rmtree(yolo_dir)
    for split in ("train", "val", "test"):
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        for rec in rows:
            img_dst = yolo_dir / "images" / split / rec.image_path.name
            lbl_dst = yolo_dir / "labels" / split / f"{rec.stem}.txt"
            if not img_dst.exists():
                shutil.copy2(rec.image_path, img_dst)
            lines = []
            for box in rec.fracture_boxes:
                try:
                    cls, cx, cy, bw, bh = box.to_yolo(rec.width, rec.height)
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                except ValueError:
                    pass
            lbl_dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_dataset_yaml(yolo_dir)
    return yolo_dir


def smoke_load_dataset(yolo_dir: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    data = model.data
    return {
        "dataset_yaml": str(yolo_dir / "dataset.yaml"),
        "model_loaded": bool(model),
        "data_keys": sorted(list(data.keys())) if isinstance(data, dict) else [],
    }


def summarize(records: list[ImageRecord], comparison: dict[str, Any]) -> dict[str, Any]:
    all_labels = Counter(b.label for rec in records for b in rec.all_boxes)
    frac = Counter(len(rec.fracture_boxes) for rec in records)
    return {
        "total_images": len(records),
        "patient_count": len({r.patient_id for r in records if r.patient_id is not None}),
        "study_count": len({r.study_id for r in records if r.study_id is not None}),
        "labels": dict(all_labels),
        "fracture_images": sum(bool(r.fracture_boxes) for r in records),
        "negative_images": sum(not r.fracture_boxes for r in records),
        "multiple_fractures": sum(len(r.fracture_boxes) > 1 for r in records),
        "annotation_comparison": comparison,
        "fracture_boxes_per_image": dict(frac),
    }


def cmd_download(args: argparse.Namespace) -> int:
    record = inspect_dataset_record()
    files = record["files"]
    manifest = [download_file(file, args.output_dir, force=args.force) for file in files]
    save_json(args.manifest, {"article": record["article"], "files": manifest})
    return 0


def validate_split(splits: dict[str, list[dict[str, Any]]]) -> list[str]:
    patient_sets = {name: {r["patient_id"] for r in rows} for name, rows in splits.items()}
    errors = []
    if patient_sets["train"] & patient_sets["val"]:
        errors.append("train/val overlap")
    if patient_sets["train"] & patient_sets["test"]:
        errors.append("train/test overlap")
    if patient_sets["val"] & patient_sets["test"]:
        errors.append("val/test overlap")
    return errors


def cmd_extract(args: argparse.Namespace) -> int:
    result = extract_archives(args.raw_dir, force=args.force)
    save_json(args.manifest, result)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    records, summary = build_records(args.raw_dir)
    comparison = compare_annotation_formats(args.raw_dir)
    save_json(args.report_json, summarize(records, comparison))
    write_csv(
        args.report_csv,
        [
            {
                "stem": r.stem,
                "patient_id": r.patient_id,
                "study_id": r.study_id,
                "width": r.width,
                "height": r.height,
                "channels": r.channels,
                "labels": ",".join(r.labels),
                "fracture_boxes": len(r.fracture_boxes),
                "annotation_format": r.annotation_format or "",
                "unreadable": r.unreadable,
            }
            for r in records
        ],
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    records, _ = build_records(args.raw_dir)
    comparison = compare_annotation_formats(args.raw_dir)
    split_records = make_splits(records, seed=args.seed)
    conversion = convert_to_yolo(records, args.processed_dir)
    split_stats = save_split_files(split_records, args.splits_dir)
    save_json(args.conversion_report, {"conversion": conversion, "comparison": comparison})
    save_json(args.split_report, split_stats)
    save_json(args.dataset_report, summarize(records, comparison))
    generate_dataset_figures(records, args.figures_dir)
    write_dataset_yaml(args.processed_dir / "yolo")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    return cmd_convert(args)


def cmd_validate(args: argparse.Namespace) -> int:
    records, _ = build_records(args.raw_dir)
    split_records = make_splits(records, seed=args.seed)
    save_json(args.validation_report, validate_records(records, split_records))
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    save_json(args.smoke_report, smoke_load_dataset(args.processed_dir / "yolo"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dataset preparation workflow")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download")
    d.add_argument("--output-dir", type=Path, default=get_paths().raw / "archives")
    d.add_argument(
        "--manifest", type=Path, default=get_paths().dataset_reports / "download_manifest.json"
    )
    d.add_argument("--force", action="store_true")
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("extract")
    e.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    e.add_argument(
        "--manifest", type=Path, default=get_paths().dataset_reports / "extraction_manifest.json"
    )
    e.add_argument("--force", action="store_true")
    e.set_defaults(func=cmd_extract)

    i = sub.add_parser("inspect")
    i.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    i.add_argument(
        "--report-json", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    i.add_argument(
        "--report-csv", type=Path, default=get_paths().dataset_reports / "dataset_report.csv"
    )
    i.set_defaults(func=cmd_inspect)

    c = sub.add_parser("convert")
    c.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    c.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    c.add_argument("--splits-dir", type=Path, default=get_paths().splits)
    c.add_argument(
        "--dataset-report", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    c.add_argument(
        "--conversion-report",
        type=Path,
        default=get_paths().dataset_reports / "conversion_report.json",
    )
    c.add_argument(
        "--split-report", type=Path, default=get_paths().dataset_reports / "split_report.json"
    )
    c.add_argument("--figures-dir", type=Path, default=get_paths().figures / "dataset_statistics")
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_convert)

    v = sub.add_parser("validate")
    v.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    v.add_argument(
        "--validation-report",
        type=Path,
        default=get_paths().dataset_reports / "validation_report.json",
    )
    v.add_argument("--seed", type=int, default=42)
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("smoke")
    s.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    s.add_argument(
        "--smoke-report", type=Path, default=get_paths().dataset_reports / "smoke_report.json"
    )
    s.set_defaults(func=cmd_smoke)

    p2 = sub.add_parser("prepare")
    p2.add_argument("--raw-dir", type=Path, default=get_paths().raw)
    p2.add_argument("--processed-dir", type=Path, default=get_paths().processed)
    p2.add_argument("--splits-dir", type=Path, default=get_paths().splits)
    p2.add_argument(
        "--dataset-report", type=Path, default=get_paths().dataset_reports / "dataset_report.json"
    )
    p2.add_argument(
        "--conversion-report",
        type=Path,
        default=get_paths().dataset_reports / "conversion_report.json",
    )
    p2.add_argument(
        "--split-report", type=Path, default=get_paths().dataset_reports / "split_report.json"
    )
    p2.add_argument("--figures-dir", type=Path, default=get_paths().figures / "dataset_statistics")
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=cmd_prepare)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
