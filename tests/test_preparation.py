from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.prepare_dataset import validate_split

from wrist_fracture.data.preparation import (
    AnnotationBox,
    ImageRecord,
    build_patient_split,
    parse_pascalvoc,
    parse_supervisely,
    json_ready,
    save_json,
)


def test_parse_pascalvoc(tmp_path: Path):
    xml = tmp_path / "sample.xml"
    xml.write_text(
        """<annotation><filename>a.png</filename><size><width>100</width><height>200</height></size><object><name>fracture</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>40</ymax></bndbox></object></annotation>""",
        encoding="utf-8",
    )
    filename, width, height, boxes = parse_pascalvoc(xml)
    assert filename == "a.png"
    assert width == 100 and height == 200
    assert boxes[0].label == "fracture"


def test_parse_supervisely(tmp_path: Path):
    js = tmp_path / "sample.json"
    js.write_text(
        '{"size":{"width":100,"height":200},"objects":[{"classTitle":"fracture","points":{"exterior":[[10,20],[30,40]]}}]}',
        encoding="utf-8",
    )
    width, height, boxes = parse_supervisely(js)
    assert width == 100 and height == 200
    assert boxes[0].xmin == 10 and boxes[0].xmax == 30


def test_yolo_conversion():
    box = AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")
    cls, cx, cy, bw, bh = box.to_yolo(100, 200)
    assert cls == 0
    assert cx == pytest.approx(0.2)
    assert cy == pytest.approx(0.15)
    assert bw == pytest.approx(0.2)
    assert bh == pytest.approx(0.1)


def test_bbox_validation():
    with pytest.raises(ValueError):
        AnnotationBox("fracture", 30, 20, 10, 40, "pascalvoc").to_yolo(100, 200)


def test_patient_split_and_leakage():
    split = build_patient_split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], seed=123)
    assert sum(len(v) for v in split.values()) == 10
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
    errors = validate_split(
        {
            "train": [{"patient_id": 1}],
            "val": [{"patient_id": 2}],
            "test": [{"patient_id": 1}],
        }
    )
    assert errors


def test_json_ready_serializes_nested_paths(tmp_path: Path):
    report = {
        "manifest": [
            {
                "archive": Path("data/raw/archives/a.zip"),
                "target": tmp_path / "data" / "raw" / "extracted" / "a",
            }
        ],
        "record": ImageRecord(
            stem="sample",
            image_path=tmp_path / "data" / "raw" / "extracted" / "sample.png",
            annotation_path=tmp_path / "data" / "raw" / "extracted" / "sample.xml",
            annotation_format="pascalvoc",
            patient_id="1",
            study_id="2",
            width=100,
            height=200,
            channels=1,
            dtype="uint8",
            fracture_boxes=[AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")],
            all_boxes=[AnnotationBox("fracture", 10, 20, 30, 40, "pascalvoc")],
            labels=["fracture"],
        ),
    }

    ready = json_ready(report, base_dir=tmp_path)
    json.dumps(ready)
    assert ready["manifest"][0]["target"] == "data/raw/extracted/a"
    assert ready["record"]["image_path"] == "data/raw/extracted/sample.png"
    assert ready["record"]["annotation_path"] == "data/raw/extracted/sample.xml"


def test_save_json_handles_paths(tmp_path: Path):
    output = tmp_path / "report.json"
    save_json(output, {"path": Path("data") / "nested" / "file.txt"})
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["path"] == "data/nested/file.txt"
