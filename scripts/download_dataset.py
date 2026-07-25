from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wrist_fracture.paths import get_paths

FIGSHARE_API = "https://api.figshare.com/v2/articles/14825193"


@dataclass(frozen=True)
class FigshareFile:
    id: int
    name: str
    download_url: str | None
    size: int | None
    checksum: str | None


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def inspect_dataset_record() -> dict[str, Any]:
    article = fetch_json(FIGSHARE_API)
    files = []
    for item in article.get("files", []):
        files.append(
            FigshareFile(
                id=int(item["id"]),
                name=str(item["name"]),
                download_url=item.get("download_url"),
                size=item.get("size"),
                checksum=item.get("supplied_md5") or item.get("computed_md5"),
            )
        )
    return {"article": article, "files": files}


def estimate_disk_requirement(files: list[FigshareFile]) -> int:
    return sum(file.size or 0 for file in files)


def checksum_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".md5")


def md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(file: FigshareFile, target_dir: Path, dry_run: bool = False) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    if not file.download_url:
        raise RuntimeError(f"No download URL available for {file.name}")
    destination = target_dir / file.name
    if destination.exists() and file.checksum:
        checksum_file = checksum_path(destination)
        if (
            checksum_file.exists()
            and checksum_file.read_text(encoding="utf-8").strip() == file.checksum
        ):
            return destination
        if md5_file(destination) == file.checksum:
            checksum_file.write_text(file.checksum, encoding="utf-8")
            return destination
    if dry_run:
        return destination
    tmp = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(file.download_url, timeout=600) as response, tmp.open("wb") as out:
        total = response.headers.get("Content-Length")
        total_int = int(total) if total else None
        with tqdm(total=total_int, unit="B", unit_scale=True, desc=file.name) as bar:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bar.update(len(chunk))
    tmp.replace(destination)
    if file.checksum:
        checksum_file = checksum_path(destination)
        checksum_file.write_text(file.checksum, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or download the GRAZPEDWRI-DX dataset.")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=get_paths().raw)
    args = parser.parse_args()

    record = inspect_dataset_record()
    article = record["article"]
    files: list[FigshareFile] = record["files"]
    payload = {
        "title": article.get("title"),
        "doi": article.get("doi"),
        "id": article.get("id"),
        "published_date": article.get("published_date"),
        "file_count": len(files),
        "files": [file.__dict__ for file in files],
        "estimated_disk_bytes": estimate_disk_requirement(files),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.metadata_only:
        return 0
    if args.dry_run:
        for file in files[:5]:
            print(f"DRY RUN: would download {file.name}")
        return 0
    for file in files:
        download_file(file, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
