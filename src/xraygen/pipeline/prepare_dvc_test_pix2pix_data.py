from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
REQUIRED_DIRS = [
    "train",
    "test",
    "matched_masks/train/tray",
    "matched_masks/train/blade",
    "matched_masks/test/tray",
    "matched_masks/test/blade",
]


def count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTS)


def safe_clear(path: Path) -> None:
    resolved = path.resolve()
    if not any("_dvc_test" in part for part in resolved.parts):
        raise ValueError(f"Refusing to replace non-test path: {resolved}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    safe_clear(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copytree(src, dst)
        return

    relative_src = os.path.relpath(src.resolve(), start=dst.parent.resolve())
    dst.symlink_to(relative_src, target_is_directory=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare an isolated Pix2Pix dataset workspace for DVC test runs.")
    ap.add_argument("--source", required=True, help="Existing Pix2Pix dataset with train/test and matched_masks.")
    ap.add_argument("--dest", required=True, help="Destination under a _dvc_test path.")
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--link", choices=["symlink", "copy"], default="symlink")
    args = ap.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    report_json = Path(args.report_json)

    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")

    missing = [rel for rel in REQUIRED_DIRS if not (source / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Source dataset is missing required directories: {missing}")

    dest.mkdir(parents=True, exist_ok=True)
    for rel in REQUIRED_DIRS:
        link_or_copy(source / rel, dest / rel, args.link)

    counts = {rel: count_images(dest / rel) for rel in REQUIRED_DIRS}
    report = {
        "source": str(source),
        "dest": str(dest),
        "link": args.link,
        "required_dirs": REQUIRED_DIRS,
        "counts": counts,
        "ready": all(count > 0 for count in counts.values()),
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2))
    print(f"[prepare] wrote {report_json}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
