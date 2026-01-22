#!/usr/bin/env python3
"""
finalize_dataset.py

Freeze a training-ready dataset:
data/interim/... -> data/processed/images/
and create:
- data/processed/splits.json
- data/processed/index.csv

Usage:
python finalize_dataset.py \
  --input_dir ../../../data/interim/Stage0/Color \
  --processed_dir ../../../data/process/Stage0/Color \
  --dataset_tag Stage0_Color \
  --train 0.8 --val 0.1 --test 0.1 \
  --seed 42
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


def list_images(root: Path) -> List[Path]:
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            paths.append(p)
    paths.sort()
    return paths


def stable_shuffle(paths: List[Path], seed: int) -> List[Path]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    return [paths[i] for i in idx]


def split_paths(paths: List[Path], train: float, val: float, test: float) -> Dict[str, List[Path]]:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"train+val+test must sum to 1.0, got {total}")

    n = len(paths)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train

    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:n_train + n_val + n_test],
    }


def infer_stage(input_dir: Path) -> str:
    # If any folder looks like "Stage0", "Stage1", etc.
    for part in input_dir.parts:
        if part.lower().startswith("stage"):
            return part
    return "unknown"


def infer_scan_session_id(path: Path, input_dir: Path) -> str:
    """
    Heuristic:
    - If file is under a subfolder, use the first folder name as session id.
      input_dir/session_001/img.png -> session_001
    - Otherwise: unknown
    """
    rel = path.relative_to(input_dir)
    return rel.parts[0] if len(rel.parts) >= 2 else "unknown"


def finalize_dataset(
    input_dir: Path,
    processed_dir: Path,
    dataset_tag: str,
    train: float,
    val: float,
    test: float,
    seed: int,
):
    images_out = processed_dir / "images" / dataset_tag
    splits_out = processed_dir / "splits.json"
    index_out = processed_dir / "index.csv"

    processed_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(parents=True, exist_ok=True)

    paths = list_images(input_dir)
    if not paths:
        raise RuntimeError(f"No images found in {input_dir}")

    stage = infer_stage(input_dir)
    shuffled = stable_shuffle(paths, seed=seed)
    splits = split_paths(shuffled, train=train, val=val, test=test)

    # We'll store filepaths relative to data/processed/
    split_lists_rel = {"train": [], "val": [], "test": []}

    with open(index_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath",        # relative to data/processed/
                "split",
                "dataset_tag",
                "stage",
                "scan_session_id",
                "source_path",     # relative to input_dir
                "width",
                "height",
                "channels",
                "dtype",
            ],
        )
        writer.writeheader()

        for split_name, split_paths_list in splits.items():
            for src in split_paths_list:
                rel_src = src.relative_to(input_dir)
                dst = images_out / rel_src
                dst.parent.mkdir(parents=True, exist_ok=True)

                # Copy bytes (interim is already training-ready)
                dst.write_bytes(src.read_bytes())

                rel_dst_to_processed = dst.relative_to(processed_dir)
                split_lists_rel[split_name].append(str(rel_dst_to_processed))

                img = load_image(dst)
                h, w = img.shape[:2]
                channels = 1 if img.ndim == 2 else img.shape[2]
                scan_session_id = infer_scan_session_id(src, input_dir)

                writer.writerow(
                    {
                        "filepath": str(rel_dst_to_processed),
                        "split": split_name,
                        "dataset_tag": dataset_tag,
                        "stage": stage,
                        "scan_session_id": scan_session_id,
                        "source_path": str(rel_src),
                        "width": int(w),
                        "height": int(h),
                        "channels": int(channels),
                        "dtype": str(img.dtype),
                    }
                )

    splits_payload = {
        "dataset_tag": dataset_tag,
        "seed": seed,
        "ratios": {"train": train, "val": val, "test": test},
        "counts": {k: len(v) for k, v in split_lists_rel.items()},
        "files": split_lists_rel,
    }
    with open(splits_out, "w") as f:
        json.dump(splits_payload, f, indent=2)

    print("✅ Finalize complete")
    print(f"  Copied images -> {images_out}")
    print(f"  splits.json   -> {splits_out}")
    print(f"  index.csv     -> {index_out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, required=True, help="Interim folder to freeze (e.g., data/interim/Stage0/Gray)")
    p.add_argument("--processed_dir", type=Path, default=Path("data/processed"))
    p.add_argument("--dataset_tag", type=str, required=True, help="Name under processed/images/<dataset_tag>/")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    finalize_dataset(
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        dataset_tag=args.dataset_tag,
        train=args.train,
        val=args.val,
        test=args.test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
