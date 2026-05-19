from __future__ import annotations

import argparse
import shutil
from pathlib import Path


NETWORKS = ("G", "D")


def copy_if_needed(src: Path, dst: Path, force: bool) -> bool:
    if not src.exists():
        raise FileNotFoundError(f"Missing source checkpoint: {src}")
    if dst.exists() and not force:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap a Pix2Pix experiment checkpoint from a previous run.")
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--dest-dir", required=True)
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)
    if "_DVC_TEST" not in dest_dir.name and "dvc_test" not in str(dest_dir):
        raise ValueError(f"Refusing to bootstrap non-test checkpoint dir: {dest_dir}")

    copied = []
    kept = []
    for network in NETWORKS:
        name = f"{args.epoch}_net_{network}.pth"
        if copy_if_needed(source_dir / name, dest_dir / name, args.force):
            copied.append(name)
        else:
            kept.append(name)

    print(f"[bootstrap] source={source_dir}")
    print(f"[bootstrap] dest={dest_dir}")
    print(f"[bootstrap] copied={copied}")
    print(f"[bootstrap] kept_existing={kept}")


if __name__ == "__main__":
    main()
