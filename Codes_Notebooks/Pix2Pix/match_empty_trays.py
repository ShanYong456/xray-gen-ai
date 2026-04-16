#!/usr/bin/env python3
"""Match and rename/copy empty-tray images to have the same basename as the
paired AB samples.

Usage:
    python scripts/match_empty_trays.py \
        --ab_dir datasets/non_contraband_V1/train \
        --empty_dir data/interim/GAN/Empty \
        [--dry-run] [--copy]

The script looks for an empty image whose filename contains the same timestamp
as the AB file (everything after the first dash and before `_tr`). If found,
its path is either printed or copied/renamed to match the AB basename.

This makes the filenames line up so the loader's exact-lookup logic succeeds.

If no matching empty image exists the script reports it; you may then need to
manually acquire or generate the missing empty tray image.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Link empty trays to AB names")
    parser.add_argument("--ab_dir", required=True, help="Directory containing AB images")
    parser.add_argument("--empty_dir", required=True, help="Directory of empty tray images")
    parser.add_argument("--dry-run", action="store_true", help="Just print actions without performing them")
    parser.add_argument("--copy", action="store_true", help="Copy matching empty files rather than rename")
    args = parser.parse_args()

    ab_dir = Path(args.ab_dir)
    empty_dir = Path(args.empty_dir)
    if not ab_dir.is_dir():
        raise FileNotFoundError(f"AB directory not found: {ab_dir}")
    if not empty_dir.is_dir():
        raise FileNotFoundError(f"Empty directory not found: {empty_dir}")

    empties = list(empty_dir.iterdir())
    if not empties:
        print("Empty directory appears to be empty")
        return

    for ab in sorted(ab_dir.glob("*.png")):
        bname = ab.name
        # extract timestamp portion between first dash and '_tr'
        if "-" not in bname or "_tr" not in bname:
            print(f"Skipping {bname}: unexpected format")
            continue
        ts = bname.split("-", 1)[1].split("_tr")[0]
        # search for an empty file containing the timestamp
        matches = [e for e in empties if ts in e.name]
        if not matches:
            print(f"no empty image matching timestamp {ts} for {bname}")
            continue
        if len(matches) > 1:
            print(f"multiple empties match {ts} for {bname}: {matches}")
            # choose first
            src = matches[0]
        else:
            src = matches[0]
        dest = empty_dir / bname
        if src.name == dest.name:
            # already good
            continue
        print(f"{src.name} -> {dest.name}")
        if not args.dry_run:
            if args.copy:
                import shutil
                shutil.copy(src, dest)
            else:
                src.rename(dest)

if __name__ == "__main__":
    main()
