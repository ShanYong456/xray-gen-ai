#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Folder of images (square recommended)")
    ap.add_argument("--dest", required=True, help="Output zip path, e.g. data/stylegan/myset.zip")
    ap.add_argument("--width", default="256", help="e.g. 256")
    ap.add_argument("--height", default="256", help="e.g. 256")
    ap.add_argument("--stylegan_repo", default="external/stylegan2-ada-pytorch", help="Path to stylegan2-ada-pytorch")
    args = ap.parse_args()

    stylegan_repo = Path(args.stylegan_repo)
    tool = stylegan_repo / "dataset_tool.py"
    if not tool.exists():
        raise SystemExit(f"dataset_tool.py not found at: {tool}")

    cmd = [
        "python", str(tool),
        f"--source={args.source}",
        f"--dest={args.dest}",
        f"--width={args.width}",
        f"--height={args.height}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print(f" Created dataset zip -> {args.dest}")

if __name__ == "__main__":
    main()


"""

python notebooks/StyleGan/make_zip.py --source data/interim/GAN/Stage1/myset_full_256_pad --dest data/interim/GAN/Stage1/myset_256.zip --width=256 --height=256 



"""