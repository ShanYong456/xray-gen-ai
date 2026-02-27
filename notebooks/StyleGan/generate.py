#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="Path to network-snapshot-xxxxxx.pkl")
    ap.add_argument("--outdir", default="models/stylegan/gen", help="Output folder for generated images")
    ap.add_argument("--seeds", default="0-99", help="Seed range like 0-99 or comma list")
    ap.add_argument("--stylegan_repo", default="external/stylegan2-ada-pytorch")
    args = ap.parse_args()

    repo = Path(args.stylegan_repo)
    gen_py = repo / "generate.py"
    if not gen_py.exists():
        raise SystemExit(f"generate.py not found at: {gen_py}")

    cmd = [
        "python", str(gen_py),
        f"--outdir={args.outdir}",
        f"--network={args.network}",
        f"--seeds={args.seeds}",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()


"""
python notebooks/StyleGan/generate.py \
  --network models/generator/stylegan/runs/00017-myset_256-paper256-kimg120-batch8-ada-color/network-snapshot-000120.pkl \
  --outdir models/stylegan/gen \
  --seeds 0-99

  python external/stylegan2-ada-pytorch/generate.py \
  --network=models/generator/stylegan/runs/V21/network-snapshot-000150.pkl \
  --outdir=models/stylegan/gen \
  --seeds=1,25,87,92,1200,6478 \
  --trunc=1 \
  --noise-mode=random

"""