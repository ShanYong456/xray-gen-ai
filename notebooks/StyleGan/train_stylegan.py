#!/usr/bin/env python3
import subprocess
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset zip, e.g. data/stylegan/myset_256.zip")
    ap.add_argument("--outdir", default="models/stylegan/runs", help="Output runs folder")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--cfg", default="paper256", help="paper256 / paper128 etc.")
    ap.add_argument("--aug", default="ada", help="ada recommended for small data")
    ap.add_argument("--mirror", type=int, default=0, help="0 disable flip, 1 enable flip")
    ap.add_argument("--kimg", type=int, default=5000, help="Training length in kimg")
    ap.add_argument("--resume", default="", help="Optional .pkl snapshot to resume")
    ap.add_argument("--transfer", default="", help="Optional pretrained .pkl to transfer from")
    ap.add_argument("--stylegan_repo", default="external/stylegan2-ada-pytorch")
    ap.add_argument("--snap", type=int, default=1, help="How often to print progress")

    # NEW: learning rates
    ap.add_argument("--glr", type=float, default=None,
                    help="Generator learning rate (default: 0.002)")
    ap.add_argument("--dlr", type=float, default=None,
                    help="Discriminator learning rate (default: 0.002)")

    # Safe for PyTorch 2.x
    ap.add_argument("--gamma", type=float, default=0.0,
                    help="R1 regularization strength (keep 0.0 for torch 2.x)")

    args = ap.parse_args()

    repo = Path(args.stylegan_repo)
    train_py = repo / "train.py"
    if not train_py.exists():
        raise SystemExit(f"train.py not found at: {train_py}")

    cmd = [
        "python", str(train_py),
        f"--outdir={args.outdir}",
        f"--data={args.data}",
        f"--gpus={args.gpus}",
        f"--batch={args.batch}",
        f"--cfg={args.cfg}",
        f"--aug={args.aug}",
        f"--mirror={args.mirror}",
        f"--kimg={args.kimg}",
        f"--snap={args.snap}",
        f"--gamma={args.gamma}",
    ]

    # Only add if user specifies
    if args.glr is not None:
        cmd.append(f"--glr={args.glr}")
    if args.dlr is not None:
        cmd.append(f"--dlr={args.dlr}")

    if args.resume:
        cmd.append(f"--resume={args.resume}")
    if args.transfer:
        cmd.append(f"--transfer={args.transfer}")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()


"""

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

which nvcc
nvcc --version



PYTHONWARNINGS='ignore:conv2d_gradfix not supported.*:UserWarning' \
python3 external/stylegan2-ada-pytorch/train.py \
  --outdir=models/generator/stylegan/runs \
  --data=data/interim/GAN/Stage1/myset_256.zip \
  --gpus=1 --batch=8 --cfg=paper256 \
  --aug=ada --augpipe=color \
  --mirror=0 \
  --kimg=120 --metrics=none --snap=1



python external/stylegan2-ada-pytorch/patch_stylegan2_auto_lr.py
####
PYTHONWARNINGS='ignore:conv2d_gradfix not supported.*:UserWarning' python3 external/stylegan2-ada-pytorch/train.py   --outdir=models/generator/stylegan/runs   --data=data/interim/GAN/Stage1/myset_256.zip   --gpus=1 --batch=8 --cfg=paper256   --aug=ada --augpipe=color   --mirror=0   --kimg=150 --metrics=none --snap=1  --gamma=0   --resume=models/generator/stylegan/runs/V17_testing/network-snapshot-000120.pkl \

V2 (without ada)
PYTHONWARNINGS='ignore:conv2d_gradfix not supported.*:UserWarning' python3 external/stylegan2-ada-pytorch/train.py   --outdir=models/generator/stylegan/runs   --data=data/interim/GAN/Stage1/myset_256.zip   --gpus=1 --batch=8 --cfg=paper256   --aug=noaug  --mirror=0   --kimg=300 --metrics=none --snap=1  --gamma=5   --resume=models/generator/stylegan/runs/V21/network-snapshot-000150.pkl \

// ada = 0.2
PYTHONWARNINGS='ignore:conv2d_gradfix not supported.*:UserWarning' python3 external/stylegan2-ada-pytorch/train.py   --outdir=models/generator/stylegan/runs   --data=data/interim/GAN/Stage1/myset_256.zip   --gpus=1 --batch=8 --cfg=paper256   --aug=ada --augpipe=color --target=0.2  --mirror=0   --kimg=300 --metrics=none --snap=1  --gamma=5   --resume=models/generator/stylegan/runs/V21/network-snapshot-000150.pkl \





WITH TRANSFER LEARNING:

PYTHONWARNINGS="ignore::UserWarning" \
PYTHONWARNINGS='ignore:conv2d_gradfix not supported.*:UserWarning' \
python external/stylegan2-ada-pytorch/train.py \
  --outdir=models/generator/stylegan/runs \
  --data=data/interim/GAN/Stage1/myset_256.zip \
  --cfg=paper256 \
  --gpus=1 \
  --batch=8 \
  --mirror=0 \
  --aug=ada \
  --kimg=300 \
  --metrics=none \
  --snap=5 \
  --resume=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl


"""