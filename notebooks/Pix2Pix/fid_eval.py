import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "pix2pix"))

from options.test_options import TestOptions
from data import create_dataset
from models import create_model


def tensor_to_u8_rgb(t: torch.Tensor) -> np.ndarray:
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().float().numpy()

    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)

    arr = np.transpose(arr, (1, 2, 0))

    if arr.min() < 0.0:
        arr = (arr + 1.0) * 0.5

    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def to_grayscale_3ch(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def extract_real_B_from_AB(ab_path: Path) -> np.ndarray:
    img = Image.open(ab_path).convert("RGB")
    w, h = img.size
    b = img.crop((w // 2, 0, w, h))
    return np.array(b, dtype=np.uint8)


def save_rgb(path: Path, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def build_test_opt(args, num_threads: int = 0):
    sys.argv = [
        "fid_eval.py",
        "--dataroot", args.dataroot,
        "--name", args.name,
        "--model", "pix2pix",
        "--dataset_mode", "aligned",
        "--direction", "AtoB",
        "--phase", args.phase,
        "--epoch", args.epoch,
        "--num_threads", str(num_threads),
        "--batch_size", "1",
        "--serial_batches",
        "--no_flip",
        "--eval",
        "--input_nc", str(args.input_nc),
        "--output_nc", str(args.output_nc),
        "--netG", args.netG,
        "--netD", args.netD,
        "--n_layers_D", str(args.n_layers_D),
        "--norm", args.norm,
        "--class_nc", str(args.class_nc),
        "--thickness_nc", str(args.thickness_nc),
        "--preprocess", "none",
        "--load_size", "0",
        "--crop_size", "0",
    ]

    if args.use_thickness_channel:
        sys.argv.append("--use_thickness_channel")
    if args.use_edge_channel:
        sys.argv.append("--use_edge_channel")
    if args.use_coord_channels:
        sys.argv.append("--use_coord_channels")

    if args.use_tray_mask:
        sys.argv.append("--use_tray_mask")

        if args.tray_mask_dir:
            sys.argv.extend(["--tray_mask_dir", args.tray_mask_dir])

        if args.tray_mask_path:
            sys.argv.extend(["--tray_mask_path", args.tray_mask_path])

        sys.argv.extend([
            "--tray_mask_thr", str(args.tray_mask_thr),
            "--tray_cc_close_px", str(args.tray_cc_close_px),
            "--tray_mask_dilate_px", str(args.tray_mask_dilate_px),
        ])

    if args.synthetic_blade_mask_dir:
        sys.argv.extend([
            "--synthetic_blade_mask_dir",
            args.synthetic_blade_mask_dir,
        ])

    if args.pad_to_canvas:
        sys.argv.append("--pad_to_canvas")
        sys.argv.extend([
            "--canvas_w", str(args.canvas_w),
            "--canvas_h", str(args.canvas_h),
            "--canvas_fill", str(args.canvas_fill),
        ])

    opt = TestOptions().parse()
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return opt


def pad_to_canvas(img, target_w, target_h, fill=0):
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)

    y_offset = (target_h - h) // 2
    x_offset = (target_w - w) // 2
    canvas[y_offset:y_offset + h, x_offset:x_offset + w] = img
    return canvas


@torch.no_grad()
def save_debug_triplet(save_path: Path, A_tensor, fake_rgb: np.ndarray, real_rgb: np.ndarray):
    A = A_tensor
    if torch.is_tensor(A) and A.dim() == 4:
        A = A[0]

    if torch.is_tensor(A):
        A_np = A.detach().cpu().float().numpy()
    else:
        A_np = np.asarray(A)

    if A_np.ndim == 3 and A_np.shape[0] <= 16:
        A_np = np.transpose(A_np, (1, 2, 0))

    if A_np.ndim == 2:
        A_np = np.repeat(A_np[:, :, None], 3, axis=2)
    elif A_np.ndim == 3 and A_np.shape[2] == 1:
        A_np = np.repeat(A_np, 3, axis=2)
    elif A_np.ndim == 3 and A_np.shape[2] > 3:
        A_np = A_np[:, :, :3]

    if A_np.dtype != np.uint8:
        A_np = A_np.astype(np.float32)
        if A_np.min() < 0.0:
            A_np = (A_np + 1.0) * 0.5
        A_np = np.clip(A_np, 0.0, 1.0)
        A_np = (A_np * 255.0).round().astype(np.uint8)

    H, W = fake_rgb.shape[:2]
    if A_np.shape[:2] != (H, W):
        A_np = cv2.resize(A_np, (W, H), interpolation=cv2.INTER_NEAREST)

    fake_rgb = to_grayscale_3ch(fake_rgb)
    real_rgb = to_grayscale_3ch(real_rgb)

    vis = np.concatenate([A_np, fake_rgb, real_rgb], axis=1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(vis).save(save_path)


@torch.no_grad()
def generate_and_collect(
    opt,
    out_fake_dir: Path,
    out_real_dir: Path,
    max_images: int = None,
    debug_every: int = 10,
):
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    debug_dir = out_fake_dir.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _, data in enumerate(tqdm(dataset, desc="Generating for FID")):
        if max_images is not None and count >= max_images:
            break

        model.set_input(data)
        model.test()

        fake_rgb = tensor_to_u8_rgb(model.fake_B)

        a_path = data["A_paths"][0] if isinstance(data["A_paths"], list) else data["A_paths"]
        ab_path = Path(a_path)
        real_rgb = extract_real_B_from_AB(ab_path)

        H, W = fake_rgb.shape[:2]
        h0, w0 = real_rgb.shape[:2]
        scale = min(W / w0, H / h0)

        new_w = int(w0 * scale)
        new_h = int(h0 * scale)

        real_resized = cv2.resize(real_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        real_rgb = pad_to_canvas(real_resized, W, H, fill=0)

        fake_rgb = to_grayscale_3ch(fake_rgb)
        real_rgb = to_grayscale_3ch(real_rgb)

        if real_rgb.shape[:2] != (H, W):
            real_rgb = cv2.resize(real_rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        filename = f"{count:06d}.png"
        save_rgb(out_fake_dir / filename, fake_rgb)
        save_rgb(out_real_dir / filename, real_rgb)

        if debug_every > 0 and (count % debug_every == 0):
            save_debug_triplet(
                debug_dir / f"debug_{filename}",
                data["A"],
                fake_rgb,
                real_rgb,
            )

        count += 1

    return count


def run_torch_fidelity(fake_dir: Path, real_dir: Path, cuda: bool = True):
    cmd = [
        sys.executable, "-m", "torch_fidelity.fidelity",
        "--fid",
        "--input1", str(fake_dir),
        "--input2", str(real_dir),
        "--json",
    ]
    if cuda and torch.cuda.is_available():
        cmd += ["--gpu", "0"]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--epoch", type=str, default="latest")
    parser.add_argument("--phase", type=str, default="val")
    parser.add_argument("--work_dir", type=str, default="fid_eval_runs")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--keep_images", action="store_true")
    parser.add_argument("--input_nc", type=int, default=6)
    parser.add_argument("--output_nc", type=int, default=3)
    parser.add_argument("--netG", type=str, default="unet_256")
    parser.add_argument("--netD", type=str, default="n_layers")
    parser.add_argument("--n_layers_D", type=int, default=4)
    parser.add_argument("--norm", type=str, default="instance")
    parser.add_argument("--class_nc", type=int, default=2)
    parser.add_argument("--thickness_nc", type=int, default=1)
    parser.add_argument("--use_thickness_channel", action="store_true")
    parser.add_argument("--use_edge_channel", action="store_true")
    parser.add_argument("--use_coord_channels", action="store_true")
    parser.add_argument("--use_tray_mask", action="store_true")
    parser.add_argument("--tray_mask_dir", type=str, default="")
    parser.add_argument("--tray_mask_path", type=str, default="")
    parser.add_argument("--tray_mask_thr", type=float, default=0.5)
    parser.add_argument("--tray_cc_close_px", type=int, default=2)
    parser.add_argument("--tray_mask_dilate_px", type=int, default=0)
    parser.add_argument("--preprocess", type=str, default="none")
    parser.add_argument("--load_size", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=0)
    parser.add_argument("--pad_to_canvas", action="store_true")
    parser.add_argument("--canvas_w", type=int, default=1024)
    parser.add_argument("--canvas_h", type=int, default=1024)
    parser.add_argument("--canvas_fill", type=int, default=0)
    parser.add_argument("--synthetic_blade_mask_dir", type=str, default="")
    args = parser.parse_args()

    work_dir = Path(args.work_dir) / args.name / f"epoch_{args.epoch}"
    fake_dir = work_dir / "fake"
    real_dir = work_dir / "real"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    opt = build_test_opt(args, num_threads=0)
    opt.phase = args.phase

    n = generate_and_collect(
        opt=opt,
        out_fake_dir=fake_dir,
        out_real_dir=real_dir,
        max_images=args.max_images,
        debug_every=10,
    )

    metrics = run_torch_fidelity(fake_dir, real_dir, cuda=True)
    fid_value = metrics.get("frechet_inception_distance", None)

    print(f"[FID] compared {n} image pairs")
    print(f"[FID] value = {fid_value}")

    out_json = work_dir / "metrics.json"
    with open(out_json, "w") as f:
        json.dump(
            {
                "num_images": n,
                "fid": fid_value,
                "raw_metrics": metrics,
            },
            f,
            indent=2,
        )

    print(f"[FID] saved metrics to {out_json}")

    if not args.keep_images:
        shutil.rmtree(fake_dir, ignore_errors=True)
        shutil.rmtree(real_dir, ignore_errors=True)


def compute_fid_for_checkpoint(
    args,
    epoch,
    phase="val",
    max_images=500,
    keep_images=False,
    work_dir="fid_eval_runs",
    debug_every=10,
):
    work_dir = Path(work_dir) / args.name / f"epoch_{epoch}"
    fake_dir = work_dir / "fake"
    real_dir = work_dir / "real"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv[:]
    try:
        args.phase = phase
        args.epoch = str(epoch)

        opt = build_test_opt(args, num_threads=0)
        opt.phase = phase
        opt.epoch = str(epoch)
        opt.isTrain = False

        n = generate_and_collect(
            opt=opt,
            out_fake_dir=fake_dir,
            out_real_dir=real_dir,
            max_images=max_images,
            debug_every=debug_every,
        )

        metrics = run_torch_fidelity(fake_dir, real_dir, cuda=True)
        fid_value = metrics.get("frechet_inception_distance", None)

        out_json = work_dir / "metrics.json"
        with open(out_json, "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "phase": phase,
                    "num_images": n,
                    "fid": fid_value,
                    "raw_metrics": metrics,
                },
                f,
                indent=2,
            )

        result = {
            "epoch": int(epoch) if str(epoch).isdigit() else epoch,
            "phase": phase,
            "num_images": n,
            "fid": fid_value,
            "raw_metrics": metrics,
            "metrics_json": str(out_json),
        }

        print(f"[FID] epoch={epoch} phase={phase} pairs={n} fid={fid_value}")
        print(f"[FID] saved metrics to {out_json}")

        return result

    finally:
        sys.argv = old_argv
        if not keep_images:
            shutil.rmtree(fake_dir, ignore_errors=True)
            shutil.rmtree(real_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

    """
python notebooks/Pix2Pix/fid_eval.py \

python notebooks/Pix2Pix/fid_eval.py \
  --dataroot datasets/SHAMPOOBLADEWITHTRAY_TGT \
  --name Shampoo_NOBGR_pix2pix_StructCond_V1_Stage18_BladeMaskSyn \
  --epoch latest \
  --phase test \
  --max_images 200 \
  --input_nc 7 \
  --output_nc 3 \
  --netG unet_256 \
  --netD n_layers \
  --n_layers_D 4 \
  --norm instance \
  --class_nc 3 \
  --preprocess none \
  --load_size 0 \
  --crop_size 0 \
  --pad_to_canvas \
  --canvas_w 1024 \
  --canvas_h 1024 \
  --use_thickness_channel \
  --use_edge_channel \
  --use_coord_channels \
  --use_tray_mask \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/test/tray \
  --synthetic_blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/test/blade



    
    """