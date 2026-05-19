from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xraygen.explain.gradcam import GradCAM


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

SPATIAL_CLASSES = ["isolated", "overlap"]
THREAT_CLASSES = ["non_contraband", "contraband"]

DEFAULT_SPATIAL_MODEL_DIR = (
    ROOT / "models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/spatial_overlap_isolated"
)
DEFAULT_THREAT_MODEL_DIR = (
    ROOT / "models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/threat_contraband_noncontraband"
)
DEFAULT_MULTIHEAD_MODEL = (
    ROOT
    / "models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2/gray_multihead_itemmask_optional_slow/checkpoints/train_best_checkpoint.pt"
)

GRAY_MEAN = (0.5,)
GRAY_STD = (0.25,)
MASK_MEAN = (0.5,)
MASK_STD = (0.5,)


class SimpleCNN_Binary(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)


class SimpleCNN_MultiHead(nn.Module):
    def __init__(self, in_channels: int = 2, dropout: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.shared_fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_head = nn.Linear(512, 2)
        self.threat_head = nn.Linear(512, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.shared_fc(x)
        return self.spatial_head(x), self.threat_head(x)


class MultiHeadGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, head_name: str):
        self.model = model
        self.target_layer = target_layer
        self.head_name = head_name
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_input, grad_output):
            del grad_input
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(self, x: torch.Tensor, target_class: int) -> np.ndarray:
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        spatial_logits, threat_logits = self.model(x)
        if self.head_name == "spatial":
            logits = spatial_logits
        elif self.head_name == "threat":
            logits = threat_logits
        else:
            raise ValueError("head_name must be 'spatial' or 'threat'")

        score = logits[:, int(target_class)].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        if (cam_max - cam_min).abs() > 1e-12:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        if was_training:
            self.model.train()

        return cam.detach().cpu().numpy().astype(np.float32)


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def list_image_files(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No image files found in: {folder}")
    return paths


def find_checkpoint(model_dir_or_file: Path) -> Path:
    if model_dir_or_file.is_file():
        return model_dir_or_file

    candidates = [
        model_dir_or_file / "checkpoints" / "best.pt",
        model_dir_or_file / "checkpoints" / "train_best.pt",
        model_dir_or_file / "checkpoints" / "best_checkpoint.pt",
        model_dir_or_file / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "No checkpoint found. Checked:\n" + "\n".join(str(p) for p in candidates)
    )


def clean_state_dict(ckpt):
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint did not contain a state_dict. Got: {type(state)}")

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def load_binary_model(model_dir_or_file: Path, device: torch.device, dropout: float) -> tuple[nn.Module, Path]:
    ckpt_path = find_checkpoint(model_dir_or_file)
    model = SimpleCNN_Binary(in_channels=1, num_classes=2, dropout=dropout).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(clean_state_dict(ckpt), strict=True)
    model.eval()
    return model, ckpt_path


def load_multihead_model(
    model_dir_or_file: Path,
    device: torch.device,
    in_channels: int,
    dropout: float,
) -> tuple[nn.Module, Path]:
    ckpt_path = find_checkpoint(model_dir_or_file)
    model = SimpleCNN_MultiHead(in_channels=in_channels, dropout=dropout).to(device)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(clean_state_dict(ckpt), strict=True)
    model.eval()
    return model, ckpt_path


def make_black_border_white(pil_img: Image.Image, threshold: int = 15) -> Image.Image:
    arr = np.array(pil_img.convert("L"))
    arr[arr <= threshold] = 255
    return Image.fromarray(arr).convert("L")


def remove_black_bands_keep_objects(
    pil_img: Image.Image,
    threshold: int = 15,
    row_black_ratio: float = 0.85,
    col_black_ratio: float = 0.85,
) -> Image.Image:
    """
    Match ClassifierModels/SimpleCNN copy.ipynb: remove mostly-black padding
    bands while preserving dark object pixels inside the image.
    """
    arr = np.array(pil_img.convert("L")).astype(np.uint8)
    black = arr <= threshold
    black_rows = black.mean(axis=1) >= row_black_ratio
    black_cols = black.mean(axis=0) >= col_black_ratio
    arr[black_rows, :] = 255
    arr[:, black_cols] = 255
    return Image.fromarray(arr).convert("L")


def crop_white_border(pil_img: Image.Image, threshold: int = 252, margin: int = 0) -> Image.Image:
    arr = np.array(pil_img.convert("L"))
    mask = arr < threshold
    if not mask.any():
        return pil_img

    ys, xs = np.where(mask)
    left = max(int(xs.min()) - margin, 0)
    right = min(int(xs.max()) + margin, pil_img.size[0] - 1)
    top = max(int(ys.min()) - margin, 0)
    bottom = min(int(ys.max()) + margin, pil_img.size[1] - 1)
    return pil_img.crop((left, top, right + 1, bottom + 1))


def load_processed_image(
    image_path: Path,
    clean_border: bool,
    crop_border: bool,
    crop_margin: int,
) -> Image.Image:
    img = Image.open(image_path)
    if clean_border:
        img = make_black_border_white(img)
    else:
        img = img.convert("L")

    if crop_border:
        img = crop_white_border(img, margin=crop_margin)
    return img


def load_simplecnn_copy_image(image_path: Path, clean_bands: bool) -> Image.Image:
    """
    Load generated image the same way as ClassifierModels/SimpleCNN copy.ipynb:
    grayscale, optional large-black-band cleanup, then no crop or resize here.
    The model transform later performs keep-ratio pad to 512.
    """
    img = Image.open(image_path).convert("L")
    if clean_bands:
        img = remove_black_bands_keep_objects(img)
    return img


def build_transform(image_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Grayscale(num_output_channels=1),
            T.Resize(int(image_size * 1.10)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(GRAY_MEAN, GRAY_STD),
        ]
    )


def resize_keep_ratio_and_pad(
    pil_img: Image.Image,
    target_size: int,
    interpolation: TF.InterpolationMode,
    fill: int,
) -> Image.Image:
    img = pil_img.convert("L")
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    img = TF.resize(img, [new_h, new_w], interpolation=interpolation)

    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    pad_right = target_size - new_w - pad_left
    pad_bottom = target_size - new_h - pad_top

    return TF.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=fill)


def build_simplecnn_input(pil_img: Image.Image, image_size: int, in_channels: int) -> torch.Tensor:
    img = resize_keep_ratio_and_pad(
        pil_img,
        target_size=image_size,
        interpolation=TF.InterpolationMode.BILINEAR,
        fill=255,
    )
    img_t = TF.normalize(TF.to_tensor(img), GRAY_MEAN, GRAY_STD)
    if in_channels == 1:
        return img_t

    if in_channels != 2:
        raise ValueError("SimpleCNN multi-head mode supports in_channels 1 or 2.")

    blank_mask = Image.new("L", pil_img.size, 0)
    mask = resize_keep_ratio_and_pad(
        blank_mask,
        target_size=image_size,
        interpolation=TF.InterpolationMode.NEAREST,
        fill=0,
    )
    mask_t = TF.normalize(TF.to_tensor(mask), MASK_MEAN, MASK_STD)
    return torch.cat([img_t, mask_t], dim=0)


def tensor_to_display_rgb(x_tensor: torch.Tensor) -> np.ndarray:
    x = x_tensor.detach().cpu().squeeze(0)[0].numpy()
    x = (x * GRAY_STD[0]) + GRAY_MEAN[0]
    x = np.clip(x, 0, 1)
    img_u8 = (x * 255).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)


def make_gradcam_overlay(display_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(display_rgb, 1 - alpha, heatmap, alpha, 0)


def predict(model: nn.Module, x: torch.Tensor) -> tuple[int, list[float]]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return int(np.argmax(probs)), [float(v) for v in probs]


def predict_multihead(model: nn.Module, x: torch.Tensor) -> tuple[int, list[float], int, list[float]]:
    with torch.no_grad():
        spatial_logits, threat_logits = model(x)
        spatial_probs = torch.softmax(spatial_logits, dim=1).squeeze(0).detach().cpu().numpy()
        threat_probs = torch.softmax(threat_logits, dim=1).squeeze(0).detach().cpu().numpy()
    return (
        int(np.argmax(spatial_probs)),
        [float(v) for v in spatial_probs],
        int(np.argmax(threat_probs)),
        [float(v) for v in threat_probs],
    )


def should_export_gradcam(index: int, args, spatial_pred: str, threat_pred: str) -> bool:
    if args.gradcam_limit <= 0 or index >= args.gradcam_limit:
        return False

    if args.gradcam_mode == "first":
        return True
    if args.gradcam_mode == "contraband":
        return threat_pred == "contraband"
    if args.gradcam_mode == "overlap_or_contraband":
        return spatial_pred == "overlap" or threat_pred == "contraband"
    if args.gradcam_mode == "interesting":
        return spatial_pred == "isolated" or threat_pred == "non_contraband"

    raise ValueError(f"Unknown gradcam mode: {args.gradcam_mode}")


def export_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    task_name: str,
    pred_name: str,
    image_stem: str,
    out_dir: Path,
    alpha: float,
) -> str:
    target_layer = model.features[-4]
    gradcam = GradCAM(model, target_layer)
    try:
        cam = gradcam(x)
    finally:
        gradcam.remove()

    display_rgb = tensor_to_display_rgb(x)
    overlay = make_gradcam_overlay(display_rgb, cam, alpha=alpha)

    task_dir = out_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = image_stem.replace("/", "_")
    out_path = task_dir / f"{safe_stem}__{task_name}__{pred_name}.png"
    Image.fromarray(overlay).save(out_path)
    return str(out_path)


def export_multihead_gradcam(
    model: nn.Module,
    x: torch.Tensor,
    head_name: str,
    target_class: int,
    pred_name: str,
    image_stem: str,
    out_dir: Path,
    alpha: float,
) -> str:
    task_name = "spatial_overlap_isolated" if head_name == "spatial" else "threat_contraband_noncontraband"
    target_layer = model.features[-4]
    gradcam = MultiHeadGradCAM(model, target_layer, head_name=head_name)
    try:
        cam = gradcam(x, target_class=target_class)
    finally:
        gradcam.remove()

    display_rgb = tensor_to_display_rgb(x)
    overlay = make_gradcam_overlay(display_rgb, cam, alpha=alpha)

    task_dir = out_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = image_stem.replace("/", "_")
    out_path = task_dir / f"{safe_stem}__{task_name}__{pred_name}.png"
    Image.fromarray(overlay).save(out_path)
    return str(out_path)


def summarize(rows: list[dict], spatial_ckpt: Path | None, threat_ckpt: Path | None, args) -> dict:
    spatial_counts = Counter(r["spatial_pred"] for r in rows)
    threat_counts = Counter(r["threat_pred"] for r in rows)

    summary = {
        "num_images": len(rows),
        "image_dir": str(resolve_path(args.image_dir)),
        "recursive": bool(args.recursive),
        "model_mode": args.model_mode,
        "model_path": str(resolve_path(args.model)) if args.model else None,
        "spatial_model": str(spatial_ckpt) if spatial_ckpt is not None else None,
        "threat_model": str(threat_ckpt) if threat_ckpt is not None else None,
        "input_channels": args.input_channels,
        "image_size": int(args.image_size),
        "preprocess": "simplecnn_keep_ratio_pad" if args.model_mode.startswith("multihead") else "resize_center_crop",
        "simplecnn_copy_compatible": bool(args.model_mode.startswith("multihead")),
        "blank_mask_channel": bool(args.model_mode == "multihead_itemmask"),
        "black_band_cleanup": bool(args.model_mode.startswith("multihead") and not args.no_clean_border),
        "crop_applied": bool(args.model_mode == "two_models" and not args.no_crop_border),
        "spatial_class_counts": dict(spatial_counts),
        "threat_class_counts": dict(threat_counts),
        "mean_spatial_prob_overlap": float(np.mean([r["spatial_prob_overlap"] for r in rows])),
        "mean_threat_prob_contraband": float(np.mean([r["threat_prob_contraband"] for r in rows])),
        "gradcam_mode": args.gradcam_mode,
        "gradcam_limit": int(args.gradcam_limit),
        "max_images": args.max_images,
    }

    if args.expected_spatial:
        hits = [r["spatial_pred"] == args.expected_spatial for r in rows]
        summary["expected_spatial"] = args.expected_spatial
        summary["spatial_expected_match_rate"] = float(np.mean(hits))

    if args.expected_threat:
        hits = [r["threat_pred"] == args.expected_threat for r in rows]
        summary["expected_threat"] = args.expected_threat
        summary["threat_expected_match_rate"] = float(np.mean(hits))

    return summary


def write_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "spatial_pred",
        "spatial_pred_id",
        "spatial_prob_isolated",
        "spatial_prob_overlap",
        "threat_pred",
        "threat_pred_id",
        "threat_prob_non_contraband",
        "threat_prob_contraband",
        "spatial_gradcam",
        "threat_gradcam",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate generated X-ray images with SimpleCNN classifiers and optional Grad-CAM overlays."
    )
    ap.add_argument(
        "--image_dir",
        default="datasets/_dvc_generated_combo/generated",
        help="Directory containing generated images to score.",
    )
    ap.add_argument("--recursive", action="store_true", help="Scan image_dir recursively.")
    ap.add_argument("--max_images", type=int, default=None, help="Optional cap for quick smoke tests.")
    ap.add_argument(
        "--model_mode",
        choices=["multihead_itemmask", "multihead_gray", "two_models"],
        default="multihead_itemmask",
        help="SimpleCNN evaluation mode. multihead_itemmask matches SimpleCNN/4_validate_model.ipynb.",
    )
    ap.add_argument(
        "--model",
        default=str(DEFAULT_MULTIHEAD_MODEL),
        help="Multi-head SimpleCNN checkpoint or model directory.",
    )
    ap.add_argument(
        "--spatial_model",
        default=str(DEFAULT_SPATIAL_MODEL_DIR),
        help="Spatial classifier directory or checkpoint file.",
    )
    ap.add_argument(
        "--threat_model",
        default=str(DEFAULT_THREAT_MODEL_DIR),
        help="Threat classifier directory or checkpoint file.",
    )
    ap.add_argument(
        "--out_dir",
        default="reports/generated_cnn_gradcam",
        help="Output directory for CSV, JSON, and Grad-CAM images.",
    )
    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument(
        "--input_channels",
        type=int,
        choices=[1, 2],
        default=2,
        help="Multi-head SimpleCNN input channels. Use 2 for image + blank item-mask channel.",
    )
    ap.add_argument("--batch_size", type=int, default=1, help="Reserved for future batching; current Grad-CAM path is per-image.")
    ap.add_argument("--dropout", type=float, default=0.5, help="Dropout value used by the classifier head.")
    ap.add_argument("--no_clean_border", action="store_true", help="Do not turn black borders white before scoring.")
    ap.add_argument("--no_crop_border", action="store_true", help="Do not crop white border before resize/crop.")
    ap.add_argument("--crop_margin", type=int, default=0)
    ap.add_argument(
        "--gradcam_mode",
        choices=["first", "contraband", "overlap_or_contraband", "interesting"],
        default="first",
        help="Which images should receive Grad-CAM overlays.",
    )
    ap.add_argument("--gradcam_limit", type=int, default=32, help="Maximum Grad-CAM overlays per task.")
    ap.add_argument("--gradcam_alpha", type=float, default=0.40)
    ap.add_argument("--expected_spatial", choices=SPATIAL_CLASSES, default=None)
    ap.add_argument("--expected_threat", choices=THREAT_CLASSES, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_dir = resolve_path(args.image_dir)
    out_dir = resolve_path(args.out_dir)
    gradcam_dir = out_dir / "gradcam"
    out_csv = out_dir / "predictions.csv"
    out_json = out_dir / "summary.json"

    image_paths = list_image_files(image_dir, recursive=args.recursive)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
        if not image_paths:
            raise FileNotFoundError(f"--max_images left no images to evaluate in: {image_dir}")
    if args.model_mode == "two_models":
        transform = build_transform(args.image_size)
        spatial_model, spatial_ckpt = load_binary_model(resolve_path(args.spatial_model), device, args.dropout)
        threat_model, threat_ckpt = load_binary_model(resolve_path(args.threat_model), device, args.dropout)
        multihead_model = None
        multihead_ckpt = None
    else:
        input_channels = 2 if args.model_mode == "multihead_itemmask" else 1
        args.input_channels = input_channels
        multihead_model, multihead_ckpt = load_multihead_model(
            resolve_path(args.model),
            device,
            in_channels=input_channels,
            dropout=args.dropout,
        )
        spatial_model = None
        threat_model = None
        spatial_ckpt = multihead_ckpt
        threat_ckpt = multihead_ckpt

    print(f"[info] device: {device}")
    print(f"[info] images: {len(image_paths)} from {image_dir}")
    print(f"[info] model mode: {args.model_mode}")
    if args.model_mode == "two_models":
        print(f"[info] spatial model: {spatial_ckpt}")
        print(f"[info] threat model: {threat_ckpt}")
    else:
        print(f"[info] multi-head model: {multihead_ckpt}")
        print(f"[info] input channels: {args.input_channels}")
        print("[info] preprocessing: keep ratio + pad, no center crop")

    rows: list[dict] = []
    exported = 0

    for i, image_path in enumerate(image_paths):
        if args.model_mode == "two_models":
            pil_img = load_processed_image(
                image_path,
                clean_border=not args.no_clean_border,
                crop_border=not args.no_crop_border,
                crop_margin=args.crop_margin,
            )
            x = transform(pil_img).unsqueeze(0).to(device)
            assert spatial_model is not None and threat_model is not None
            spatial_pred_id, spatial_probs = predict(spatial_model, x)
            threat_pred_id, threat_probs = predict(threat_model, x)
        else:
            pil_img = load_simplecnn_copy_image(
                image_path,
                clean_bands=not args.no_clean_border,
            )
            assert multihead_model is not None
            x = build_simplecnn_input(pil_img, args.image_size, args.input_channels).unsqueeze(0).to(device)
            spatial_pred_id, spatial_probs, threat_pred_id, threat_probs = predict_multihead(multihead_model, x)

        spatial_pred = SPATIAL_CLASSES[spatial_pred_id]
        threat_pred = THREAT_CLASSES[threat_pred_id]

        spatial_cam_path = ""
        threat_cam_path = ""
        if should_export_gradcam(exported, args, spatial_pred, threat_pred):
            rel_stem = image_path.relative_to(image_dir).with_suffix("").as_posix()
            if args.model_mode == "two_models":
                assert spatial_model is not None and threat_model is not None
                spatial_cam_path = export_gradcam(
                    spatial_model,
                    x,
                    "spatial_overlap_isolated",
                    spatial_pred,
                    rel_stem,
                    gradcam_dir,
                    alpha=args.gradcam_alpha,
                )
                threat_cam_path = export_gradcam(
                    threat_model,
                    x,
                    "threat_contraband_noncontraband",
                    threat_pred,
                    rel_stem,
                    gradcam_dir,
                    alpha=args.gradcam_alpha,
                )
            else:
                assert multihead_model is not None
                spatial_cam_path = export_multihead_gradcam(
                    multihead_model,
                    x,
                    "spatial",
                    spatial_pred_id,
                    spatial_pred,
                    rel_stem,
                    gradcam_dir,
                    alpha=args.gradcam_alpha,
                )
                threat_cam_path = export_multihead_gradcam(
                    multihead_model,
                    x,
                    "threat",
                    threat_pred_id,
                    threat_pred,
                    rel_stem,
                    gradcam_dir,
                    alpha=args.gradcam_alpha,
                )
            exported += 1

        rows.append(
            {
                "image": str(image_path.relative_to(ROOT) if image_path.is_relative_to(ROOT) else image_path),
                "spatial_pred": spatial_pred,
                "spatial_pred_id": spatial_pred_id,
                "spatial_prob_isolated": spatial_probs[0],
                "spatial_prob_overlap": spatial_probs[1],
                "threat_pred": threat_pred,
                "threat_pred_id": threat_pred_id,
                "threat_prob_non_contraband": threat_probs[0],
                "threat_prob_contraband": threat_probs[1],
                "spatial_gradcam": spatial_cam_path,
                "threat_gradcam": threat_cam_path,
            }
        )

        if (i + 1) % 25 == 0 or i + 1 == len(image_paths):
            print(f"[eval] {i + 1}/{len(image_paths)}")

    summary = summarize(rows, spatial_ckpt, threat_ckpt, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_csv)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[done] predictions:", out_csv)
    print("[done] summary:", out_json)
    print("[done] gradcam overlays:", gradcam_dir)
    print("[summary] spatial:", summary["spatial_class_counts"])
    print("[summary] threat:", summary["threat_class_counts"])


if __name__ == "__main__":
    main()
