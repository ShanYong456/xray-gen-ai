from pathlib import Path
import cv2
import numpy as np

# =========================
# Config
# =========================
INPUT_DIR = Path("data/raw/Empty")

OUT_GRAY  = Path("data/interim/GAN/Empty")
OUT_COLOR = Path("data/interim/GAN/Empty")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
TARGET_SIZE = (1500, 1000)  # (width, height)

# Create output dirs
OUT_GRAY.mkdir(parents=True, exist_ok=True)
OUT_COLOR.mkdir(parents=True, exist_ok=True)

clahe_gray  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
clahe_color = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

# =========================
# Helpers
# =========================
def ensure_uint8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def trim_black_borders(img, thresh=8):
    """Remove near-black borders by bounding box of non-black pixels."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = gray > thresh
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]

import random

def resize_crop_fill_random(img, target_size=(1500, 1000), jitter=0.10):
    """Scale to cover target size then random crop (adds translation diversity)."""
    h, w = img.shape[:2]
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    max_x = max(new_w - target_w, 0)
    max_y = max(new_h - target_h, 0)

    # jitter as fraction of available crop range
    jx = int(max_x * jitter)
    jy = int(max_y * jitter)

    # choose crop around center but random within jitter window (stable)
    cx = max_x // 2
    cy = max_y // 2
    x0 = np.clip(cx + random.randint(-jx, jx), 0, max_x)
    y0 = np.clip(cy + random.randint(-jy, jy), 0, max_y)

    return resized[y0:y0 + target_h, x0:x0 + target_w]


# =========================
# Main
# =========================
image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
print(f"Found {len(image_paths)} raw images in {INPUT_DIR}")

n_saved = 0
n_failed = 0

for img_path in image_paths:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Could not read: {img_path}")
        n_failed += 1
        continue

    base_name = img_path.stem

    # 1) trim borders first (works on original)
    img_bgr = trim_black_borders(img_bgr, thresh=8)

    # -----------------------
    # Grayscale pipeline
    # -----------------------
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)   # <-- define it first
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = clahe_gray.apply(ensure_uint8(gray))
    gray = resize_crop_fill_random(gray, TARGET_SIZE)

    # convert FINAL grayscale to 3-channel for StyleGAN
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   

    # -----------------------
    # Color pipeline (CLAHE on L channel)
    # -----------------------
    bgr = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe_color.apply(ensure_uint8(l))
    lab = cv2.merge((l, a, b))
    color = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    color = resize_crop_fill_random(color, TARGET_SIZE)

    # Save
    out_name = f"{base_name}.png"
    cv2.imwrite(str(OUT_GRAY / out_name), gray3)
    cv2.imwrite(str(OUT_COLOR / out_name), color)

    n_saved += 1

print("\nDone preprocessing (no augmentation, no split).")
print(f"Saved: {n_saved} | Failed: {n_failed}")
print(f"Outputs:\n - {OUT_GRAY}\n - {OUT_COLOR}")
