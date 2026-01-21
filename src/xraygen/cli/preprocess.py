import os
from pathlib import Path

import cv2
import numpy as np


# =====================
# CONFIG
# =====================
INPUT_DIR  = Path("data/raw")
OUTPUT_DIR = Path("data/interim")

TARGET_SIZE = (512, 512)   # (width, height)
OUTPUT_EXT = ".png"        # ".png" or ".tif"


# =====================
# HELPERS
# =====================
def load_image(path):
    # Load image exactly as-is (keeps 8-bit or 16-bit)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


def to_grayscale(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_bit_depth(img):
    """
    - If 16-bit: scale to 0–65535
    - If 8-bit: keep as-is
    """
    if img.dtype == np.uint16:
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = (img * 65535).astype(np.uint16)
    return img


def resize_image(img):
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


def mild_denoise(img):
    # Optional but safe
    return cv2.medianBlur(img, 3)


# =====================
# MAIN PIPELINE
# =====================
def preprocess_image(in_path, out_path):
    img = load_image(in_path)
    img = to_grayscale(img)
    img = normalize_bit_depth(img)
    img = resize_image(img)
    img = mild_denoise(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    for path in INPUT_DIR.rglob("*"):
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            continue

        rel_path = path.relative_to(INPUT_DIR)
        out_path = (OUTPUT_DIR / rel_path).with_suffix(OUTPUT_EXT)

        preprocess_image(path, out_path)
        print(f"Processed: {path} → {out_path}")

    print("✅ Preprocessing complete")


if __name__ == "__main__":
    main()
