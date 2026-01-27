import os
from pathlib import Path

import cv2
import numpy as np


# =====================
# CONFIG
# =====================
INPUT_DIR  = Path("data/raw/Stage0/Gray")
OUTPUT_DIR = Path("data/interim/Stage0/Gray")

TARGET_SIZE = (512, 512)   # (width, height)
OUTPUT_EXT = ".png"        # ".png" or ".tif"

# 🔑 CHOOSE OUTPUT MODE
# "gray"  -> 1 channel grayscale
# "color" -> 3 channel (keep original)
OUTPUT_MODE = "gray"       # <-- change to "color" when needed


# =====================
# HELPERS
# =====================
def load_image(path):
    """Load image exactly as-is (8-bit or 16-bit, any channels)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


def convert_channels(img):
    """Convert image based on OUTPUT_MODE."""
    if OUTPUT_MODE == "gray":
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif OUTPUT_MODE == "color":
        if img.ndim == 2:
            # expand grayscale to 3 channels if needed
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    else:
        raise ValueError(f"Invalid OUTPUT_MODE: {OUTPUT_MODE}")


def normalize_bit_depth(img):
    """
    - If 16-bit: normalize to full 0–65535 range
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
    # Median blur is safe for X-ray (edge-preserving)
    if img.ndim == 2:
        return cv2.medianBlur(img, 3)
    else:
        # Apply per-channel
        return cv2.merge([cv2.medianBlur(c, 3) for c in cv2.split(img)])


# =====================
# MAIN PIPELINE
# =====================
def preprocess_image(in_path, out_path):
    img = load_image(in_path)
    img = convert_channels(img)
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
