from pathlib import Path
import cv2
import numpy as np
import random

# =========================
# Config
# =========================
INPUT_DIR = Path("data/raw/Stage1/Color")

OUT_GRAY  = Path("data/interim/GAN/Stage1/gray_clahe_1500x1000_noborder_aug")
OUT_COLOR = Path("data/interim/GAN/Stage1/color_clahe_1500x1000_noborder_aug")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
TARGET_SIZE = (1500, 1000)  # (width, height)

N_AUG_PER_IMAGE = 8
SEED = 42
rng = random.Random(SEED)
np.random.seed(SEED)

# 3 parts (rename however you want)
PARTS = ("part1", "part2", "part3")

# Create output dirs
for p in PARTS:
    (OUT_GRAY / p).mkdir(parents=True, exist_ok=True)
    (OUT_COLOR / p).mkdir(parents=True, exist_ok=True)

clahe_gray  = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
clahe_color = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))

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

def resize_crop_fill(img, target_size=(1500, 1000)):
    """Scale to cover target size then center crop. No padding."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    x0 = (new_w - target_w) // 2
    y0 = (new_h - target_h) // 2
    return resized[y0:y0 + target_h, x0:x0 + target_w]

# =========================
# Augmentations
# =========================
def random_affine(img, max_rotate=5.0, max_translate=0.02, max_scale=0.03):
    h, w = img.shape[:2]
    angle = rng.uniform(-max_rotate, max_rotate)
    scale = rng.uniform(1.0 - max_scale, 1.0 + max_scale)
    tx = rng.uniform(-max_translate, max_translate) * w
    ty = rng.uniform(-max_translate, max_translate) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

def random_gamma(img, gamma_range=(0.85, 1.15)):
    gamma = rng.uniform(*gamma_range)
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def random_noise(img, sigma_range=(2, 8)):
    sigma = rng.uniform(*sigma_range)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def random_flip(img):
    if rng.random() < 0.5:
        return cv2.flip(img, 1)  # horizontal only
    return img

def augment(img):
    out = random_flip(img)
    out = random_affine(out, max_rotate=2.0, max_translate=0.015, max_scale=0.02)
    out = random_gamma(out, gamma_range=(0.9, 1.1))
    out = random_noise(out, sigma_range=(2, 6))
    return out

# =========================
# 1) List raw images
# =========================
image_paths = sorted([p for p in INPUT_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS])
print(f"Found {len(image_paths)} raw images in {INPUT_DIR}")

# =========================
# 2) AUGMENT FIRST -> collect all outputs in memory (then shuffle + split)
# =========================
# We will create a list of "items" where each item corresponds to ONE output image
# and we’ll shuffle those items and distribute across part1/2/3.
#
# To keep gray/color paired, we store both arrays together, then write together.

all_items = []  # list of dicts: {"base": str, "kind": "orig"/"augXX", "gray": img, "color": img}

for img_path in image_paths:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Could not read: {img_path}")
        continue

    base_name = img_path.stem

    # trim borders
    img_bgr = trim_black_borders(img_bgr, thresh=8)

    # ---- Base grayscale ----
    gray_base = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_base = cv2.GaussianBlur(gray_base, (5, 5), 0)
    gray_base = clahe_gray.apply(ensure_uint8(gray_base))
    gray_base = resize_crop_fill(gray_base, TARGET_SIZE)

    # ---- Base color ----
    bgr_base = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    lab = cv2.cvtColor(bgr_base, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe_color.apply(ensure_uint8(l))
    lab = cv2.merge((l, a, b))
    color_base = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    color_base = resize_crop_fill(color_base, TARGET_SIZE)

    # store original
    all_items.append({
        "base": base_name,
        "kind": "orig",
        "gray": gray_base,
        "color": color_base
    })

    # store augmented copies
    for i in range(N_AUG_PER_IMAGE):
        gray_aug  = resize_crop_fill(augment(gray_base), TARGET_SIZE)
        color_aug = resize_crop_fill(augment(color_base), TARGET_SIZE)
        all_items.append({
            "base": base_name,
            "kind": f"aug{i:02d}",
            "gray": gray_aug,
            "color": color_aug
        })

print(f"Total output items (orig + aug): {len(all_items)}")

# =========================
# 3) Shuffle ALL outputs, then split into 3 parts
# =========================
rng.shuffle(all_items)

counts = {p: 0 for p in PARTS}

for idx, item in enumerate(all_items):
    part = PARTS[idx % len(PARTS)]  # round-robin distribution after shuffle
    base = item["base"]
    kind = item["kind"]

    out_name = f"{base}_{kind}.png"

    cv2.imwrite(str((OUT_GRAY / part) / out_name), item["gray"])
    cv2.imwrite(str((OUT_COLOR / part) / out_name), item["color"])

    counts[part] += 1

print("Augmented first, then SHUFFLED all outputs, then split into 3 parts.")
print("Saved items per part:", counts)
print(f"Saved to:\n - {OUT_GRAY}\n - {OUT_COLOR}")
