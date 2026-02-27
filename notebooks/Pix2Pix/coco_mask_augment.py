from pathlib import Path
import cv2
import numpy as np
import random

# ============================================================
# PURPOSE
# Build Pix2Pix "aligned" AB dataset (A|B in one PNG) WITHOUT:
#   - NO cropping
#   - NO dropping
#   - NO cut-paste
#
# Always use FULL TRAY:
#   A = full 3-ch color mask
#   B = full real tray image
#
# Augment by applying the SAME geometric transform to A and B
# (keeps alignment) + optional B-only intensity/noise jitter.
#
# EDITS:
#   - Make affine "safe": less rotation/shift, avoid scale-down,
#     retry if too much padding occurs, and less ugly borders.
# ============================================================

# ===== INPUTS =====
IMG_DIR  = Path("data/raw/Contraband/Metal")                    # B
MASK_DIR = Path("data/interim/contraband_metalV2/masks_color")  # A (3ch color masks)

# ===== OUTPUT =====
OUT_ROOT  = Path("datasets/contraband_metal_V3")
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR  = OUT_ROOT / "test"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ===== SETTINGS =====
SIZE = 512
TEST_RATIO = 0.2
SEED = 123

PALETTE_RGB = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
    6: (255, 0, 255),
}
PALETTE_ARRAY = np.array(list(PALETTE_RGB.values()), dtype=np.uint8)

def normalize_mask_to_palette(mask_bgr):
    """Convert mask to EXACT palette colors (fix compression/resize artifacts)."""
    mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = mask.shape

    flat = mask.reshape(-1, 3).astype(np.int16)
    palette = PALETTE_ARRAY.astype(np.int16)

    dists = ((flat[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    nearest = dists.argmin(axis=1)

    corrected = palette[nearest].reshape(h, w, 3).astype(np.uint8)
    return cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR)

# How many augmented duplicates per original image
AUG_PER_IMAGE_TRAIN = 10
AUG_PER_IMAGE_TEST  = 4

# Also save some un-augmented originals
KEEP_ORIG_PER_IMAGE_TRAIN = 3
KEEP_ORIG_PER_IMAGE_TEST  = 1

# Optional: strengthen conditioning with edges (still full mask)
ADD_CONTOUR = True

# ===== Paired geometric augmentation (same for A & B) =====
USE_PAIRED_AFFINE = True

# --- SAFER affine params (tray shouldn't look weird) ---
ROT_DEG = 3                 # was 8; keep small (or set to 0)
SCALE_MIN, SCALE_MAX = 1.00, 1.04   # avoid scale-down to prevent edge padding artifacts
SHIFT_FRAC = 0.02           # was 0.07; smaller shift avoids black borders

# Retry policy: reject transforms that create too much padding on A
AFFINE_TRIES = 10
MAX_BLACK_FRAC = 0.015      # max fraction of pixels that become palette-black due to border fill

# ===== B-only photometric augmentation (does NOT touch A) =====
USE_B_PHOTO_AUG = True
BRIGHTNESS_DELTA = 10     # +/- pixel value
CONTRAST_DELTA = 0.08     # +/- fraction
NOISE_SIGMA = 2.0         # gaussian noise std in pixel space

random.seed(SEED)
np.random.seed(SEED)

def list_pairs(mask_dir, img_dir):
    pairs = []
    for mp in sorted(mask_dir.glob("*.png")):
        ip = img_dir / mp.name
        if ip.exists():
            pairs.append((mp, ip))
    return pairs

def resize_mask(m):
    return cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

def resize_img(i):
    return cv2.resize(i, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def add_contour(mask_bgr):
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    out = mask_bgr.copy()
    out[edges > 0] = (255, 255, 255)
    return out

def apply_b_photo_aug(B):
    """Apply intensity jitter/noise to B only (A must keep exact palette colors)."""
    x = B.astype(np.float32)
    alpha = 1.0 + random.uniform(-CONTRAST_DELTA, CONTRAST_DELTA)   # contrast
    beta  = random.uniform(-BRIGHTNESS_DELTA, BRIGHTNESS_DELTA)     # brightness
    x = x * alpha + beta

    if NOISE_SIGMA > 0:
        noise = np.random.normal(0, NOISE_SIGMA, size=x.shape).astype(np.float32)
        x = x + noise

    return np.clip(x, 0, 255).astype(np.uint8)

def _black_frac(mask_bgr):
    """Fraction of pixels that are exactly (0,0,0) in BGR."""
    black = np.all(mask_bgr == (0, 0, 0), axis=2)
    return float(black.mean())

def _sample_affine(H, W):
    angle = random.uniform(-ROT_DEG, ROT_DEG)
    scale = random.uniform(SCALE_MIN, SCALE_MAX)
    tx = random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * W
    ty = random.uniform(-SHIFT_FRAC, SHIFT_FRAC) * H

    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[:, 2] += (tx, ty)
    return M

def apply_paired_affine_safe(A, B):
    """
    Same affine on A and B, but retries until we don't create
    ugly padding/distortion at the sides (too much black fill).
    """
    H, W = B.shape[:2]

    # IMPORTANT: A should be palette-clean before geometry
    # (optional but helps if masks have stray colors)
    A = normalize_mask_to_palette(A)

    best = None
    best_black = 1e9

    for _ in range(AFFINE_TRIES):
        M = _sample_affine(H, W)

        A2 = cv2.warpAffine(
            A, M, (W, H),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        black = _black_frac(A2)

        # keep the first "good enough"
        if black <= MAX_BLACK_FRAC:
            B2 = cv2.warpAffine(
                B, M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,  # less mirror-weirdness than REFLECT
            )
            return A2, B2

        # otherwise remember best attempt and try again
        if black < best_black:
            best_black = black
            best = (A2, M)

    # fallback: use least-bad transform
    A_best, M_best = best
    B_best = cv2.warpAffine(
        B, M_best, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return A_best, B_best

def save_AB(A, B, out_path):
    AB = np.concatenate([A, B], axis=1)
    cv2.imwrite(str(out_path), AB)

def process_split(pairs, out_dir, tag, aug_per_image, keep_orig_per_image):
    idx = 0
    for mp, ip in pairs:
        A = cv2.imread(str(mp))
        B = cv2.imread(str(ip))
        if A is None or B is None:
            continue

        if A.shape[:2] != B.shape[:2]:
            A = cv2.resize(A, (B.shape[1], B.shape[0]), interpolation=cv2.INTER_NEAREST)

        # --- save some originals (no aug) ---
        for k in range(keep_orig_per_image):
            A0 = normalize_mask_to_palette(A)
            A0 = resize_mask(A0)
            B0 = resize_img(B)
            if ADD_CONTOUR:
                A0 = add_contour(A0)
            save_AB(A0, B0, out_dir / f"{ip.stem}_{tag}_orig_{k}_{idx:06d}.png")
            idx += 1

        # --- augmented duplicates ---
        for k in range(aug_per_image):
            A_aug = A
            B_aug = B

            if USE_PAIRED_AFFINE:
                A_aug, B_aug = apply_paired_affine_safe(A_aug, B_aug)

            if USE_B_PHOTO_AUG:
                B_aug = apply_b_photo_aug(B_aug)

            # resize to training size
            A2 = resize_mask(A_aug)
            B2 = resize_img(B_aug)

            # ensure palette exact after all ops
            A2 = normalize_mask_to_palette(A2)

            if ADD_CONTOUR:
                A2 = add_contour(A2)

            save_AB(A2, B2, out_dir / f"{ip.stem}_{tag}_aug_{k}_{idx:06d}.png")
            idx += 1

    print(f"[{tag}] wrote {idx} samples -> {out_dir}")

def main():
    pairs = list_pairs(MASK_DIR, IMG_DIR)
    print("Pairs found:", len(pairs))
    random.shuffle(pairs)

    n_test = max(1, int(len(pairs) * TEST_RATIO))
    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]

    process_split(train_pairs, TRAIN_DIR, "tr", AUG_PER_IMAGE_TRAIN, KEEP_ORIG_PER_IMAGE_TRAIN)
    process_split(test_pairs,  TEST_DIR,  "te", AUG_PER_IMAGE_TEST,  KEEP_ORIG_PER_IMAGE_TEST)

    print("Done:", OUT_ROOT.resolve())
    print("Tip: train pix2pix with --preprocess none (already 512x512 samples).")

if __name__ == "__main__":
    main()