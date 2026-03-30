from pathlib import Path
import cv2
import numpy as np
import random

# ===== INPUTS =====
IMG_DIR  = Path("data/raw/Empty")
MASK_DIR = Path("data/interim/Empty/masks")

# ===== OUTPUT =====
OUT_ROOT  = Path("datasets/Empty")
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR  = OUT_ROOT / "test"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ===== SETTINGS =====
TEST_RATIO = 0.2
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# Fixed canvas size: MUST be >= all source image sizes
#Shampoo
"""
CANVAS_W = 1024
CANVAS_H = 1536
"""

#Tray
CANVAS_W = 1584
CANVAS_H = 1152

# White / light-gray padding for X-ray style background
PAD_VALUE = 255  # use 235 if you want slightly gray instead of pure white

# FOR SHAMPOO:
"""
PALETTE_BGR = {
    0: (0, 0, 0),      # background
    1: (0, 255, 0),    # green
}
"""
# FOR TRAY
PALETTE_BGR = {
    0: (0, 0, 0),      # background
    1: (255, 0, 0),    # green
}

# store dynamically generated colors (for unknown ids)
DYNAMIC_COLORS = {}


def deterministic_color(idx):
    """Generate stable unique color for unknown label ids."""
    rng = np.random.RandomState(idx * 999)
    color = rng.randint(40, 255, size=3)
    return tuple(int(x) for x in color)


def ensure_single_channel(mask):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def id_to_color(mask_ids):
    """
    Convert label-id mask -> color mask
    Unknown ids automatically assigned unique stable colors
    """
    ids = mask_ids.astype(np.int32)
    h, w = ids.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    unique_ids = np.unique(ids)
    for uid in unique_ids:
        if uid in PALETTE_BGR:
            out[ids == uid] = PALETTE_BGR[uid]
        else:
            if uid not in DYNAMIC_COLORS:
                DYNAMIC_COLORS[uid] = deterministic_color(uid)
                print(f"[WARN] Unknown label id {uid} -> assigned color {DYNAMIC_COLORS[uid]}")
            out[ids == uid] = DYNAMIC_COLORS[uid]

    return out


def pad_image_to_canvas(img, canvas_w, canvas_h, pad_value=255):
    """
    Keep original image unchanged and pad outward to target canvas.
    No resizing, no warping.
    """
    h, w = img.shape[:2]

    if w > canvas_w or h > canvas_h:
        raise ValueError(
            f"Image size {(w, h)} exceeds canvas {(canvas_w, canvas_h)}. "
            f"Increase CANVAS_W/CANVAS_H."
        )

    if img.ndim == 2:
        canvas = np.full((canvas_h, canvas_w), pad_value, dtype=img.dtype)
    else:
        if isinstance(pad_value, int):
            fill = [pad_value] * img.shape[2]
        else:
            fill = pad_value
        canvas = np.full((canvas_h, canvas_w, img.shape[2]), fill, dtype=img.dtype)

    x0 = (canvas_w - w) // 2
    y0 = (canvas_h - h) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def make_AB(mask_path: Path, img_path: Path, out_path: Path) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print("WARN: failed reading image:", img_path)
        return False

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    mask = ensure_single_channel(mask)
    if mask is None:
        print("WARN: failed reading mask:", mask_path)
        return False

    # Ensure mask and image spatial sizes match BEFORE padding
    if mask.shape[:2] != img.shape[:2]:
        print(f"WARN: size mismatch img={img.shape[:2]} mask={mask.shape[:2]} for {img_path.name}")
        return False

    # Convert label ids to RGB mask at ORIGINAL size
    A3 = id_to_color(mask)

    # Pad both A and B to the SAME canvas, no resize
    A3_pad  = pad_image_to_canvas(A3, CANVAS_W, CANVAS_H, pad_value=0)
    img_pad = pad_image_to_canvas(img, CANVAS_W, CANVAS_H, pad_value=PAD_VALUE)

    # Concatenate A|B
    AB = np.concatenate([A3_pad, img_pad], axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(out_path), AB))


# ===== collect pairs =====
pairs = []
for mp in sorted(MASK_DIR.glob("*.png")):
    ip = IMG_DIR / mp.name
    if ip.exists():
        pairs.append((mp, ip))

print("Pairs found:", len(pairs))
random.shuffle(pairs)

n_test = max(1, int(len(pairs) * TEST_RATIO))
test_pairs = pairs[:n_test]
train_pairs = pairs[n_test:]


def write_split(split_pairs, out_dir, prefix):
    ok = 0
    for i, (mp, ip) in enumerate(split_pairs):
        out_name = f"{ip.stem}_{prefix}_{i:06d}.png"
        ok += int(make_AB(mp, ip, out_dir / out_name))
    return ok


ok_train = write_split(train_pairs, TRAIN_DIR, "tr")
ok_test  = write_split(test_pairs, TEST_DIR,  "te")

print("\nDone.")
print("Train wrote:", ok_train, "->", TRAIN_DIR.resolve())
print("Test  wrote:", ok_test,  "->", TEST_DIR.resolve())
print(f"\nPix2Pix tip: --preprocess none --no_flip --load_size 0 --crop_size 0")
print(f"Canvas used: {CANVAS_W}x{CANVAS_H}")