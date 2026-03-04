from pathlib import Path
import cv2
import numpy as np
import random

# ===== INPUTS =====
IMG_DIR  = Path("data/raw/Shampoo&Blade")
MASK_DIR = Path("data/interim/Shampoo&Blade/masks")

# ===== OUTPUT =====
OUT_ROOT  = Path("datasets/Shampoo&Blade")
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR  = OUT_ROOT / "test"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ===== SETTINGS =====
SIZE = 1024
TEST_RATIO = 0.2
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

#FOR CONTRABAND METAL:
"""
# ===== FIXED BASE PALETTE (BGR) =====
PALETTE_BGR = {
    0:  (0, 0, 0),

    1:  (0, 0, 255),
    2:  (0, 255, 0),
    3:  (255, 0, 0),

    4:  (0, 255, 255),
    5:  (255, 255, 0),
    6:  (255, 0, 255),

    7:  (0, 128, 255),
    8:  (255, 128, 0),
    9:  (128, 0, 255),

    10: (0, 255, 128),
    11: (255, 0, 128),
    12: (128, 255, 0),

    13: (255, 128, 128),
    14: (128, 255, 255),
    15: (255, 255, 128),
    16: (112, 55, 89),

}
"""
"""
#FOR NON-CONTRABAND:
PALETTE_BGR ={
    0: (0, 0, 0),         # background
    
    1: (255, 0, 0),       # blue
    2: (0, 255, 0),       # green
    3: (0, 0, 255),       # red
    4: (255, 255, 0),     # cyan
    5: (0, 255, 255),     # yellow
    6: (255, 0, 255),     # magenta
}
"""

#FOR SHAMPOO&BLADE:
PALETTE_BGR ={
    0: (0, 0, 0),         # background
    1: (0, 255, 0),       # green
    2: (255, 0, 0),       # blue
   
}


# store dynamically generated colors (for unknown ids)
DYNAMIC_COLORS = {}

def deterministic_color(idx):
    """Generate stable unique color for unknown label ids."""
    rng = np.random.RandomState(idx * 999)
    color = rng.randint(40, 255, size=3)
    return tuple(int(x) for x in color)

def resize_mask(mask, size):
    return cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

def resize_img(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

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
            # generate stable color once
            if uid not in DYNAMIC_COLORS:
                DYNAMIC_COLORS[uid] = deterministic_color(uid)
                print(f"[WARN] Unknown label id {uid} -> assigned color {DYNAMIC_COLORS[uid]}")
            out[ids == uid] = DYNAMIC_COLORS[uid]

    return out

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

    img_r  = resize_img(img, SIZE)
    mask_r = resize_mask(mask, SIZE)

    A3 = id_to_color(mask_r)
    AB = np.concatenate([A3, img_r], axis=1)

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
ok_test  = write_split(test_pairs,  TEST_DIR,  "te")

print("\nDone.")
print("Train wrote:", ok_train, "->", TRAIN_DIR.resolve())
print("Test  wrote:", ok_test,  "->", TEST_DIR.resolve())
print(f"\nPix2Pix tip: --preprocess none --no_flip --load_size {SIZE} --crop_size {SIZE}")