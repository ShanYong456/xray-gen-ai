import cv2
import numpy as np
import random
from pathlib import Path

# =========================
# CONFIG
# =========================
SIZE = 1024
OUT_COUNT = 10

OBJECT_LIB = Path("data/raw/Non-Contraband/Cropped")
OUT_DIR = Path("datasets/Non-Contraband/gen_random_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tray masks folder (exported from Label Studio COCO -> PNGs)
TRAY_MASK_DIR = Path("data/interim/GAN/Empty_Tray_mask/Mask")  # put multiple tray masks here

# Behavior toggles
ALLOW_OVERLAP = False  # False = no overlaps between placed objects

# Placement controls
N_MIN, N_MAX = 1, 3
MAX_TRIES_PER_OBJECT = 200  # attempts before giving up on placing one object

# Cluster controls
CLUSTER_PROB = 0.35
CLUSTER_SPREAD = 220  # placement spread around cluster center (px)

# Transform controls
SCALE_MIN, SCALE_MAX = 0.3, 1.3
ROT_MIN, ROT_MAX = 0.0, 360.0

# Optional: keep tiny objects visible by restricting min scale for certain classes
CLASS_SCALE_OVERRIDES = {
    # "Nail": (0.8, 1.6),
}

# =========================
# UTILS
# =========================

def rotate_preserve(obj: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate without cropping by expanding canvas to fit full rotated image."""
    h, w = obj.shape[:2]
    if h == 0 or w == 0:
        return obj

    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    rotated = cv2.warpAffine(
        obj,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderValue=(0, 0, 0),
    )
    return rotated


def get_nonzero_mask(rgb: np.ndarray) -> np.ndarray:
    """Binary mask where object exists (any channel > 0)."""
    return (rgb.sum(axis=2) > 0)


def bbox_from_mask(mask: np.ndarray):
    """Return bbox (x1,y1,x2,y2) of True pixels; None if empty."""
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return x1, y1, x2, y2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================
# LOAD TRAY MASKS
# =========================

def load_tray_masks():
    """
    Loads all tray mask PNGs from TRAY_MASK_DIR.
    Each mask must be 1024x1024, white=tray(255), black=outside(0).
    Returns: list of (name, tray_mask_bool)
    """
    paths = sorted(list(TRAY_MASK_DIR.glob("*.png")))
    if not paths:
        raise RuntimeError(f"No tray mask PNGs found in: {TRAY_MASK_DIR}")

    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (SIZE, SIZE):
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        m = (m > 127)  # bool
        masks.append((p.name, m))

    if not masks:
        raise RuntimeError(f"Could not read any tray masks from: {TRAY_MASK_DIR}")

    print(f"Loaded tray masks: {len(masks)} from {TRAY_MASK_DIR}")
    return masks


# =========================
# LOAD OBJECT CUTOUTS
# =========================

def load_objects():
    """
    Loads cutouts from OBJECT_LIB/<class>/*.png
    Returns list of dicts: {"cls": class_name, "img": HxWx3 uint8}
    Supports RGBA: alpha used to zero background.
    """
    objs = []
    for cls_dir in OBJECT_LIB.iterdir():
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name

        for img_path in cls_dir.glob("*.png"):
            m = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            if m.ndim != 3:
                continue

            # Convert RGBA -> RGB with alpha mask
            if m.shape[2] == 4:
                rgb = m[:, :, :3].copy()
                alpha = m[:, :, 3] > 0
                rgb[~alpha] = 0
                m = rgb
            else:
                if m.shape[2] != 3:
                    continue

            # Tight trim
            nz = get_nonzero_mask(m)
            bb = bbox_from_mask(nz)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            m = m[y1:y2, x1:x2].copy()

            objs.append({"cls": cls_name, "img": m})

    print("Loaded objects:", len(objs))
    if len(objs) == 0:
        print(f"WARNING: No PNG cutouts found under {OBJECT_LIB}")
    return objs


# =========================
# TRANSFORM OBJECT
# =========================

def transform_object(obj_rgb: np.ndarray, cls_name: str) -> np.ndarray:
    if cls_name in CLASS_SCALE_OVERRIDES:
        smin, smax = CLASS_SCALE_OVERRIDES[cls_name]
    else:
        smin, smax = SCALE_MIN, SCALE_MAX

    scale = random.uniform(smin, smax)
    obj = cv2.resize(obj_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    angle = random.uniform(ROT_MIN, ROT_MAX)
    obj = rotate_preserve(obj, angle)

    # trim again after rotation
    nz = get_nonzero_mask(obj)
    bb = bbox_from_mask(nz)
    if bb is None:
        return obj_rgb
    x1, y1, x2, y2 = bb
    obj = obj[y1:y2, x1:x2].copy()

    return obj


# =========================
# PLACEMENT LOGIC (tray-aware)
# =========================

def can_place(canvas_occ: np.ndarray,
              tray_mask: np.ndarray,
              obj_mask: np.ndarray,
              x: int, y: int,
              allow_overlap: bool) -> bool:
    """
    tray_mask: (SIZE,SIZE) bool allowed tray region
    """
    h, w = obj_mask.shape[:2]

    # must be fully inside image bounds
    if x < 0 or y < 0 or (x + w) > SIZE or (y + h) > SIZE:
        return False

    occ_region = canvas_occ[y:y+h, x:x+w]
    tray_region = tray_mask[y:y+h, x:x+w]

    # All object pixels must lie inside tray
    if not np.all(tray_region[obj_mask]):
        return False

    # Overlap constraint
    if not allow_overlap and np.any(occ_region & obj_mask):
        return False

    return True


def place_object(canvas_rgb: np.ndarray, canvas_occ: np.ndarray, obj_rgb: np.ndarray,
                 x: int, y: int):
    """Paste obj onto canvas at (x,y) using its nonzero mask."""
    h, w = obj_rgb.shape[:2]
    obj_mask = get_nonzero_mask(obj_rgb)

    region = canvas_rgb[y:y+h, x:x+w]
    region[obj_mask] = obj_rgb[obj_mask]

    canvas_occ[y:y+h, x:x+w][obj_mask] = True


def sample_position(h: int, w: int, center=None):
    """
    Always returns a position that keeps object inside the 1024x1024 image.
    (Tray constraint is checked in can_place.)
    """
    x_min, x_max = 0, SIZE - w
    y_min, y_max = 0, SIZE - h

    if center is None:
        x = random.randint(x_min, x_max)
        y = random.randint(y_min, y_max)
    else:
        cx, cy = center
        x = int(random.gauss(cx - w // 2, CLUSTER_SPREAD / 2))
        y = int(random.gauss(cy - h // 2, CLUSTER_SPREAD / 2))
        x = clamp(x, x_min, x_max)
        y = clamp(y, y_min, y_max)

    return x, y


# =========================
# GENERATE ONE MASK (random tray)
# =========================

def generate_one(objects, tray_masks, idx: int):
    tray_name, tray_mask = random.choice(tray_masks)

    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    occ = np.zeros((SIZE, SIZE), dtype=bool)

    n = random.randint(N_MIN, N_MAX)

    center = None
    if random.random() < CLUSTER_PROB:
        center = (random.randint(250, 774), random.randint(250, 774))

    placed = 0
    for _ in range(n):
        item = random.choice(objects)
        cls_name = item["cls"]
        obj = transform_object(item["img"], cls_name)

        obj_mask = get_nonzero_mask(obj)
        h, w = obj.shape[:2]
        if h <= 1 or w <= 1:
            continue

        ok = False
        for _try in range(MAX_TRIES_PER_OBJECT):
            x, y = sample_position(h, w, center=center)
            if can_place(occ, tray_mask, obj_mask, x, y, allow_overlap=ALLOW_OVERLAP):
                place_object(canvas, occ, obj, x, y)
                ok = True
                placed += 1
                break

        if not ok:
            continue

    out_path = OUT_DIR / f"{idx:06d}.png"
    cv2.imwrite(str(out_path), canvas)

    # Optional: save a debug text file or print which tray was used
    # print(f"Saved {out_path.name} using tray={tray_name}, placed={placed}")
    return placed, tray_name


# =========================
# MAIN
# =========================

def main():
    objects = load_objects()
    if len(objects) == 0:
        return

    tray_masks = load_tray_masks()

    used = {}
    for i in range(OUT_COUNT):
        placed, tray_name = generate_one(objects, tray_masks, i)
        used[tray_name] = used.get(tray_name, 0) + 1
        if i % 50 == 0:
            print(f"generated {i} (placed {placed})")

    print("DONE")
    print("Tray usage counts:")
    for k in sorted(used.keys()):
        print(f"  {k}: {used[k]}")


if __name__ == "__main__":
    main()