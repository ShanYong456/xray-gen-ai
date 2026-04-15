from pathlib import Path
import cv2
import numpy as np
import random
import json

# ===== INPUTS =====
IMG_DIR = Path("data/raw/SHAMPOOBLADEWITHTRAY_TGT")

# Separate binary mask folders
SHAMPOO_MASK_DIR = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/shampoo_masks")
TRAY_MASK_DIR = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/tray_masks")
BLADE_MASK_DIR = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/blade_masks")

# ===== OUTPUT =====
OUT_ROOT = Path("datasets/SHAMPOOBLADEWITHTRAY_TGT")
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR = OUT_ROOT / "test"

# Save matched padded masks with EXACT SAME final filename as AB
MASK_ROOT = OUT_ROOT / "matched_masks"
TRAIN_SHAMPOO_MATCHED_DIR = MASK_ROOT / "train" / "shampoo"
TRAIN_TRAY_MATCHED_DIR = MASK_ROOT / "train" / "tray"
TRAIN_BLADE_MATCHED_DIR = MASK_ROOT / "train" / "blade"

TEST_SHAMPOO_MATCHED_DIR = MASK_ROOT / "test" / "shampoo"
TEST_TRAY_MATCHED_DIR = MASK_ROOT / "test" / "tray"
TEST_BLADE_MATCHED_DIR = MASK_ROOT / "test" / "blade"

# Manifest
MANIFEST_PATH = OUT_ROOT / "build_manifest.json"

for d in [
    TRAIN_DIR, TEST_DIR,
    TRAIN_SHAMPOO_MATCHED_DIR, TRAIN_TRAY_MATCHED_DIR, TRAIN_BLADE_MATCHED_DIR,
    TEST_SHAMPOO_MATCHED_DIR, TEST_TRAY_MATCHED_DIR, TEST_BLADE_MATCHED_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ===== SETTINGS =====
TEST_RATIO = 0.2
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# Fixed canvas size: MUST be >= all source image sizes
CANVAS_W = 1584
CANVAS_H = 1152

# White / light-gray padding for X-ray style background
PAD_VALUE = 255

# A-side encoding for pix2pix image input:
#   B = tray
#   G = shampoo
#   R = blade
#
# Mixed colors appear automatically from channel combination:
#   tray only            -> blue      [255,   0,   0]
#   shampoo only         -> green     [  0, 255,   0]
#   blade only           -> red       [  0,   0, 255]
#   tray + shampoo       -> cyan      [255, 255,   0]
#   tray + blade         -> magenta   [255,   0, 255]
#   shampoo + blade      -> yellow    [  0, 255, 255]
#   all three            -> white     [255, 255, 255]
def build_A_image(shampoo_mask, tray_mask, blade_mask):
    shampoo = (shampoo_mask > 0)
    tray = (tray_mask > 0)
    blade = (blade_mask > 0)

    A = np.zeros((shampoo.shape[0], shampoo.shape[1], 3), dtype=np.uint8)

    # B = tray
    A[..., 0][tray] = 255

    # G = shampoo
    A[..., 1][shampoo] = 255

    # R = blade
    A[..., 2][blade] = 255

    return A


def ensure_single_channel(mask):
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def binarize_mask(mask):
    mask = ensure_single_channel(mask)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def pad_image_to_canvas(img, canvas_w, canvas_h, pad_value=255):
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


def make_ab_and_save_masks(
    shampoo_mask_path: Path,
    tray_mask_path: Path,
    blade_mask_path: Path,
    img_path: Path,
    ab_out_path: Path,
    shampoo_out_path: Path,
    tray_out_path: Path,
    blade_out_path: Path,
) -> bool:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print("WARN: failed reading image:", img_path)
        return False

    shampoo_mask = cv2.imread(str(shampoo_mask_path), cv2.IMREAD_UNCHANGED)
    tray_mask = cv2.imread(str(tray_mask_path), cv2.IMREAD_UNCHANGED)
    blade_mask = cv2.imread(str(blade_mask_path), cv2.IMREAD_UNCHANGED)

    shampoo_mask = binarize_mask(shampoo_mask)
    tray_mask = binarize_mask(tray_mask)
    blade_mask = binarize_mask(blade_mask)

    if shampoo_mask is None:
        print("WARN: failed reading shampoo mask:", shampoo_mask_path)
        return False
    if tray_mask is None:
        print("WARN: failed reading tray mask:", tray_mask_path)
        return False
    if blade_mask is None:
        print("WARN: failed reading blade mask:", blade_mask_path)
        return False

    if shampoo_mask.shape[:2] != img.shape[:2]:
        print(f"WARN: shampoo size mismatch img={img.shape[:2]} mask={shampoo_mask.shape[:2]} for {img_path.name}")
        return False

    if tray_mask.shape[:2] != img.shape[:2]:
        print(f"WARN: tray size mismatch img={img.shape[:2]} mask={tray_mask.shape[:2]} for {img_path.name}")
        return False

    if blade_mask.shape[:2] != img.shape[:2]:
        print(f"WARN: blade size mismatch img={img.shape[:2]} mask={blade_mask.shape[:2]} for {img_path.name}")
        return False

    # Build A-side training image
    A3 = build_A_image(shampoo_mask, tray_mask, blade_mask)

    # Pad to same canvas
    A3_pad = pad_image_to_canvas(A3, CANVAS_W, CANVAS_H, pad_value=0)
    img_pad = pad_image_to_canvas(img, CANVAS_W, CANVAS_H, pad_value=PAD_VALUE)

    shampoo_pad = pad_image_to_canvas((shampoo_mask * 255).astype(np.uint8), CANVAS_W, CANVAS_H, pad_value=0)
    tray_pad = pad_image_to_canvas((tray_mask * 255).astype(np.uint8), CANVAS_W, CANVAS_H, pad_value=0)
    blade_pad = pad_image_to_canvas((blade_mask * 255).astype(np.uint8), CANVAS_W, CANVAS_H, pad_value=0)

    # Concatenate A|B
    AB = np.concatenate([A3_pad, img_pad], axis=1)

    ab_out_path.parent.mkdir(parents=True, exist_ok=True)
    shampoo_out_path.parent.mkdir(parents=True, exist_ok=True)
    tray_out_path.parent.mkdir(parents=True, exist_ok=True)
    blade_out_path.parent.mkdir(parents=True, exist_ok=True)

    ok_ab = bool(cv2.imwrite(str(ab_out_path), AB))
    ok_shampoo = bool(cv2.imwrite(str(shampoo_out_path), shampoo_pad))
    ok_tray = bool(cv2.imwrite(str(tray_out_path), tray_pad))
    ok_blade = bool(cv2.imwrite(str(blade_out_path), blade_pad))

    if not ok_ab:
        print("WARN: failed writing AB:", ab_out_path)
    if not ok_shampoo:
        print("WARN: failed writing shampoo mask:", shampoo_out_path)
    if not ok_tray:
        print("WARN: failed writing tray mask:", tray_out_path)
    if not ok_blade:
        print("WARN: failed writing blade mask:", blade_out_path)

    return ok_ab and ok_shampoo and ok_tray and ok_blade


def build_file_map(folder: Path):
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
    by_name = {}
    by_stem = {}

    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            by_name[p.name] = p
            by_stem[p.stem] = p
        for p in folder.glob(f"*{ext.upper()}"):
            by_name[p.name] = p
            by_stem[p.stem] = p

    return by_name, by_stem


def collect_quadruplets():
    """
    Match image, shampoo mask, tray mask, blade mask by:
    1. exact filename
    2. stem
    Only keep samples where all 4 exist.
    """
    img_by_name, img_by_stem = build_file_map(IMG_DIR)
    shampoo_by_name, shampoo_by_stem = build_file_map(SHAMPOO_MASK_DIR)
    tray_by_name, tray_by_stem = build_file_map(TRAY_MASK_DIR)
    blade_by_name, blade_by_stem = build_file_map(BLADE_MASK_DIR)

    candidate_stems = (
        set(img_by_stem.keys())
        | set(shampoo_by_stem.keys())
        | set(tray_by_stem.keys())
        | set(blade_by_stem.keys())
    )

    quadruplets = []
    missing = []

    for stem in sorted(candidate_stems):
        ip = img_by_stem.get(stem, None)

        sp = shampoo_by_name.get(ip.name, None) if ip is not None else None
        tp = tray_by_name.get(ip.name, None) if ip is not None else None
        bp = blade_by_name.get(ip.name, None) if ip is not None else None

        if sp is None:
            sp = shampoo_by_stem.get(stem, None)
        if tp is None:
            tp = tray_by_stem.get(stem, None)
        if bp is None:
            bp = blade_by_stem.get(stem, None)

        if ip is not None and sp is not None and tp is not None and bp is not None:
            quadruplets.append((ip, sp, tp, bp))
        else:
            missing.append({
                "stem": stem,
                "has_image": ip is not None,
                "has_shampoo_mask": sp is not None,
                "has_tray_mask": tp is not None,
                "has_blade_mask": bp is not None,
            })

    return quadruplets, missing


quadruplets, missing_items = collect_quadruplets()

print("Quadruplets found:", len(quadruplets))
if missing_items:
    print("Incomplete quadruplets:", len(missing_items))
    for x in missing_items[:20]:
        print("  MISSING:", x)

random.shuffle(quadruplets)

if len(quadruplets) == 0:
    raise RuntimeError("No valid image+shampoo_mask+tray_mask+blade_mask quadruplets found.")

n_test = max(1, int(len(quadruplets) * TEST_RATIO))
test_quadruplets = quadruplets[:n_test]
train_quadruplets = quadruplets[n_test:]

manifest = {
    "seed": SEED,
    "test_ratio": TEST_RATIO,
    "canvas_w": CANVAS_W,
    "canvas_h": CANVAS_H,
    "pad_value": PAD_VALUE,
    "img_dir": str(IMG_DIR),
    "shampoo_mask_dir": str(SHAMPOO_MASK_DIR),
    "tray_mask_dir": str(TRAY_MASK_DIR),
    "blade_mask_dir": str(BLADE_MASK_DIR),
    "num_quadruplets": len(quadruplets),
    "num_train": len(train_quadruplets),
    "num_test": len(test_quadruplets),
    "entries": [],
    "missing_items": missing_items,
}


def write_split(split_quadruplets, out_dir, shampoo_dir, tray_dir, blade_dir, prefix):
    ok = 0
    split_name = "train" if prefix == "tr" else "test"

    for i, (ip, sp, tp, bp) in enumerate(split_quadruplets):
        out_name = f"{ip.stem}_{prefix}_{i:06d}.png"

        ab_out_path = out_dir / out_name
        shampoo_out_path = shampoo_dir / out_name
        tray_out_path = tray_dir / out_name
        blade_out_path = blade_dir / out_name

        success = make_ab_and_save_masks(
            shampoo_mask_path=sp,
            tray_mask_path=tp,
            blade_mask_path=bp,
            img_path=ip,
            ab_out_path=ab_out_path,
            shampoo_out_path=shampoo_out_path,
            tray_out_path=tray_out_path,
            blade_out_path=blade_out_path,
        )
        ok += int(success)

        manifest["entries"].append({
            "split": split_name,
            "final_name": out_name,
            "source_image": ip.name,
            "source_shampoo_mask": sp.name,
            "source_tray_mask": tp.name,
            "source_blade_mask": bp.name,
            "ab_path": str(ab_out_path),
            "shampoo_mask_path": str(shampoo_out_path),
            "tray_mask_path": str(tray_out_path),
            "blade_mask_path": str(blade_out_path),
            "success": bool(success),
        })

    return ok


ok_train = write_split(
    train_quadruplets,
    TRAIN_DIR,
    TRAIN_SHAMPOO_MATCHED_DIR,
    TRAIN_TRAY_MATCHED_DIR,
    TRAIN_BLADE_MATCHED_DIR,
    "tr",
)

ok_test = write_split(
    test_quadruplets,
    TEST_DIR,
    TEST_SHAMPOO_MATCHED_DIR,
    TEST_TRAY_MATCHED_DIR,
    TEST_BLADE_MATCHED_DIR,
    "te",
)

MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

print("\nDone.")
print("Train wrote:", ok_train, "->", TRAIN_DIR.resolve())
print("Test  wrote:", ok_test, "->", TEST_DIR.resolve())
print("Train shampoo masks ->", TRAIN_SHAMPOO_MATCHED_DIR.resolve())
print("Train tray masks    ->", TRAIN_TRAY_MATCHED_DIR.resolve())
print("Train blade masks   ->", TRAIN_BLADE_MATCHED_DIR.resolve())
print("Test shampoo masks  ->", TEST_SHAMPOO_MATCHED_DIR.resolve())
print("Test tray masks     ->", TEST_TRAY_MATCHED_DIR.resolve())
print("Test blade masks    ->", TEST_BLADE_MATCHED_DIR.resolve())
print("Manifest ->", MANIFEST_PATH.resolve())
print(f"\nPix2Pix tip: --preprocess none --no_flip --load_size 0 --crop_size 0")
print(f"Canvas used: {CANVAS_W}x{CANVAS_H}")