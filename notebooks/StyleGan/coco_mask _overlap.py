from pathlib import Path
import json
import cv2
import numpy as np

# =========================
# Paths
# =========================
IMAGES_DIR = Path("data/raw/SHAMPOOWITHTRAY")
COCO_JSON = Path("data/raw/SHAMPOOWITHTRAY/result.json")

# Per-class binary masks
OUT_SHAMPOO_MASKS = Path("data/interim/SHAMPOOWITHTRAY/shampoo_masks")
OUT_TRAY_MASKS = Path("data/interim/SHAMPOOWITHTRAY/tray_masks")

# Viz folders
OUT_VIZ = Path("data/interim/SHAMPOOWITHTRAY/masks_viz")
OUT_COLOR = Path("data/interim/SHAMPOOWITHTRAY/masks_color")
OUT_OVERLAY = Path("data/interim/SHAMPOOWITHTRAY/masks_overlay")

MANIFEST_JSON = Path("data/interim/SHAMPOOWITHTRAY/masks_manifest.json")

for d in [
    OUT_SHAMPOO_MASKS,
    OUT_TRAY_MASKS,
    OUT_VIZ,
    OUT_COLOR,
    OUT_OVERLAY,
]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Load COCO
# =========================
coco = json.loads(COCO_JSON.read_text())

images = {im["id"]: im for im in coco.get("images", [])}
cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
cat_id_to_train = {c["id"]: i + 1 for i, c in enumerate(cats)}  # background=0
train_id_to_name = {i + 1: c["name"].lower() for i, c in enumerate(cats)}

ann_by_img = {}
for ann in coco.get("annotations", []):
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

# =========================
# Palette (BGR for OpenCV)
# =========================
# tray only   -> blue
# shampoo only-> green
# overlap     -> yellow
PALETTE_COMBINED = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # shampoo only
    2: (255, 0, 0),      # tray only
    3: (0, 255, 255),    # overlap
}

# =========================
# Helpers
# =========================
def poly_to_pts(seg, W, H):
    pts_list = []

    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        polys = seg
    else:
        polys = [seg]

    for poly in polys:
        if not poly or len(poly) < 6:
            continue

        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)

        if arr[:, 0].max() <= 1.5 and arr[:, 1].max() <= 1.5:
            arr[:, 0] *= W
            arr[:, 1] *= H

        arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)

        pts_list.append(arr.astype(np.int32))

    return pts_list


def contains_rle_annotations():
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict):
            return True
    return False


def make_binary_viz(mask: np.ndarray) -> np.ndarray:
    """0/1 mask -> 0/255 for viewing."""
    return (mask > 0).astype(np.uint8) * 255


def build_combined_label(shampoo_mask: np.ndarray, tray_mask: np.ndarray) -> np.ndarray:
    """
    Build a visualization-only label map:
      0 = background
      1 = shampoo only
      2 = tray only
      3 = overlap
    """
    combined = np.zeros_like(shampoo_mask, dtype=np.uint8)

    s = shampoo_mask > 0
    t = tray_mask > 0

    combined[s & ~t] = 1
    combined[t & ~s] = 2
    combined[s & t] = 3

    return combined


def colorize_combined_mask(combined_mask: np.ndarray, palette: dict) -> np.ndarray:
    h, w = combined_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in np.unique(combined_mask):
        cls_id_int = int(cls_id)
        color[combined_mask == cls_id_int] = palette.get(cls_id_int, (255, 255, 255))

    return color


def augment_no_scale_multimask(image, shampoo_mask, tray_mask):
    h, w = image.shape[:2]

    angle = np.random.uniform(-3, 3)
    center = (w // 2, h // 2)
    R = cv2.getRotationMatrix2D(center, angle, 1.0)

    image = cv2.warpAffine(
        image,
        R,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    shampoo_mask = cv2.warpAffine(
        shampoo_mask,
        R,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    tray_mask = cv2.warpAffine(
        tray_mask,
        R,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    alpha = np.random.uniform(0.95, 1.05)
    beta = np.random.uniform(-5, 5)
    image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

    if np.random.rand() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0.3)

    shampoo_mask = (shampoo_mask > 0).astype(np.uint8)
    tray_mask = (tray_mask > 0).astype(np.uint8)

    return image, shampoo_mask, tray_mask


if contains_rle_annotations():
    print("Detected RLE-style segmentations (segmentation is a dict).")
    print("This script currently handles polygon segmentations only.")
    print("If you need RLE support, install pycocotools and decode RLE masks.")

# =========================
# Main
# =========================
AUG_PER_IMAGE = 2

image_ids = list(images.keys())
print(f"Found {len(image_ids)} COCO images in {COCO_JSON}")
print(f"Category remap (COCO id -> train id): {cat_id_to_train}")
print(f"Train id to name: {train_id_to_name}")
print(f"Augmentations per image: {AUG_PER_IMAGE}")

n_written = 0
n_aug_written = 0
n_missing = 0
n_failed = 0
n_empty = 0

manifest = {
    "images_dir": str(IMAGES_DIR),
    "coco_json": str(COCO_JSON),
    "aug_per_image": AUG_PER_IMAGE,
    "category_remap": {str(k): int(v) for k, v in cat_id_to_train.items()},
    "train_id_to_name": {str(k): v for k, v in train_id_to_name.items()},
    "entries": {}
}

for image_id in image_ids:
    im = images[image_id]
    file_name = im.get("file_name", "")
    img_path = IMAGES_DIR / Path(file_name).name

    if not img_path.exists():
        print(f"Missing image: COCO file_name='{file_name}' -> tried '{img_path}'")
        n_missing += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read image: {img_path}")
        n_failed += 1
        continue

    H, W = img.shape[:2]

    # Separate masks so overlap is preserved
    shampoo_mask = np.zeros((H, W), dtype=np.uint8)
    tray_mask = np.zeros((H, W), dtype=np.uint8)

    anns = ann_by_img.get(image_id, [])
    if len(anns) == 0:
        n_empty += 1

    for ann in anns:
        seg = ann.get("segmentation", None)
        if not seg:
            continue

        if isinstance(seg, dict):
            continue

        cat = ann.get("category_id", None)
        if cat is None:
            continue

        train_id = int(cat_id_to_train.get(cat, 1))
        cls_name = train_id_to_name.get(train_id, "")
        pts_list = poly_to_pts(seg, W, H)

        for pts in pts_list:
            if "shampoo" in cls_name:
                cv2.fillPoly(shampoo_mask, [pts], 1)
            elif "tray" in cls_name:
                cv2.fillPoly(tray_mask, [pts], 1)

    combined_mask = build_combined_label(shampoo_mask, tray_mask)
    color_mask = colorize_combined_mask(combined_mask, PALETTE_COMBINED)
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    base_name = f"{img_path.stem}.png"

    out_shampoo = OUT_SHAMPOO_MASKS / base_name
    out_tray = OUT_TRAY_MASKS / base_name
    out_viz = OUT_VIZ / f"{img_path.stem}_combined_viz.png"
    out_color = OUT_COLOR / base_name
    out_overlay = OUT_OVERLAY / base_name

    cv2.imwrite(str(out_shampoo), shampoo_mask)
    cv2.imwrite(str(out_tray), tray_mask)
    cv2.imwrite(str(out_viz), combined_mask * 85)  # 0,85,170,255
    cv2.imwrite(str(out_color), color_mask)
    cv2.imwrite(str(out_overlay), overlay)

    overlap_pixels = int(np.count_nonzero((shampoo_mask > 0) & (tray_mask > 0)))

    manifest["entries"][base_name] = {
        "image_id": int(image_id),
        "source_image": img_path.name,
        "source_stem": img_path.stem,
        "is_aug": False,
        "aug_index": 0,
        "height": int(H),
        "width": int(W),
        "shampoo_mask_path": str(out_shampoo),
        "tray_mask_path": str(out_tray),
        "combined_viz_path": str(out_viz),
        "color_path": str(out_color),
        "overlay_path": str(out_overlay),
        "shampoo_pixels": int(np.count_nonzero(shampoo_mask)),
        "tray_pixels": int(np.count_nonzero(tray_mask)),
        "overlap_pixels": overlap_pixels,
    }

    print(
        f"{img_path.name} | ORIGINAL | "
        f"shampoo={int(np.count_nonzero(shampoo_mask))} | "
        f"tray={int(np.count_nonzero(tray_mask))} | "
        f"overlap={overlap_pixels} | "
        f"uniq_combined={np.unique(combined_mask)}"
    )

    n_written += 1

    for i in range(AUG_PER_IMAGE):
        aug_img, aug_shampoo, aug_tray = augment_no_scale_multimask(
            img.copy(), shampoo_mask.copy(), tray_mask.copy()
        )

        aug_combined = build_combined_label(aug_shampoo, aug_tray)
        aug_color = colorize_combined_mask(aug_combined, PALETTE_COMBINED)
        aug_overlay = cv2.addWeighted(aug_img, 0.7, aug_color, 0.3, 0)

        aug_name = f"{img_path.stem}_aug{i + 1}"
        aug_file = f"{aug_name}.png"

        aug_img_path = IMAGES_DIR / aug_file
        cv2.imwrite(str(aug_img_path), aug_img)

        aug_shampoo_path = OUT_SHAMPOO_MASKS / aug_file
        aug_tray_path = OUT_TRAY_MASKS / aug_file
        aug_viz_path = OUT_VIZ / f"{aug_name}_combined_viz.png"
        aug_color_path = OUT_COLOR / aug_file
        aug_overlay_path = OUT_OVERLAY / aug_file

        cv2.imwrite(str(aug_shampoo_path), aug_shampoo)
        cv2.imwrite(str(aug_tray_path), aug_tray)
        cv2.imwrite(str(aug_viz_path), aug_combined * 85)
        cv2.imwrite(str(aug_color_path), aug_color)
        cv2.imwrite(str(aug_overlay_path), aug_overlay)

        aug_overlap_pixels = int(np.count_nonzero((aug_shampoo > 0) & (aug_tray > 0)))

        manifest["entries"][aug_file] = {
            "image_id": int(image_id),
            "source_image": img_path.name,
            "source_stem": img_path.stem,
            "is_aug": True,
            "aug_index": int(i + 1),
            "parent_mask": base_name,
            "augmented_image_path": str(aug_img_path),
            "height": int(H),
            "width": int(W),
            "shampoo_mask_path": str(aug_shampoo_path),
            "tray_mask_path": str(aug_tray_path),
            "combined_viz_path": str(aug_viz_path),
            "color_path": str(aug_color_path),
            "overlay_path": str(aug_overlay_path),
            "shampoo_pixels": int(np.count_nonzero(aug_shampoo)),
            "tray_pixels": int(np.count_nonzero(aug_tray)),
            "overlap_pixels": aug_overlap_pixels,
        }

        print(
            f"    -> {aug_file} | AUG | "
            f"shampoo={int(np.count_nonzero(aug_shampoo))} | "
            f"tray={int(np.count_nonzero(aug_tray))} | "
            f"overlap={aug_overlap_pixels} | "
            f"uniq_combined={np.unique(aug_combined)}"
        )

        n_aug_written += 1

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
print(f"Manifest written: {MANIFEST_JSON}")

print("\n===== Summary =====")
print(f"Wrote original masks:   {n_written}")
print(f"Wrote augmented masks:  {n_aug_written}")
print(f"Total outputs:          {n_written + n_aug_written}")
print(f"Missing imgs:           {n_missing}")
print(f"Read failed:            {n_failed}")
print(f"Empty masks:            {n_empty}")
print("Outputs:")
print(f" - shampoo masks: {OUT_SHAMPOO_MASKS}")
print(f" - tray masks:    {OUT_TRAY_MASKS}")
print(f" - combined viz:  {OUT_VIZ}")
print(f" - colored masks: {OUT_COLOR}")
print(f" - overlays:      {OUT_OVERLAY}")
print(f" - manifest:      {MANIFEST_JSON}")