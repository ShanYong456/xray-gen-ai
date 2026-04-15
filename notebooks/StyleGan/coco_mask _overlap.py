from pathlib import Path
import json
import cv2
import numpy as np

# =========================
# Paths
# =========================
IMAGES_DIR = Path("data/raw/SHAMPOOBLADEWITHTRAY_TGT")
COCO_JSON = Path("data/raw/SHAMPOOBLADEWITHTRAY_TGT/result.json")

# Per-class binary masks
OUT_SHAMPOO_MASKS = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/shampoo_masks")
OUT_TRAY_MASKS = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/tray_masks")
OUT_BLADE_MASKS = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/blade_masks")

# Viz folders
OUT_VIZ = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/masks_viz")
OUT_COLOR = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/masks_color")
OUT_OVERLAY = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/masks_overlay")

MANIFEST_JSON = Path("data/interim/SHAMPOOBLADEWITHTRAY_TGT/masks_manifest_filtered.json")

for d in [
    OUT_SHAMPOO_MASKS,
    OUT_TRAY_MASKS,
    OUT_BLADE_MASKS,
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
# 0 = background
# 1 = shampoo only
# 2 = tray only
# 3 = blade only
# 4 = shampoo + tray overlap
# 5 = blade + tray overlap
# 6 = shampoo + blade overlap
# 7 = shampoo + blade + tray overlap
PALETTE_COMBINED = {
    0: (0, 0, 0),        # background -> black
    1: (0, 255, 0),      # shampoo only -> green
    2: (255, 0, 0),      # tray only -> blue
    3: (0, 0, 255),      # blade only -> red
    4: (0, 255, 255),    # shampoo + tray -> yellow
    5: (255, 0, 255),    # blade + tray -> magenta
    6: (255, 255, 0),    # shampoo + blade -> cyan
    7: (255, 255, 255),  # all three -> white
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


def build_combined_label(
    shampoo_mask: np.ndarray,
    tray_mask: np.ndarray,
    blade_mask: np.ndarray,
) -> np.ndarray:
    """
    Build a visualization-only label map:
      0 = background
      1 = shampoo only
      2 = tray only
      3 = blade only
      4 = shampoo + tray
      5 = blade + tray
      6 = shampoo + blade
      7 = shampoo + blade + tray
    """
    combined = np.zeros_like(shampoo_mask, dtype=np.uint8)

    s = shampoo_mask > 0
    t = tray_mask > 0
    b = blade_mask > 0

    combined[s & ~t & ~b] = 1
    combined[~s & t & ~b] = 2
    combined[~s & ~t & b] = 3
    combined[s & t & ~b] = 4
    combined[~s & t & b] = 5
    combined[s & ~t & b] = 6
    combined[s & t & b] = 7

    return combined


def colorize_combined_mask(combined_mask: np.ndarray, palette: dict) -> np.ndarray:
    h, w = combined_mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in np.unique(combined_mask):
        cls_id_int = int(cls_id)
        color[combined_mask == cls_id_int] = palette.get(cls_id_int, (255, 255, 255))

    return color


def augment_no_scale_multimask(image, shampoo_mask, tray_mask, blade_mask):
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

    blade_mask = cv2.warpAffine(
        blade_mask,
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
    blade_mask = (blade_mask > 0).astype(np.uint8)

    return image, shampoo_mask, tray_mask, blade_mask


if contains_rle_annotations():
    print("Detected RLE-style segmentations (segmentation is a dict).")
    print("This script currently handles polygon segmentations only.")
    print("If you need RLE support, install pycocotools and decode RLE masks.")

# =========================
# Main
# =========================
AUG_PER_IMAGE = 0

image_ids = list(images.keys())
print(f"Found {len(image_ids)} COCO images in {COCO_JSON}")
print(f"Category remap (COCO id -> train id): {cat_id_to_train}")
print(f"Train id to name: {train_id_to_name}")
print(f"Augmentations per image: {AUG_PER_IMAGE}")
print("Keeping only images with:")
print("  - tray + blade")
print("  - tray + shampoo")
print("Combined visualization now includes blade classes too.")

n_written = 0
n_aug_written = 0
n_missing = 0
n_failed = 0
n_empty = 0
n_filtered_out = 0
n_aug_filtered_out = 0

manifest = {
    "images_dir": str(IMAGES_DIR),
    "coco_json": str(COCO_JSON),
    "aug_per_image": AUG_PER_IMAGE,
    "filter_rule": "keep only (tray AND blade) OR (tray AND shampoo)",
    "combined_classes": {
        "0": "background",
        "1": "shampoo_only",
        "2": "tray_only",
        "3": "blade_only",
        "4": "shampoo_tray_overlap",
        "5": "blade_tray_overlap",
        "6": "shampoo_blade_overlap",
        "7": "shampoo_blade_tray_overlap",
    },
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

    shampoo_mask = np.zeros((H, W), dtype=np.uint8)
    tray_mask = np.zeros((H, W), dtype=np.uint8)
    blade_mask = np.zeros((H, W), dtype=np.uint8)

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
            elif "blade" in cls_name:
                cv2.fillPoly(blade_mask, [pts], 1)

    has_tray = np.count_nonzero(tray_mask) > 0
    has_shampoo = np.count_nonzero(shampoo_mask) > 0
    has_blade = np.count_nonzero(blade_mask) > 0

    keep_original = (has_tray and has_blade) or (has_tray and has_shampoo)
    if not keep_original:
        print(
            f"{img_path.name} | FILTERED OUT | "
            f"has_tray={has_tray} has_shampoo={has_shampoo} has_blade={has_blade}"
        )
        n_filtered_out += 1
        continue

    combined_mask = build_combined_label(shampoo_mask, tray_mask, blade_mask)
    color_mask = colorize_combined_mask(combined_mask, PALETTE_COMBINED)
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    base_name = f"{img_path.stem}.png"

    out_shampoo = OUT_SHAMPOO_MASKS / base_name
    out_tray = OUT_TRAY_MASKS / base_name
    out_blade = OUT_BLADE_MASKS / base_name
    out_viz = OUT_VIZ / f"{img_path.stem}_combined_viz.png"
    out_color = OUT_COLOR / base_name
    out_overlay = OUT_OVERLAY / base_name

    cv2.imwrite(str(out_shampoo), shampoo_mask)
    cv2.imwrite(str(out_tray), tray_mask)
    cv2.imwrite(str(out_blade), blade_mask)
    cv2.imwrite(str(out_viz), combined_mask * 36)  # 0..252 for 8 classes
    cv2.imwrite(str(out_color), color_mask)
    cv2.imwrite(str(out_overlay), overlay)

    overlap_shampoo_tray = int(np.count_nonzero((shampoo_mask > 0) & (tray_mask > 0)))
    overlap_blade_tray = int(np.count_nonzero((blade_mask > 0) & (tray_mask > 0)))
    overlap_shampoo_blade = int(np.count_nonzero((shampoo_mask > 0) & (blade_mask > 0)))
    overlap_all_three = int(np.count_nonzero((shampoo_mask > 0) & (tray_mask > 0) & (blade_mask > 0)))

    shampoo_pixels = int(np.count_nonzero(shampoo_mask))
    tray_pixels = int(np.count_nonzero(tray_mask))
    blade_pixels = int(np.count_nonzero(blade_mask))

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
        "blade_mask_path": str(out_blade),
        "combined_viz_path": str(out_viz),
        "color_path": str(out_color),
        "overlay_path": str(out_overlay),
        "has_shampoo": bool(has_shampoo),
        "has_tray": bool(has_tray),
        "has_blade": bool(has_blade),
        "shampoo_pixels": shampoo_pixels,
        "tray_pixels": tray_pixels,
        "blade_pixels": blade_pixels,
        "overlap_shampoo_tray": overlap_shampoo_tray,
        "overlap_blade_tray": overlap_blade_tray,
        "overlap_shampoo_blade": overlap_shampoo_blade,
        "overlap_all_three": overlap_all_three,
        "combined_unique_values": [int(v) for v in np.unique(combined_mask)],
    }

    print(
        f"{img_path.name} | ORIGINAL | "
        f"shampoo={shampoo_pixels} | "
        f"tray={tray_pixels} | "
        f"blade={blade_pixels} | "
        f"uniq_combined={np.unique(combined_mask)}"
    )

    n_written += 1

    for i in range(AUG_PER_IMAGE):
        aug_img, aug_shampoo, aug_tray, aug_blade = augment_no_scale_multimask(
            img.copy(),
            shampoo_mask.copy(),
            tray_mask.copy(),
            blade_mask.copy(),
        )

        aug_has_tray = np.count_nonzero(aug_tray) > 0
        aug_has_shampoo = np.count_nonzero(aug_shampoo) > 0
        aug_has_blade = np.count_nonzero(aug_blade) > 0

        keep_aug = (aug_has_tray and aug_has_blade) or (aug_has_tray and aug_has_shampoo)
        if not keep_aug:
            print(
                f"    -> {img_path.stem}_aug{i + 1}.png | AUG FILTERED OUT | "
                f"has_tray={aug_has_tray} has_shampoo={aug_has_shampoo} has_blade={aug_has_blade}"
            )
            n_aug_filtered_out += 1
            continue

        aug_combined = build_combined_label(aug_shampoo, aug_tray, aug_blade)
        aug_color = colorize_combined_mask(aug_combined, PALETTE_COMBINED)
        aug_overlay = cv2.addWeighted(aug_img, 0.7, aug_color, 0.3, 0)

        aug_name = f"{img_path.stem}_aug{i + 1}"
        aug_file = f"{aug_name}.png"

        aug_img_path = IMAGES_DIR / aug_file
        cv2.imwrite(str(aug_img_path), aug_img)

        aug_shampoo_path = OUT_SHAMPOO_MASKS / aug_file
        aug_tray_path = OUT_TRAY_MASKS / aug_file
        aug_blade_path = OUT_BLADE_MASKS / aug_file
        aug_viz_path = OUT_VIZ / f"{aug_name}_combined_viz.png"
        aug_color_path = OUT_COLOR / aug_file
        aug_overlay_path = OUT_OVERLAY / aug_file

        cv2.imwrite(str(aug_shampoo_path), aug_shampoo)
        cv2.imwrite(str(aug_tray_path), aug_tray)
        cv2.imwrite(str(aug_blade_path), aug_blade)
        cv2.imwrite(str(aug_viz_path), aug_combined * 36)
        cv2.imwrite(str(aug_color_path), aug_color)
        cv2.imwrite(str(aug_overlay_path), aug_overlay)

        aug_overlap_shampoo_tray = int(np.count_nonzero((aug_shampoo > 0) & (aug_tray > 0)))
        aug_overlap_blade_tray = int(np.count_nonzero((aug_blade > 0) & (aug_tray > 0)))
        aug_overlap_shampoo_blade = int(np.count_nonzero((aug_shampoo > 0) & (aug_blade > 0)))
        aug_overlap_all_three = int(np.count_nonzero((aug_shampoo > 0) & (aug_tray > 0) & (aug_blade > 0)))

        aug_shampoo_pixels = int(np.count_nonzero(aug_shampoo))
        aug_tray_pixels = int(np.count_nonzero(aug_tray))
        aug_blade_pixels = int(np.count_nonzero(aug_blade))

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
            "blade_mask_path": str(aug_blade_path),
            "combined_viz_path": str(aug_viz_path),
            "color_path": str(aug_color_path),
            "overlay_path": str(aug_overlay_path),
            "has_shampoo": bool(aug_has_shampoo),
            "has_tray": bool(aug_has_tray),
            "has_blade": bool(aug_has_blade),
            "shampoo_pixels": aug_shampoo_pixels,
            "tray_pixels": aug_tray_pixels,
            "blade_pixels": aug_blade_pixels,
            "overlap_shampoo_tray": aug_overlap_shampoo_tray,
            "overlap_blade_tray": aug_overlap_blade_tray,
            "overlap_shampoo_blade": aug_overlap_shampoo_blade,
            "overlap_all_three": aug_overlap_all_three,
            "combined_unique_values": [int(v) for v in np.unique(aug_combined)],
        }

        print(
            f"    -> {aug_file} | AUG | "
            f"shampoo={aug_shampoo_pixels} | "
            f"tray={aug_tray_pixels} | "
            f"blade={aug_blade_pixels} | "
            f"uniq_combined={np.unique(aug_combined)}"
        )

        n_aug_written += 1

MANIFEST_JSON.write_text(json.dumps(manifest, indent=2))
print(f"Manifest written: {MANIFEST_JSON}")

print("\n===== Summary =====")
print(f"Wrote original masks:        {n_written}")
print(f"Wrote augmented masks:       {n_aug_written}")
print(f"Filtered out originals:      {n_filtered_out}")
print(f"Filtered out augmentations:  {n_aug_filtered_out}")
print(f"Total outputs kept:          {n_written + n_aug_written}")
print(f"Missing imgs:                {n_missing}")
print(f"Read failed:                 {n_failed}")
print(f"Empty masks:                 {n_empty}")
print("Outputs:")
print(f" - shampoo masks: {OUT_SHAMPOO_MASKS}")
print(f" - tray masks:    {OUT_TRAY_MASKS}")
print(f" - blade masks:   {OUT_BLADE_MASKS}")
print(f" - combined viz:  {OUT_VIZ}")
print(f" - colored masks: {OUT_COLOR}")
print(f" - overlays:      {OUT_OVERLAY}")
print(f" - manifest:      {MANIFEST_JSON}")