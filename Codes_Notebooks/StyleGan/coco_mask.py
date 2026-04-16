from pathlib import Path
import json
import cv2
import numpy as np

# =========================
# Paths
# =========================
IMAGES_DIR = Path("data/raw/SHAMPOOWITHTRAY")                    # your images
COCO_JSON = Path("data/raw/SHAMPOOWITHTRAY/result.json")        # COCO export

OUT_MASKS = Path("data/interim/SHAMPOOWITHTRAY/masks")          # raw label-id masks (0..K)
OUT_VIZ = Path("data/interim/SHAMPOOWITHTRAY/masks_viz")        # grayscale view (0..255)
OUT_COLOR = Path("data/interim/SHAMPOOWITHTRAY/masks_color")    # colored mask (per-class color)
OUT_OVERLAY = Path("data/interim/SHAMPOOWITHTRAY/masks_overlay")# colored mask overlaid on image

# NEW: manifest to preserve mask <-> source relationship
MANIFEST_JSON = Path("data/interim/SHAMPOOWITHTRAY/masks_manifest.json")

for d in [OUT_MASKS, OUT_VIZ, OUT_COLOR, OUT_OVERLAY]:
    d.mkdir(parents=True, exist_ok=True)

# =========================
# Load COCO
# =========================
coco = json.loads(COCO_JSON.read_text())

# Map COCO image_id -> info
images = {im["id"]: im for im in coco.get("images", [])}

# Remap category ids to contiguous 1..K
cats = sorted(coco.get("categories", []), key=lambda c: c["id"])
cat_id_to_train = {c["id"]: i + 1 for i, c in enumerate(cats)}  # background=0

# Group annotations by image_id
ann_by_img = {}
for ann in coco.get("annotations", []):
    ann_by_img.setdefault(ann["image_id"], []).append(ann)

# =========================
# Palette (BGR for OpenCV)
# =========================
# background = 0 (black)
# Classes = 1..K

# FOR SHAMPOO & TRAY / BLADE:
PALETTE = {
    0: (0, 0, 0),       # background
    1: (0, 255, 0),     # green
    2: (255, 0, 0),     # blue
}

# =========================
# Helpers
# =========================
def poly_to_pts(seg, W, H):
    """
    seg can be:
      - list-of-lists (multiple polygons), each polygon: [x1,y1,x2,y2,...]
      - list (single polygon): [x1,y1,...]
    Handles normalized coords (0..1) by auto-scaling to pixels.
    """
    pts_list = []

    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        polys = seg
    else:
        polys = [seg]

    for poly in polys:
        if not poly or len(poly) < 6:
            continue

        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)

        # If coordinates look normalized (0..1), scale to pixel coords
        if arr[:, 0].max() <= 1.5 and arr[:, 1].max() <= 1.5:
            arr[:, 0] *= W
            arr[:, 1] *= H

        # clip to bounds
        arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)

        pts_list.append(arr.astype(np.int32))

    return pts_list


def make_viz(mask: np.ndarray) -> np.ndarray:
    """Scale label mask (0..K) to 0..255 for easy viewing."""
    mmax = int(mask.max())
    if mmax <= 0:
        return mask.copy()
    viz = (mask.astype(np.float32) / mmax * 255.0).clip(0, 255).astype(np.uint8)
    return viz


def colorize_mask(mask: np.ndarray, palette: dict) -> np.ndarray:
    """Convert label-id mask to a color image using a palette."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in np.unique(mask):
        cls_id_int = int(cls_id)
        bgr = palette.get(cls_id_int, (255, 255, 255))  # unknown -> white
        color[mask == cls_id_int] = bgr

    return color


def contains_rle_annotations():
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation", None)
        if isinstance(seg, dict):
            return True
    return False


def augment_no_scale(image, mask):
    h, w = image.shape[:2]

    # Rotation only (no scaling, no shifting)
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
    mask = cv2.warpAffine(
        mask,
        R,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Mild brightness / contrast
    alpha = np.random.uniform(0.95, 1.05)
    beta = np.random.uniform(-5, 5)
    image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

    # Optional slight blur
    if np.random.rand() < 0.2:
        image = cv2.GaussianBlur(image, (3, 3), 0.3)

    return image, mask


if contains_rle_annotations():
    print("Detected RLE-style segmentations (segmentation is a dict).")
    print("This script currently handles polygon segmentations only.")
    print("If you need RLE support, install pycocotools and decode RLE masks.")

# =========================
# Main
# =========================
AUG_PER_IMAGE = 2   # number of augmented samples per original image

image_ids = list(images.keys())
print(f"Found {len(image_ids)} COCO images in {COCO_JSON}")
print(f"Category remap (COCO id -> train id): {cat_id_to_train}")
print(f"Augmentations per image: {AUG_PER_IMAGE}")

n_written = 0
n_aug_written = 0
n_missing = 0
n_failed = 0
n_empty = 0

# NEW: keeps source relationship for later renaming / AB building
manifest = {
    "images_dir": str(IMAGES_DIR),
    "coco_json": str(COCO_JSON),
    "aug_per_image": AUG_PER_IMAGE,
    "category_remap": {str(k): int(v) for k, v in cat_id_to_train.items()},
    "entries": {}
}

for image_id in image_ids:
    im = images[image_id]
    file_name = im.get("file_name", "")

    # Map COCO file_name to a local filename
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
    mask = np.zeros((H, W), dtype=np.uint8)

    anns = ann_by_img.get(image_id, [])
    if len(anns) == 0:
        n_empty += 1

    for ann in anns:
        seg = ann.get("segmentation", None)
        if not seg:
            continue

        # Skip RLE if present
        if isinstance(seg, dict):
            continue

        cat = ann.get("category_id", None)
        if cat is None:
            continue

        train_id = int(cat_id_to_train.get(cat, 1))
        pts_list = poly_to_pts(seg, W, H)

        for pts in pts_list:
            temp = np.zeros_like(mask)
            cv2.fillPoly(temp, [pts], train_id)

            # Only write where mask is still background
            mask[(mask == 0) & (temp > 0)] = temp[(mask == 0) & (temp > 0)]
    # -----------------------
    # Save original outputs
    # -----------------------
    base_name = f"{img_path.stem}.png"

    out_mask = OUT_MASKS / base_name
    cv2.imwrite(str(out_mask), mask)

    out_viz = OUT_VIZ / f"{img_path.stem}_viz.png"
    cv2.imwrite(str(out_viz), make_viz(mask))

    color_mask = colorize_mask(mask, PALETTE)
    out_color = OUT_COLOR / base_name
    cv2.imwrite(str(out_color), color_mask)

    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    out_overlay = OUT_OVERLAY / base_name
    cv2.imwrite(str(out_overlay), overlay)

    manifest["entries"][base_name] = {
        "image_id": int(image_id),
        "source_image": img_path.name,
        "source_stem": img_path.stem,
        "is_aug": False,
        "aug_index": 0,
        "height": int(H),
        "width": int(W),
        "mask_path": str(out_mask),
        "viz_path": str(out_viz),
        "color_path": str(out_color),
        "overlay_path": str(out_overlay),
        "nonzero_pixels": int(np.count_nonzero(mask)),
        "max_label": int(mask.max()),
    }

    uniq = np.unique(mask)
    if mask.max() == 0:
        print(f"{img_path.name} | EMPTY MASK | uniq={uniq[:10]}")
    else:
        print(f"{img_path.name} | ORIGINAL | max={int(mask.max())} | nonzero={int(np.count_nonzero(mask))} | uniq={uniq[:10]}")

    n_written += 1

    # -----------------------
    # Save augmented outputs
    # -----------------------
    for i in range(AUG_PER_IMAGE):
        aug_img, aug_mask = augment_no_scale(img.copy(), mask.copy())
        aug_name = f"{img_path.stem}_aug{i + 1}"
        aug_file = f"{aug_name}.png"

        # optional: save augmented image too
        aug_img_path = IMAGES_DIR / aug_file
        cv2.imwrite(str(aug_img_path), aug_img)

        aug_out_mask = OUT_MASKS / aug_file
        cv2.imwrite(str(aug_out_mask), aug_mask)

        aug_out_viz = OUT_VIZ / f"{aug_name}_viz.png"
        cv2.imwrite(str(aug_out_viz), make_viz(aug_mask))

        aug_color_mask = colorize_mask(aug_mask, PALETTE)
        aug_out_color = OUT_COLOR / aug_file
        cv2.imwrite(str(aug_out_color), aug_color_mask)

        aug_overlay = cv2.addWeighted(aug_img, 0.7, aug_color_mask, 0.3, 0)
        aug_out_overlay = OUT_OVERLAY / aug_file
        cv2.imwrite(str(aug_out_overlay), aug_overlay)

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
            "mask_path": str(aug_out_mask),
            "viz_path": str(aug_out_viz),
            "color_path": str(aug_out_color),
            "overlay_path": str(aug_out_overlay),
            "nonzero_pixels": int(np.count_nonzero(aug_mask)),
            "max_label": int(aug_mask.max()),
        }

        aug_uniq = np.unique(aug_mask)
        print(f"    -> {aug_file} | AUG | max={int(aug_mask.max())} | nonzero={int(np.count_nonzero(aug_mask))} | uniq={aug_uniq[:10]}")

        n_aug_written += 1

# Save manifest at end
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
print(f" - raw masks:     {OUT_MASKS}")
print(f" - grayscale viz: {OUT_VIZ}")
print(f" - colored masks: {OUT_COLOR}")
print(f" - overlays:      {OUT_OVERLAY}")
print(f" - manifest:      {MANIFEST_JSON}")