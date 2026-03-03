from pathlib import Path
import json
import cv2
import numpy as np

# =========================
# Paths
# =========================
IMAGES_DIR = Path("data/raw/Shampoo")                 # your images
COCO_JSON  = Path("data/raw/Shampoo/result.json")     # COCO export

OUT_MASKS   = Path("data/interim/Shampoo/masks")         # raw label-id masks (0..K)  (will look black)
OUT_VIZ     = Path("data/interim/Shampoo/masks_viz")     # grayscale view (0..255)
OUT_COLOR   = Path("data/interim/Shampoo/masks_color")   # colored mask (per-class color)
OUT_OVERLAY = Path("data/interim/Shampoo/masks_overlay") # colored mask overlaid on image

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
# If you have more than 6 classes, extend this list.

"""
# FOR CONTRABAND METAL:
PALETTE = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # blue
    2: (0, 255, 0),       # green
    3: (0, 0, 255),       # red
    4: (255, 255, 0),     # cyan
    5: (0, 255, 255),     # yellow
    6: (255, 0, 255),     # magenta
}
"""

#FOR SHAMPOO:
PALETTE = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # blue
}

#FOR NON_CONTRABAND:
"""
PALETTE = {
    0: (0, 0, 0),         # background
    1:  (255, 0, 0),        # red
    2:  (0, 255, 0),        # green
    3:  (0, 0, 255),        # blue
    4:  (255, 255, 0),      # yellow
    5:  (255, 0, 255),      # magenta
    6:  (0, 255, 255),      # cyan
    7:  (255, 255, 255),    # white
    8:  (128, 0, 0),        # dark red
    9:  (0, 128, 0),        # dark green
    10: (0, 0, 128),        # dark blue
    11: (128, 128, 0),      # olive
    12: (128, 0, 128),      # purple
    13: (0, 128, 128),      # teal
    14: (128, 128, 128),    # gray
}
"""


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

        #If coordinates look normalized (0..1), scale to pixel coords
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

    # color each class
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


if contains_rle_annotations():
    print("   Detected RLE-style segmentations (segmentation is a dict).")
    print("   This script currently handles polygon segmentations only.")
    print("   If you need RLE support, install pycocotools and decode RLE masks.")

# =========================
# Main
# =========================
image_ids = list(images.keys())
print(f"Found {len(image_ids)} COCO images in {COCO_JSON}")
print(f"Category remap (COCO id -> train id): {cat_id_to_train}")

n_written = 0
n_missing = 0
n_failed  = 0
n_empty   = 0

for image_id in image_ids:
    im = images[image_id]
    file_name = im.get("file_name", "")

    # Map COCO file_name to a local filename
    img_path = IMAGES_DIR / Path(file_name).name

    if not img_path.exists():
        print(f" Missing image: COCO file_name='{file_name}' -> tried '{img_path}'")
        n_missing += 1
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        print(f" Could not read image: {img_path}")
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

        # Skip RLE if present (handled separately if you enable pycocotools)
        if isinstance(seg, dict):
            continue

        cat = ann.get("category_id", None)
        if cat is None:
            continue

        train_id = int(cat_id_to_train.get(cat, 1))
        pts_list = poly_to_pts(seg, W, H)

        for pts in pts_list:
            # Fill polygon region with train_id
            cv2.fillPoly(mask, [pts], train_id)

    # -----------------------
    # Save outputs
    # -----------------------

    # 1) raw label ids (0..K) -> will look black in a viewer, but correct for ML
    out_mask = OUT_MASKS / f"{img_path.stem}.png"
    cv2.imwrite(str(out_mask), mask)

    # 2) grayscale visualization (0..255)
    out_viz = OUT_VIZ / f"{img_path.stem}_viz.png"
    cv2.imwrite(str(out_viz), make_viz(mask))

    # 3) colored mask
    color_mask = colorize_mask(mask, PALETTE)
    out_color = OUT_COLOR / f"{img_path.stem}.png"
    cv2.imwrite(str(out_color), color_mask)

    # 4) overlay on original image
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)
    out_overlay = OUT_OVERLAY / f"{img_path.stem}.png"
    cv2.imwrite(str(out_overlay), overlay)

    # Debug line
    uniq = np.unique(mask)
    if mask.max() == 0:
        n_empty += 1
    print(f" {img_path.name} | max={int(mask.max())} | nonzero={int(np.count_nonzero(mask))} | uniq={uniq[:10]}")

    n_written += 1

print("\n===== Summary =====")
print(f"Wrote masks:   {n_written}")
print(f"Missing imgs:  {n_missing}")
print(f"Read failed:   {n_failed}")
print(f"Empty masks:   {n_empty}")
print("Outputs:")
print(f" - raw masks:     {OUT_MASKS}")
print(f" - grayscale viz: {OUT_VIZ}")
print(f" - colored masks: {OUT_COLOR}")
print(f" - overlays:      {OUT_OVERLAY}")