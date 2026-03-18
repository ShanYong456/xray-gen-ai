#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import cv2

# ============================================================
# Fixed palette (YOUR exact palette) in BGR
# ============================================================
"""
# CONTRABAND METAL:

PALETTE_BGR = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),       # blue
    2: (0, 255, 0),       # green
    3: (0, 0, 255),       # red
    4: (255, 255, 0),     # cyan
    5: (0, 255, 255),     # yellow
    6: (255, 0, 255),     # magenta
}
"""

# Shampoo:
PALETTE_BGR = {
    0: (0, 0, 0),         # background
    1: (0, 255, 0),       # blue
}


"""
# NON-CONTRABAND:
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

def safe_name(s: str) -> str:
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep) or "category"

def load_cat_to_palette(path: str):
    """
    JSON: { "Blade": 1, "Vape": 2, ... }
    Returns: { "Blade": 1, "Vape": 2, ... } with safe_name() applied to keys.
    """
    if not path:
        return {}
    d = json.load(open(path, "r"))
    return {safe_name(k): int(v) for k, v in d.items()}

def load_palette_rgb(path: str):
    """
    Optional fallback JSON: { "Blade": [R,G,B], ... }
    Only used if --use_category_color is set AND cat_to_palette has no entry.
    """
    if not path:
        return {}
    d = json.load(open(path, "r"))
    return {safe_name(k): tuple(map(int, v)) for k, v in d.items()}

def ann_to_mask_poly(segmentation, H, W):
    """COCO polygon segmentation -> binary mask (H,W)"""
    mask = np.zeros((H, W), dtype=np.uint8)
    # segmentation can be list of polygons, each polygon is [x1,y1,x2,y2,...]
    if isinstance(segmentation, list):
        # NOTE: polygon format is list[list[float]]; RLE is dict
        # If it's list of dicts, it's not polygon -> handled by RLE decoder
        if len(segmentation) > 0 and isinstance(segmentation[0], dict):
            return None

        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    return None  # not polygon

def ann_to_mask_rle(segmentation, H, W):
    """Try decode COCO RLE if present. Requires pycocotools. Returns binary mask or None."""
    try:
        from pycocotools import mask as mask_utils  # type: ignore
    except Exception:
        return None

    if isinstance(segmentation, dict) and "counts" in segmentation:
        rle = segmentation
    elif isinstance(segmentation, list) and len(segmentation) > 0 and isinstance(segmentation[0], dict):
        rle = segmentation[0]
    else:
        return None

    m = mask_utils.decode(rle)  # (H,W) or (H,W,1)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    return m

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--coco_json", required=True, type=str, help="Path to COCO result.json")
    p.add_argument("--out_dir", default="data/object_library", type=str, help="Output root dir")
    p.add_argument("--pad", default=8, type=int, help="Padding (pixels) around cropped object")
    p.add_argument("--min_area", default=50, type=int, help="Skip tiny masks smaller than this area")
    p.add_argument("--use_category_color", action="store_true",
                   help="If set, cutout RGB is filled with the category color (from --cat_to_palette_json / PALETTE_BGR). Otherwise white.")
    p.add_argument("--cat_to_palette_json", default="", type=str,
                   help='JSON mapping category_name -> palette_index (0..6), e.g. {"Blade":1,"Vape":2}')
    p.add_argument("--palette_json", default="", type=str,
                   help="Optional fallback JSON mapping category_name -> [R,G,B]. Used if --use_category_color and cat not in cat_to_palette_json.")
    p.add_argument("--max_per_category", default=0, type=int,
                   help="If >0, limit saved cutouts per category")
    return p.parse_args()

def main():
    args = parse_args()
    coco = json.load(open(args.coco_json, "r"))

    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    print("\n=== COCO Categories ===")
    for c in cats:
        raw = c["name"]
        safe = safe_name(raw)
        print(f"id={c['id']:>3} | raw='{raw}' | safe='{safe}'")
    print("=======================\n")

    if not images or not anns or not cats:
        raise ValueError("JSON does not look like COCO: must contain images/annotations/categories")

    id_to_img = {im["id"]: im for im in images}
    id_to_cat = {c["id"]: c for c in cats}

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Mapping: category_name -> palette index (0..6)
    cat_to_pal = load_cat_to_palette(args.cat_to_palette_json)

    # Optional fallback: category_name -> (R,G,B)
    palette_rgb_fallback = load_palette_rgb(args.palette_json)

    saved_per_cat = {}
    total_saved = 0

    for ann in anns:
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id not in id_to_img or cat_id not in id_to_cat:
            continue

        im = id_to_img[img_id]
        H, W = int(im["height"]), int(im["width"])

        cat_name_raw = id_to_cat[cat_id].get("name", f"cat_{cat_id}")
        cat_name = safe_name(cat_name_raw)

        # Per-category folder
        cat_dir = out_root / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        if args.max_per_category > 0 and saved_per_cat.get(cat_name, 0) >= args.max_per_category:
            continue

        seg = ann.get("segmentation", None)
        if seg is None:
            continue

        mask = ann_to_mask_poly(seg, H, W)
        if mask is None:
            mask = ann_to_mask_rle(seg, H, W)

        if mask is None:
            continue

        area = int(mask.sum())
        if area < args.min_area:
            continue

        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            continue

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        pad = int(args.pad)
        y1 = max(0, y1 - pad); y2 = min(H - 1, y2 + pad)
        x1 = max(0, x1 - pad); x2 = min(W - 1, x2 + pad)

        crop_m = mask[y1:y2+1, x1:x2+1]  # (h,w) in {0,1}

        # Build RGBA cutout
        h, w = crop_m.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # ----------------------------
        # COLOR LINKING (the key part)
        # ----------------------------
        if args.use_category_color and cat_name in cat_to_pal:
            # Prefer: category -> palette index -> PALETTE_BGR
            pal_idx = int(cat_to_pal[cat_name])
            if pal_idx not in PALETTE_BGR:
                print(f"[WARN] {cat_name=} has pal_idx={pal_idx} not in PALETTE_BGR (0..6). Using WHITE fallback.")
            if cat_name in cat_to_pal:
                pal_idx = int(cat_to_pal[cat_name])
                bgr = PALETTE_BGR.get(pal_idx, (255, 255, 255))
                rgb = (bgr[2], bgr[1], bgr[0])  # BGR -> RGB
            
            else:
                # Fallback: optional direct RGB json
                rgb = palette_rgb_fallback.get(cat_name, (255, 255, 255))
        else:
            rgb = (255, 255, 255)

        rgba[..., 0] = int(rgb[0])
        rgba[..., 1] = int(rgb[1])
        rgba[..., 2] = int(rgb[2])
        rgba[..., 3] = (crop_m * 255).astype(np.uint8)

        # Save file name
        ann_id = ann.get("id", total_saved)
        out_path = cat_dir / f"{int(ann_id):06d}.png"

        # cv2 wants BGRA on disk
        cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        saved_per_cat[cat_name] = saved_per_cat.get(cat_name, 0) + 1
        total_saved += 1

        if total_saved % 500 == 0:
            print(f"[saved {total_saved}] ...")

    print("\nDONE")
    print("Total saved:", total_saved)
    print("Per category:")
    for k in sorted(saved_per_cat.keys()):
        print(f"  {k}: {saved_per_cat[k]}")

if __name__ == "__main__":
    main()



"""
python notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Contraband/Metal/result.json --out_dir data/raw/Contraband/Metal/Cropped --use_category_color   --cat_to_palette_json data/raw/Contraband/Metal/color_palette.json



python notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Non-Contraband/result.json --out_dir data/raw/Non-Contraband/Cropped --use_category_color   --cat_to_palette_json data/raw/Non-Contraband/color_palette.json


python notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo/result.json --out_dir data/raw/Shampoo/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo/color_palette.json

python notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo_Blade/result.json --out_dir data/raw/Shampoo_Blade/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo_Blade/color_palette.json

python notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo_nobackground/result.json --out_dir data/raw/Shampoo_nobackground/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo_nobackground/color_palette.json
"""