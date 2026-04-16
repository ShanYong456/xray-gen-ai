#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

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


# ============================================================
# Fixed palette in BGR
# ============================================================
PALETTE_BGR = {
    0: (0, 0, 0),   # background
    1: (0, 255, 0), # Shampoo
}


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
    if isinstance(segmentation, list):
        if len(segmentation) > 0 and isinstance(segmentation[0], dict):
            return None

        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    return None


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

    m = mask_utils.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    return m


def imread_any(path: Path):
    """
    Read image safely.
    - preserves grayscale if grayscale file
    - preserves color if color file
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_gray_u8(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 grayscale."""
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 1:
        gray = img[:, :, 0]
    elif img.ndim == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", required=True, type=str,
                   help="Directory containing original source images used in COCO JSON")
    p.add_argument("--coco_json", required=True, type=str,
                   help="Path to COCO result.json")
    p.add_argument("--out_dir", default="data/object_library", type=str,
                   help="Output root dir")

    p.add_argument("--pad", default=8, type=int,
                   help="Padding (pixels) around cropped object")
    p.add_argument("--min_area", default=50, type=int,
                   help="Skip tiny masks smaller than this area")
    p.add_argument("--max_per_category", default=0, type=int,
                   help="If >0, limit saved cutouts per category")

    p.add_argument("--use_category_color", action="store_true",
                   help="If set, semantic cutout RGB is filled with category color; otherwise white.")
    p.add_argument("--cat_to_palette_json", default="", type=str,
                   help='JSON mapping category_name -> palette_index, e.g. {"Shampoo":1}')
    p.add_argument("--palette_json", default="", type=str,
                   help="Optional fallback JSON mapping category_name -> [R,G,B]")

    p.add_argument("--save_semantic_rgba", action="store_true",
                   help="Save semantic RGBA cutout for A")
    p.add_argument("--save_gray_crop", action="store_true",
                   help="Save real grayscale masked crop for B")
    p.add_argument("--save_mask", action="store_true",
                   help="Save binary mask crop")
    p.add_argument("--save_preview", action="store_true",
                   help="Save a side-by-side preview image")

    return p.parse_args()


def main():
    args = parse_args()

    coco = json.load(open(args.coco_json, "r"))
    images = coco.get("images", [])
    anns = coco.get("annotations", [])
    cats = coco.get("categories", [])

    if not images or not anns or not cats:
        raise ValueError("JSON does not look like COCO: must contain images/annotations/categories")

    print("\n=== COCO Categories ===")
    for c in cats:
        raw = c["name"]
        safe = safe_name(raw)
        print(f"id={c['id']:>3} | raw='{raw}' | safe='{safe}'")
    print("=======================\n")

    images_dir = Path(args.images_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Output folders
    out_sem = out_root / "semantic_rgba"
    out_gray = out_root / "gray"
    out_mask = out_root / "mask"
    out_preview = out_root / "preview"

    if args.save_semantic_rgba:
        out_sem.mkdir(parents=True, exist_ok=True)
    if args.save_gray_crop:
        out_gray.mkdir(parents=True, exist_ok=True)
    if args.save_mask:
        out_mask.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        out_preview.mkdir(parents=True, exist_ok=True)

    id_to_img = {im["id"]: im for im in images}
    id_to_cat = {c["id"]: c for c in cats}

    cat_to_pal = load_cat_to_palette(args.cat_to_palette_json)
    palette_rgb_fallback = load_palette_rgb(args.palette_json)

    saved_per_cat = {}
    total_saved = 0
    total_skipped_missing = 0

    for ann in anns:
        img_id = ann.get("image_id")
        cat_id = ann.get("category_id")
        if img_id not in id_to_img or cat_id not in id_to_cat:
            continue

        im_meta = id_to_img[img_id]
        H, W = int(im_meta["height"]), int(im_meta["width"])

        cat_name_raw = id_to_cat[cat_id].get("name", f"cat_{cat_id}")
        cat_name = safe_name(cat_name_raw)

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

        file_name = im_meta.get("file_name", "")
        if not file_name:
            continue

        file_name_clean = Path(file_name).name  # get only filename
        img_path = images_dir / file_name_clean
        if not img_path.exists():
            print(f"[WARN] missing source image: {img_path}")
            total_skipped_missing += 1
            continue

        src = imread_any(img_path)
        gray = to_gray_u8(src)

        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())

        pad = int(args.pad)
        y1 = max(0, y1 - pad)
        y2 = min(H - 1, y2 + pad)
        x1 = max(0, x1 - pad)
        x2 = min(W - 1, x2 + pad)

        crop_m = mask[y1:y2 + 1, x1:x2 + 1].astype(np.uint8)   # (h,w), 0/1
        crop_gray = gray[y1:y2 + 1, x1:x2 + 1]                 # real grayscale pixels

        h, w = crop_m.shape
        crop_alpha = (crop_m * 255).astype(np.uint8)

        # ----------------------------
        # semantic RGBA cutout for A
        # ----------------------------
        if args.use_category_color and cat_name in cat_to_pal:
            pal_idx = int(cat_to_pal[cat_name])
            bgr = PALETTE_BGR.get(pal_idx, (255, 255, 255))
            rgb = (bgr[2], bgr[1], bgr[0])
        elif args.use_category_color:
            rgb = palette_rgb_fallback.get(cat_name, (255, 255, 255))
        else:
            rgb = (255, 255, 255)

        semantic_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        semantic_rgba[..., 0] = int(rgb[0])
        semantic_rgba[..., 1] = int(rgb[1])
        semantic_rgba[..., 2] = int(rgb[2])
        semantic_rgba[..., 3] = crop_alpha

        # ----------------------------
        # real grayscale masked crop for B
        # ----------------------------
        gray_masked = np.zeros((h, w), dtype=np.uint8)
        gray_masked[crop_m > 0] = crop_gray[crop_m > 0]

        # ----------------------------
        # naming
        # ----------------------------
        ann_id = int(ann.get("id", total_saved))
        stem = f"{ann_id:06d}_{cat_name}"

        # Save by category subfolders
        if args.save_semantic_rgba:
            save_dir = out_sem / cat_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(semantic_rgba, cv2.COLOR_RGBA2BGRA))

        if args.save_gray_crop:
            save_dir = out_gray / cat_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), gray_masked)

        if args.save_mask:
            save_dir = out_mask / cat_name
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), crop_alpha)

        if args.save_preview:
            save_dir = out_preview / cat_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # preview: semantic RGB | gray_masked | mask
            sem_rgb = semantic_rgba[..., :3].copy()
            gray_rgb = cv2.cvtColor(gray_masked, cv2.COLOR_GRAY2BGR)
            mask_rgb = cv2.cvtColor(crop_alpha, cv2.COLOR_GRAY2BGR)

            preview = np.concatenate([sem_rgb[:, :, ::-1], gray_rgb, mask_rgb], axis=1)
            out_path = save_dir / f"{stem}.png"
            cv2.imwrite(str(out_path), preview)

        saved_per_cat[cat_name] = saved_per_cat.get(cat_name, 0) + 1
        total_saved += 1

    print(f"[DONE] total saved objects = {total_saved}")
    if total_skipped_missing > 0:
        print(f"[WARN] skipped missing source images = {total_skipped_missing}")

    print("\nSaved per category:")
    for k in sorted(saved_per_cat.keys()):
        print(f"  {k}: {saved_per_cat[k]}")


if __name__ == "__main__":
    main()



"""
python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Contraband/Metal/result.json --out_dir data/raw/Contraband/Metal/Cropped --use_category_color   --cat_to_palette_json data/raw/Contraband/Metal/color_palette.json



python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Non-Contraband/result.json --out_dir data/raw/Non-Contraband/Cropped --use_category_color   --cat_to_palette_json data/raw/Non-Contraband/color_palette.json


python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo/result.json --out_dir data/raw/Shampoo/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo/color_palette.json

python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo_Blade/result.json --out_dir data/raw/Shampoo_Blade/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo_Blade/color_palette.json

python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py --coco_json data/raw/Shampoo_nobackground/result.json --out_dir data/raw/Shampoo_nobackground/Cropped --use_category_color   --cat_to_palette_json data/raw/Shampoo_nobackground/color_palette.json


python Codes_Notebooks/Pix2Pix/pix2pix_object_library.py \
  --images_dir data/raw/Shampoo_nobackground \
  --coco_json data/raw/Shampoo_nobackground/result.json \
  --out_dir data/raw/Shampoo_nobackground/Cropped_Library --use_category_color   --cat_to_palette_json data/raw/Shampoo_nobackground/color_palette.json \
  --pad 8 \
  --min_area 50 \
  --use_category_color \
  --save_semantic_rgba \
  --save_gray_crop \
  --save_mask \
  --save_preview


"""
