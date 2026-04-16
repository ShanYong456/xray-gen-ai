#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import cv2

SIZE = 1024

def ann_to_mask_poly(segmentation, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    if isinstance(segmentation, list):
        if len(segmentation) > 0 and isinstance(segmentation[0], dict):
            return None
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask
    return None

def ann_to_mask_rle(segmentation, H, W):
    try:
        from pycocotools import mask as mask_utils
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
    return (m > 0).astype(np.uint8) * 255

def main():
    coco_json = Path("data/interim/GAN/Empty_Tray_mask/result.json")  # <-- CHANGE THIS
    out_dir = Path("data/interim/GAN/Empty_Tray_mask/Mask")               # <-- OUTPUT FOLDER
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = json.load(open(coco_json, "r"))
    images = coco["images"]
    anns = coco["annotations"]
    cats = coco["categories"]

    id_to_img = {im["id"]: im for im in images}
    id_to_cat = {c["id"]: c for c in cats}

    # Find tray category ids
    tray_cat_ids = []
    for c in cats:
        name = str(c.get("name","")).lower()
        if "tray" in name:
            tray_cat_ids.append(c["id"])

    if not tray_cat_ids:
        raise RuntimeError(f"No category containing 'tray'. Categories: {[c.get('name') for c in cats]}")

    saved = 0
    for ann in anns:
        if ann.get("category_id") not in tray_cat_ids:
            continue

        img = id_to_img[ann["image_id"]]
        H, W = int(img["height"]), int(img["width"])

        seg = ann.get("segmentation")
        if seg is None:
            continue

        mask = ann_to_mask_poly(seg, H, W)
        if mask is None:
            mask = ann_to_mask_rle(seg, H, W)
        if mask is None:
            continue

        # Resize to SIZE and binarize
        if mask.shape != (SIZE, SIZE):
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)

        # name using image filename (nice for debugging)
        base = Path(img.get("file_name", f"img_{ann['image_id']}")).stem
        out_path = out_dir / f"{base}_traymask.png"
        cv2.imwrite(str(out_path), mask)
        saved += 1

    print("Saved tray masks:", saved, "to", out_dir)

if __name__ == "__main__":
    main()