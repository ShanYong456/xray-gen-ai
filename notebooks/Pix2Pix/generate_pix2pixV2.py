from pathlib import Path
import argparse
import json
import random
import subprocess
import shutil

import cv2
import numpy as np
from PIL import Image


# =============================================================================
# CONFIG — must match training options exactly
# =============================================================================

SIZE_H = 1024
SIZE_W = 1024

# Stored with cv2.imwrite => BGR
PALETTE_BGR = {
    0: (0, 0, 0),
    1: (0, 255, 0),    # shampoo -> RGB green
    2: (255, 0, 0),    # tray    -> RGB blue
    3: (0, 0, 255),    # blade   -> RGB red
}
OVERLAP_BGR = (255, 255, 0)  # shampoo + tray -> cyan in RGB after cv2 write
OVERLAP_BLADE_BGR = (255, 0, 255)  # blade + tray -> magenta in RGB after cv2 write

MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage17_BladeMaskSyn"
PIX2PIX_DIR = Path("external/pix2pix")

TRAIN_CFG = dict(
    input_nc=7,
    norm="instance",
    use_appearance_channel=False,
    use_tray_mask=True,
    class_nc=3,
    thickness_nc=1,
    canvas_h=1024,
    canvas_w=1024,
    pad_to_canvas=True,
    tray_mask_thr=0.5,
    tray_cc_close_px=2,
    tray_mask_dilate_px=0,
)

# Search params
X_SEARCH_STEP = 12
MAX_TRANSFORM_CANDIDATES = 8

# Match training-side Q_score weights in pix2pix_model.py
TRAIN_SCORE_WEIGHTS = {
    "gan": 0.10,
    "l1": 0.35,
    "grad": 0.15,
    "lap": 0.10,
    "ssim": 0.20,
    "stats": 0.10,
}


# =============================================================================
# Geometry helpers
# =============================================================================

def poly_to_pts_list(seg, W, H):
    polys = seg if (isinstance(seg, list) and seg and isinstance(seg[0], list)) else [seg]
    result = []
    for poly in polys:
        if not poly or len(poly) < 6:
            continue
        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if arr[:, 0].max() <= 1.5 and arr[:, 1].max() <= 1.5:
            arr[:, 0] *= W
            arr[:, 1] *= H
        arr = np.clip(arr, [0, 0], [W - 1, H - 1])
        result.append(arr.astype(np.int32))
    return result


def rasterize_instance_mask(pts_list, W, H):
    m = np.zeros((H, W), dtype=np.uint8)
    for pts in pts_list:
        cv2.fillPoly(m, [pts], 255)
    return m


def tight_crop(mask_bin):
    ys, xs = np.where(mask_bin > 0)
    if not len(xs):
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


# =============================================================================
# Tray helpers
# =============================================================================

def largest_cc(m01):
    m = (m01 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == k).astype(np.uint8)


def morph_close(m01, px):
    if px <= 0:
        return m.astype(np.uint8) if (m := m01) is not None else m01
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.morphologyEx(m01.astype(np.uint8), cv2.MORPH_CLOSE, k)


def compute_fit_to_canvas(src_w: int, src_h: int, target_w: int, target_h: int):
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    return scale, new_w, new_h, x_off, y_off


def map_bbox_to_canvas(x0, y0, x1, y1, src_w, src_h, target_w, target_h):
    scale, _, _, x_off, y_off = compute_fit_to_canvas(src_w, src_h, target_w, target_h)

    nx0 = int(round(x0 * scale)) + x_off
    ny0 = int(round(y0 * scale)) + y_off
    nx1 = int(round(x1 * scale)) + x_off
    ny1 = int(round(y1 * scale)) + y_off

    return nx0, ny0, nx1, ny1, scale, x_off, y_off

def resize_and_pad_mask(mask, target_h, target_w):
    h, w = mask.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas = np.zeros((target_h, target_w), dtype=resized.dtype)

    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def dilate_bin(m01, px):
    if px <= 0:
        return m01.astype(np.uint8)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(m01.astype(np.uint8), k, iterations=1)


def preprocess_tray_mask(mask_gray, thr=0.5, invert=False, close_px=2, dilate_px=0):
    T = (mask_gray > int(np.clip(thr * 255, 0, 255))).astype(np.uint8)
    if invert:
        T = 1 - T
    T = largest_cc(T)
    T = morph_close(T, close_px)
    T = dilate_bin(T, dilate_px)
    return T.astype(bool)


def load_tray_masks(tray_dir: Path, out_h: int, out_w: int, thr=0.5, invert=False, close_px=2, dilate_px=0):
    paths = sorted(tray_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No tray mask PNGs in {tray_dir}")
    masks = []
    valid_paths = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (out_h, out_w):
            m = resize_and_pad_mask(m, out_h, out_w)
        masks.append(preprocess_tray_mask(m, thr, invert, close_px, dilate_px))
        valid_paths.append(p)
    if not masks:
        raise SystemExit(f"Could not read tray masks from {tray_dir}")
    print(f"[tray] loaded {len(masks)} masks")
    return masks, valid_paths


def build_tray_condition_from_mask(mask_bool: np.ndarray) -> np.ndarray:
    h, w = mask_bool.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask_bool] = PALETTE_BGR[2]
    return out


# =============================================================================
# Cutout library
# =============================================================================

def normalize_xray_intensity(img_np):
    img = img_np.astype(np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    img = (img - p1) / (p99 - p1 + 1e-6)
    return (img * 255).astype(np.uint8)


def build_real_cutout_library(coco, images_dir: Path):
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats = coco.get("categories", [])
    cats_by_id = {c["id"]: c for c in cats}
    sorted_cats = sorted(cats, key=lambda c: c["id"])
    cat_to_train = {c["id"]: i + 1 for i, c in enumerate(sorted_cats)}

    ann_by_img = {}
    for ann in coco.get("annotations", []):
        seg = ann.get("segmentation")
        if not seg or isinstance(seg, dict):
            continue
        ann_by_img.setdefault(ann["image_id"], []).append(ann)

    lib = []
    for img_id, anns in ann_by_img.items():
        im = images_by_id.get(img_id)
        if not im:
            continue

        img_path = images_dir / Path(im.get("file_name", "")).name
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape[:2]

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or isinstance(seg, dict):
                continue

            pts = poly_to_pts_list(seg, W, H)
            if not pts:
                continue

            inst = rasterize_instance_mask(pts, W, H)
            bbox = tight_crop(inst)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            mask_crop = inst[y0:y1 + 1, x0:x1 + 1]
            gray_crop = img_gray[y0:y1 + 1, x0:x1 + 1].copy()
            gray_crop[mask_crop == 0] = 0
            gray_crop = normalize_xray_intensity(gray_crop)

            cat_id = ann.get("category_id")
            train_id = int(cat_to_train.get(cat_id, 1))

            orig_w = int(x1 - x0 + 1)
            orig_h = int(y1 - y0 + 1)
            orig_cx = 0.5 * (x0 + x1)
            orig_cy = 0.5 * (y0 + y1)

            fit_x0, fit_y0, fit_x1, fit_y1, fit_scale, fit_x_off, fit_y_off = map_bbox_to_canvas(
                x0, y0, x1, y1, W, H, SIZE_W, SIZE_H
            )

            fit_w = int(fit_x1 - fit_x0 + 1)
            fit_h = int(fit_y1 - fit_y0 + 1)
            fit_cx = 0.5 * (fit_x0 + fit_x1)
            fit_cy = 0.5 * (fit_y0 + fit_y1)

            lib.append({
                "train_id": train_id,
                "class_name": cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}"),
                "image_id": img_id,
                "file_name": Path(im.get("file_name", "")).name,
                "mask_bin": mask_crop,
                "gray": gray_crop,
                "orig_x0": int(x0),
                "orig_y0": int(y0),
                "orig_x1": int(x1),
                "orig_y1": int(y1),
                "orig_w": orig_w,
                "orig_h": orig_h,
                "orig_cx": float(orig_cx),
                "orig_cy": float(orig_cy),
                "src_W": int(W),
                "src_H": int(H),
                "fit_scale": float(fit_scale),
                "fit_x_off": int(fit_x_off),
                "fit_y_off": int(fit_y_off),
                "fit_x0": int(fit_x0),
                "fit_y0": int(fit_y0),
                "fit_x1": int(fit_x1),
                "fit_y1": int(fit_y1),
                "fit_w": int(fit_w),
                "fit_h": int(fit_h),
                "fit_cx": float(fit_cx),
                "fit_cy": float(fit_cy),
            })

    if not lib:
        raise SystemExit("No valid cutouts from COCO + images_dir")

    # Only shampoo is useful here
    lib = [x for x in lib if int(x["train_id"]) == 1]
    if not lib:
        raise SystemExit("No shampoo cutouts found in COCO/images_dir.")

    print(f"[cutouts] built {len(lib)} shampoo instances")
    return lib


def build_blade_mask_library(blade_mask_dir: Path):
    paths = sorted(blade_mask_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No blade mask PNGs in {blade_mask_dir}")

    lib = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        m = (m > 0).astype(np.uint8) * 255
        bbox = tight_crop(m)
        if bbox is None:
            continue

        x0, y0, x1, y1 = bbox
        mask_crop = m[y0:y1 + 1, x0:x1 + 1]

        fit_x0, fit_y0, fit_x1, fit_y1, fit_scale, fit_x_off, fit_y_off = map_bbox_to_canvas(
            x0, y0, x1, y1, m.shape[1], m.shape[0], SIZE_W, SIZE_H
        )

        fit_w = int(fit_x1 - fit_x0 + 1)
        fit_h = int(fit_y1 - fit_y0 + 1)
        fit_cx = 0.5 * (fit_x0 + fit_x1)
        fit_cy = 0.5 * (fit_y0 + fit_y1)

        lib.append({
            "train_id": 3,
            "class_name": "Blade",
            "image_id": -1,
            "file_name": p.name,
            "mask_bin": mask_crop,
            "gray": None,
            "orig_x0": int(x0),
            "orig_y0": int(y0),
            "orig_x1": int(x1),
            "orig_y1": int(y1),
            "orig_w": int(x1 - x0 + 1),
            "orig_h": int(y1 - y0 + 1),
            "orig_cx": float(0.5 * (x0 + x1)),
            "orig_cy": float(0.5 * (y0 + y1)),
            "src_W": int(m.shape[1]),
            "src_H": int(m.shape[0]),
            "fit_scale": float(fit_scale),
            "fit_x_off": int(fit_x_off),
            "fit_y_off": int(fit_y_off),
            "fit_x0": int(fit_x0),
            "fit_y0": int(fit_y0),
            "fit_x1": int(fit_x1),
            "fit_y1": int(fit_y1),
            "fit_w": int(fit_w),
            "fit_h": int(fit_h),
            "fit_cx": float(fit_cx),
            "fit_cy": float(fit_cy),
        })

    if not lib:
        raise SystemExit("No valid blade mask cutouts found.")

    print(f"[cutouts] built {len(lib)} blade masks")
    return lib


# =============================================================================
# Cutout transforms
# =============================================================================

def transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max):
    mask_bin = (item["mask_bin"] > 127).astype(np.uint8) * 255
    gray = None if item.get("gray") is None else item["gray"].copy()

    base_fit_scale = float(item.get("fit_scale", 1.0))
    s = rng.uniform(scale_min, scale_max) * base_fit_scale
    base_h, base_w = mask_bin.shape[:2]
    aug_w = max(1, int(round(base_w * s)))
    aug_h = max(1, int(round(base_h * s)))

    mask_soft = cv2.resize(mask_bin.astype(np.float32), (aug_w, aug_h), interpolation=cv2.INTER_LINEAR)
    if gray is not None:
        gray = cv2.resize(gray, (aug_w, aug_h), interpolation=cv2.INTER_LINEAR)

    ang = rng.uniform(rot_min, rot_max)

    h0, w0 = mask_soft.shape[:2]
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
    M[0, 2] += nw / 2 - w0 / 2
    M[1, 2] += nh / 2 - h0 / 2

    mask_soft = cv2.warpAffine(mask_soft, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    if gray is not None:
        gray = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)

    mask_soft = cv2.GaussianBlur(mask_soft, (0, 0), 1.2)
    mask_u8 = np.clip(mask_soft, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, m = cv2.threshold(mask_u8, 96, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(m > 0)
    if not len(xs):
        return None

    if gray is not None:
        gray[m == 0] = 0

    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out_gray = None if gray is None else gray[y1:y2, x1:x2].copy()
    out_soft = (mask_soft[y1:y2, x1:x2] / 255.0).astype(np.float32)
    out_hard = (m[y1:y2, x1:x2] > 0)

    out_h, out_w = out_hard.shape[:2]

    return {
        "gray": out_gray,
        "soft_mask": out_soft,
        "hard_mask": out_hard,
        "h": out_h,
        "w": out_w,
        "item_type": item.get("item_type", "shampoo"),
    }


def build_transform_candidates(item, rng, scale_min, scale_max, rot_min, rot_max, num_candidates):
    candidates = []
    for _ in range(num_candidates):
        t = transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max)
        if t is not None:
            candidates.append(t)
    return candidates


# =============================================================================
# Placement helpers
# =============================================================================

def build_candidate_xs(x_min, x_max, x_base, step, rng):
    candidate_xs = list(range(x_min, x_max + 1, step))
    x_base = max(x_min, min(x_max, x_base))
    candidate_xs.append(x_base)
    candidate_xs = sorted(set(candidate_xs), key=lambda xx: abs(xx - x_base))

    chunks = [candidate_xs[i:i + 8] for i in range(0, len(candidate_xs), 8)]
    for chunk in chunks:
        rng.shuffle(chunk)
    return [x for chunk in chunks for x in chunk]


def build_soft_y_candidates(canvas_h, obj_h, y_base, radius, step, rng):
    if canvas_h <= obj_h:
        return [0]

    y_min = 0
    y_max = canvas_h - obj_h
    y_base = max(y_min, min(y_max, int(y_base)))
    radius = max(step, int(radius))
    step = max(1, int(step))

    ys = []
    lo = max(y_min, y_base - radius)
    hi = min(y_max, y_base + radius)

    for y in range(lo, hi + 1, step):
        ys.append(y)

    ys.append(y_base)

    ys = sorted(set(ys), key=lambda yy: abs(yy - y_base))
    return ys

def adjust_y_base_into_valid_band(tray_place: np.ndarray, obj_h: int, y_base: int) -> int:
    H = tray_place.shape[0]
    y_base = max(0, min(H - obj_h, int(y_base)))

    valid_ys = []
    for y in range(0, H - obj_h + 1):
        roi = tray_place[y:y + obj_h, :]
        if roi.shape[0] == obj_h and roi.any():
            valid_ys.append(y)

    if not valid_ys:
        return y_base

    return min(valid_ys, key=lambda y: abs(y - y_base))

def apply_vertical_inner_margin(mask_bool: np.ndarray, margin_px: int) -> np.ndarray:
    m = (mask_bool > 0)
    if margin_px <= 0:
        return m.copy()

    xs = np.where(m.any(axis=0))[0]
    if len(xs) == 0:
        return m.copy()

    out = np.zeros_like(m, dtype=bool)
    for x in xs:
        ys = np.where(m[:, x])[0]
        if len(ys) == 0:
            continue
        y0 = int(ys.min()) + int(margin_px)
        y1 = int(ys.max()) - int(margin_px)
        if y1 >= y0:
            out[y0:y1 + 1, x] = True
    return out


def apply_horizontal_inner_margin(mask_bool: np.ndarray, margin_px: int) -> np.ndarray:
    """
    Shrink the valid placement region only in the horizontal direction so
    the object stays away from the tray's left/right border.

    This works per row instead of using one global bounding box, so it
    respects trays whose widths vary with y.
    """
    m = (mask_bool > 0)
    if margin_px <= 0:
        return m.copy()

    ys = np.where(m.any(axis=1))[0]
    if len(ys) == 0:
        return m.copy()

    out = np.zeros_like(m, dtype=bool)
    for y in ys:
        xs = np.where(m[y])[0]
        if len(xs) == 0:
            continue

        x0 = int(xs.min()) + int(margin_px)
        x1 = int(xs.max()) - int(margin_px)

        if x1 >= x0:
            out[y, x0:x1 + 1] = True

    return out


# =============================================================================
# Blade pseudo X-ray helper
# =============================================================================

def render_blade_on_tray(region_B: np.ndarray, obj_soft: np.ndarray, obj_mask: np.ndarray) -> np.ndarray:
    tray01 = np.clip(region_B / 255.0, 1e-4, 1.0)
    tray_abs = -np.log(tray01)

    dt = cv2.distanceTransform(obj_mask.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    if dt.max() > 0:
        dt = dt / (dt.max() + 1e-6)

    blade_abs = 0.90 + 0.65 * dt
    target_abs = tray_abs + blade_abs * obj_soft
    blended = np.exp(-target_abs)
    blended = blended - (0.02 + 0.04 * dt) * obj_soft
    blended = np.clip(blended, 0.0, 1.0)
    return blended * 255.0


# =============================================================================
# Scene builder — MATCHES STAGE13 SEMANTICS
# =============================================================================

def build_scene(
    rng,
    tray_masks,
    cutouts,
    scale_min,
    scale_max,
    rot_min,
    rot_max,
    horizontal_shift_only=True,
    max_horizontal_shift=150,
    max_vertical_shift=0,
    no_overlap=True,
    x_search_step=X_SEARCH_STEP,
    max_transform_candidates=MAX_TRANSFORM_CANDIDATES,
    tray_horizontal_margin_px=0,
    tray_vertical_margin_px=0,
    canvas_h=1024,
    canvas_w=1024,
):
    tray = rng.choice(tray_masks)
    tray_place = tray.copy()
    tray_place = apply_horizontal_inner_margin(tray_place, tray_horizontal_margin_px)
    tray_place = apply_vertical_inner_margin(tray_place, tray_vertical_margin_px)

    if not tray_place.any():
        tray_place = tray.copy()

    tray_rows = np.where(tray_place.any(axis=1))[0]
    tray_y_min = int(tray_rows.min()) if len(tray_rows) else 0
    tray_y_max = int(tray_rows.max()) if len(tray_rows) else (canvas_h - 1)

    # A side
    canvas_mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask[tray] = PALETTE_BGR[2]  # tray-only = blue in RGB after write

    # pseudo_B side for debugging only; model does not use appearance channel here
    tray_preview = np.full((canvas_h, canvas_w), 235, dtype=np.float32)
    tray_preview[~tray] = 0
    canvas_app = tray_preview.copy()

    occ = np.zeros((canvas_h, canvas_w), dtype=bool)

    placed_count = 0
    for idx, item in enumerate(cutouts):
        placed = False
        transformed_candidates = build_transform_candidates(
            item=item,
            rng=rng,
            scale_min=scale_min,
            scale_max=scale_max,
            rot_min=rot_min,
            rot_max=rot_max,
            num_candidates=max(1, max_transform_candidates),
        )

        for t in transformed_candidates:
            h, w = t["h"], t["w"]
            if h >= canvas_h or w >= canvas_w or h < 2 or w < 2:
                continue

            obj_mask = t["hard_mask"]
            obj_soft = t["soft_mask"]
            obj_gray = None if t.get("gray", None) is None else t["gray"].astype(np.float32)
            item_type = item.get("item_type", "shampoo")

            y_base = int(round(float(item.get("fit_cy", item["orig_cy"])) - h / 2.0))
            y_base = max(0, min(canvas_h - h, y_base))

            # Clamp the preferred y into the tray's valid vertical band first.
            # This makes tray_vertical_margin_px behave smoothly instead of doing
            # nothing until a sudden large jump.
            valid_y_min = max(0, tray_y_min)
            valid_y_max = min(canvas_h - h, tray_y_max - h + 1)
            if valid_y_max >= valid_y_min:
                y_base = min(max(y_base, valid_y_min), valid_y_max)

            if horizontal_shift_only:
                y_candidates = build_soft_y_candidates(
                    canvas_h=canvas_h,
                    obj_h=h,
                    y_base=y_base,
                    radius=max(0, int(max_vertical_shift)),
                    step=1,
                    rng=rng,
                )
            else:
                y_candidates = build_soft_y_candidates(
                    canvas_h=canvas_h,
                    obj_h=h,
                    y_base=y_base,
                    radius=max(0, int(max_vertical_shift)) if max_vertical_shift > 0 else max(1, int(x_search_step)),
                    step=1,
                    rng=rng,
                )

            x_base_scaled = int(round(float(item.get("fit_x0", item["orig_x0"]))))
            x_base_scaled = max(0, min(canvas_w - w, x_base_scaled))
            x_min = max(0, x_base_scaled - max_horizontal_shift)
            x_max = min(canvas_w - w, x_base_scaled + max_horizontal_shift)
            if x_min > x_max:
                continue

            candidate_xs = build_candidate_xs(x_min, x_max, x_base_scaled, max(1, x_search_step), rng)

            for y in y_candidates:
                for x in candidate_xs:
                    roi_tray = tray_place[y:y + h, x:x + w]
                    if roi_tray.shape[:2] != obj_mask.shape or not roi_tray.any():
                        continue
                    if not np.all(roi_tray[obj_mask]):
                        continue

                    roi_occ = occ[y:y + h, x:x + w]
                    if no_overlap and np.any(roi_occ & obj_mask):
                        continue

                    region_A = canvas_mask[y:y + h, x:x + w]

                    # pseudo_B preview only
                    region_B = canvas_app[y:y + h, x:x + w].astype(np.float32)

                    if int(item.get("train_id", 1)) == 3:
                        region_A[obj_mask] = OVERLAP_BLADE_BGR
                        canvas_app[y:y + h, x:x + w] = render_blade_on_tray(region_B, obj_soft, obj_mask)
                    else:
                        # Stage15/16 shampoo-over-tray overlap
                        region_A[obj_mask] = OVERLAP_BGR

                        obj_gray = t["gray"].astype(np.float32)
                        tray01 = np.clip(region_B / 255.0, 1e-4, 1.0)
                        obj01 = np.clip(obj_gray / 255.0, 1e-4, 1.0)

                        tray_abs = -np.log(tray01)
                        obj_abs = -np.log(obj01)

                        scale_est = max(h / max(1, item["orig_h"]), w / max(1, item["orig_w"]))
                        atten_scale = float(scale_est ** 0.7)

                        target_abs = tray_abs + obj_abs * atten_scale
                        blended_abs = tray_abs * (1.0 - obj_soft) + target_abs * obj_soft
                        blended = np.exp(-blended_abs)

                        obj_norm = (obj_gray - obj_gray.min()) / (obj_gray.max() - obj_gray.min() + 1e-6)
                        blended = blended - (obj_norm * 0.03 * obj_soft)
                        blended = np.clip(blended, 0.0, 1.0)

                        canvas_app[y:y + h, x:x + w] = blended * 255.0

                    occ[y:y + h, x:x + w][obj_mask] = True
                    placed = True
                    placed_count += 1
                    print(f"[place] item {idx + 1}/{len(cutouts)} placed at x={x}, y={y}, w={w}, h={h}")
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            raise RuntimeError(
                f"Failed to place item {idx + 1}/{len(cutouts)} | "
                f"file={item.get('file_name')} orig_x0={item.get('orig_x0')} "
                f"orig_y0={item.get('orig_y0')} orig_w={item.get('orig_w')} orig_h={item.get('orig_h')}"
            )

    pseudo_B_bgr = cv2.cvtColor(normalize_xray_intensity(canvas_app), cv2.COLOR_GRAY2BGR)
    return canvas_mask, canvas_app.astype(np.uint8), pseudo_B_bgr, placed_count, tray


# =============================================================================
# Test image writing
# =============================================================================

def clean_test_dir(test_dir: Path):
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.glob("*.png"):
        p.unlink()


def cleanup_intermediate_outputs(out_root: Path, results_root: Path, keep_preview=False):
    test_dir = out_root / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)

    images_dir = results_root / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir, ignore_errors=True)

    for html_dir in [results_root, results_root.parent]:
        if html_dir.exists():
            for p in html_dir.glob("*.html"):
                try:
                    p.unlink()
                except OSError:
                    pass

    if not keep_preview:
        preview_dir = out_root / "preview"
        if preview_dir.exists():
            shutil.rmtree(preview_dir, ignore_errors=True)


def write_single_test_image(
    test_dir: Path,
    canvas_mask: np.ndarray,
    pseudo_B_bgr: np.ndarray,
    tray_mask_bool: np.ndarray = None,
    stem: str = "scene_0000",
):
    clean_test_dir(test_dir)

    if canvas_mask.shape[:2] != pseudo_B_bgr.shape[:2]:
        raise RuntimeError(
            f"canvas_mask and pseudo_B must have same size. "
            f"Got mask={canvas_mask.shape[:2]} pseudo_B={pseudo_B_bgr.shape[:2]}"
        )

    ab = np.concatenate([canvas_mask, pseudo_B_bgr], axis=1)
    out_path = test_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), ab)

    if tray_mask_bool is not None:
        T_path = test_dir / f"{stem}_T.png"
        cv2.imwrite(str(T_path), (tray_mask_bool.astype(np.uint8) * 255))
        print(f"[T] wrote tray mask preview: {T_path.name}")

    print(f"[single] wrote test image: {out_path.name}")
    return stem


def resolve_fake_image_path(images_dir: Path, stem: str):
    direct = images_dir / f"{stem}_fake_B.png"
    if direct.exists():
        return direct
    candidates = sorted(images_dir.glob(f"{stem}*_fake_B.png"))
    if candidates:
        return candidates[0]
    return None


def export_smooth_2x(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 0.6)
    return up


# =============================================================================
# Pix2pix inference
# =============================================================================

def run_pix2pix_test(
    temp_dataset_dir,
    epoch="latest",
    num_test=None,
    tray_mask_dir="",
    use_tray_mask=True,
):
    cfg = TRAIN_CFG

    cmd = [
        "python", str(PIX2PIX_DIR / "test.py"),
        f"--dataroot={temp_dataset_dir}",
        f"--name={MODEL_NAME}",
        "--model=pix2pix",
        "--dataset_mode=aligned",
        "--direction=AtoB",
        f"--input_nc={cfg['input_nc']}",
        "--output_nc=3",
        "--netG=unet_256",
        f"--norm={cfg['norm']}",
        "--preprocess=none",
        "--load_size=0",
        "--crop_size=0",
        "--no_flip",
        f"--epoch={epoch}",
        "--eval",
        "--class_nc=3",
        "--thickness_nc=1",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--return_instance_masks",
        "--mask_thr=0.05",
        f"--canvas_h={cfg['canvas_h']}",
        f"--canvas_w={cfg['canvas_w']}",
        "--pad_to_canvas",
    ]

    if cfg.get("use_appearance_channel", False):
        cmd += [
            "--use_appearance_channel",
            "--appearance_nc=1",
        ]

    if use_tray_mask:
        cmd += [
            "--use_tray_mask",
            f"--tray_mask_thr={cfg['tray_mask_thr']}",
            f"--tray_cc_close_px={cfg['tray_cc_close_px']}",
            f"--tray_mask_dilate_px={cfg['tray_mask_dilate_px']}",
        ]
        if tray_mask_dir:
            cmd.append(f"--tray_mask_dir={tray_mask_dir}")

    if num_test is not None:
        cmd.append(f"--num_test={num_test}")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    return Path("results") / MODEL_NAME / f"test_{epoch}"


# =============================================================================
# Tray-only generation
# =============================================================================

def choose_tray_mask_path(tray_mask_dir: Path, seed: int):
    paths = sorted(tray_mask_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No tray mask PNGs in {tray_mask_dir}")
    rng = random.Random(seed)
    return rng.choice(paths)


def build_tray_scene_from_mask(mask_path: Path, out_w: int, out_h: int, thr=0.5, invert=False, close_px=2, dilate_px=0):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"Could not read tray mask: {mask_path}")

    if m.shape != (out_h, out_w):
        m = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    tray_bool = preprocess_tray_mask(m, thr=thr, invert=invert, close_px=close_px, dilate_px=dilate_px)
    canvas_mask = build_tray_condition_from_mask(tray_bool)
    pseudo_B_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    return canvas_mask, pseudo_B_bgr, tray_bool


# =============================================================================
# Inference quality metrics
# =============================================================================

def semantic_masks_from_canvas_mask(canvas_mask: np.ndarray):
    shampoo_only = np.all(canvas_mask == np.array(PALETTE_BGR[1], dtype=np.uint8), axis=2)
    tray_only = np.all(canvas_mask == np.array(PALETTE_BGR[2], dtype=np.uint8), axis=2)
    blade_only = np.all(canvas_mask == np.array(PALETTE_BGR[3], dtype=np.uint8), axis=2)
    overlap_shampoo = np.all(canvas_mask == np.array(OVERLAP_BGR, dtype=np.uint8), axis=2)
    overlap_blade = np.all(canvas_mask == np.array(OVERLAP_BLADE_BGR, dtype=np.uint8), axis=2)

    object_mask = shampoo_only | blade_only | overlap_shampoo | overlap_blade
    tray_mask = tray_only | overlap_shampoo | overlap_blade
    overlap_mask = overlap_shampoo | overlap_blade
    return object_mask.astype(bool), tray_mask.astype(bool), overlap_mask.astype(bool)


def _score_from_distance(value: float, target: float, tolerance: float) -> float:
    tolerance = max(float(tolerance), 1e-6)
    return float(max(0.0, 1.0 - abs(float(value) - float(target)) / tolerance))


def _to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.clip(gray, 0.0, 1.0)

def _resize_bool_mask(mask: np.ndarray, target_hw):
    target_h, target_w = target_hw
    if mask is None:
        return None

    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask

    if mask_u8.shape[:2] == (target_h, target_w):
        return (mask_u8 > 0)

    mask_rs = cv2.resize(mask_u8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (mask_rs > 0)


def _safe_ref(ref: dict, key: str, default=None):
    if not isinstance(ref, dict):
        return default
    return ref.get(key, default)


def load_training_score_reference(path_str: str):
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        print(f"[warn] training score reference not found: {p}")
        return None
    try:
        ref = json.loads(p.read_text())
        print(f"[metric] loaded training score reference: {p}")
        return ref
    except Exception as e:
        print(f"[warn] could not parse training score reference {p}: {e}")
        return None


def _gaussian_match_score(value: float, mean: float, std: float, floor: float = 1e-3) -> float:
    std = max(float(std), floor)
    z = abs(float(value) - float(mean)) / std
    return float(np.exp(-0.5 * z * z))


def _compute_stats_like_score(metrics: dict, training_ref: dict = None) -> float:
    if not training_ref:
        return float(np.clip(
            0.45 * metrics.get("score_exposure", 0.0) +
            0.35 * metrics.get("score_contrast", 0.0) +
            0.20 * metrics.get("score_nonblank", 0.0),
            0.0, 1.0
        ))

    refs = training_ref.get("real_train_reference", training_ref)
    matches = []

    if "global_mean" in metrics and _safe_ref(refs, "global_mean_mean") is not None:
        matches.append(_gaussian_match_score(metrics["global_mean"], refs["global_mean_mean"], _safe_ref(refs, "global_mean_std", 0.05)))
    if "global_std" in metrics and _safe_ref(refs, "global_std_mean") is not None:
        matches.append(_gaussian_match_score(metrics["global_std"], refs["global_std_mean"], _safe_ref(refs, "global_std_std", 0.05)))
    if "object_mean" in metrics and _safe_ref(refs, "object_mean_mean") is not None:
        matches.append(_gaussian_match_score(metrics["object_mean"], refs["object_mean_mean"], _safe_ref(refs, "object_mean_std", 0.05)))
    if "tray_mean" in metrics and _safe_ref(refs, "tray_mean_mean") is not None:
        matches.append(_gaussian_match_score(metrics["tray_mean"], refs["tray_mean_mean"], _safe_ref(refs, "tray_mean_std", 0.05)))

    if matches:
        return float(np.clip(np.mean(matches), 0.0, 1.0))

    return float(np.clip(
        0.45 * metrics.get("score_exposure", 0.0) +
        0.35 * metrics.get("score_contrast", 0.0) +
        0.20 * metrics.get("score_nonblank", 0.0),
        0.0, 1.0
    ))


def _compute_train_aligned_proxy_scores(metrics: dict, generate_mode: str, training_ref: dict = None):
    # GAN-like proxy: broad realism / plausibility
    if generate_mode == "tray":
        proxy_gan = float(np.clip(
            0.35 * metrics.get("score_contrast", 0.0) +
            0.35 * metrics.get("score_sharpness", 0.0) +
            0.30 * metrics.get("score_nonblank", 0.0),
            0.0, 1.0
        ))
        proxy_l1 = float(np.clip(
            0.40 * metrics.get("score_exposure", 0.0) +
            0.30 * metrics.get("score_contrast", 0.0) +
            0.30 * metrics.get("score_nonblank", 0.0),
            0.0, 1.0
        ))
        proxy_grad = float(np.clip(metrics.get("score_sharpness", 0.0), 0.0, 1.0))
        proxy_lap = float(np.clip(
            0.70 * metrics.get("score_sharpness", 0.0) +
            0.30 * metrics.get("score_contrast", 0.0),
            0.0, 1.0
        ))
        proxy_ssim = float(np.clip(
            0.45 * metrics.get("score_nonblank", 0.0) +
            0.30 * metrics.get("score_contrast", 0.0) +
            0.25 * metrics.get("tray_contrast", 0.0),
            0.0, 1.0
        ))
    else:
        proxy_gan = float(np.clip(
            0.25 * metrics.get("score_contrast", 0.0) +
            0.25 * metrics.get("score_sharpness", 0.0) +
            0.20 * metrics.get("score_obj_contrast", 0.0) +
            0.15 * metrics.get("score_inside", 0.0) +
            0.15 * metrics.get("score_nonblank", 0.0),
            0.0, 1.0
        ))
        proxy_l1 = float(np.clip(
            0.35 * metrics.get("score_exposure", 0.0) +
            0.25 * metrics.get("score_contrast", 0.0) +
            0.20 * metrics.get("score_obj_contrast", 0.0) +
            0.20 * metrics.get("score_nonblank", 0.0),
            0.0, 1.0
        ))
        proxy_grad = float(np.clip(metrics.get("score_sharpness", 0.0), 0.0, 1.0))
        proxy_lap = float(np.clip(
            0.70 * metrics.get("score_sharpness", 0.0) +
            0.30 * metrics.get("score_contrast", 0.0),
            0.0, 1.0
        ))
        proxy_ssim = float(np.clip(
            0.40 * metrics.get("score_inside", 0.0) +
            0.20 * metrics.get("score_overlap", 0.0) +
            0.20 * metrics.get("score_area", 0.0) +
            0.20 * metrics.get("score_obj_contrast", 0.0),
            0.0, 1.0
        ))

    proxy_stats = _compute_stats_like_score(metrics, training_ref=training_ref)

    return {
        "gan": proxy_gan,
        "l1": proxy_l1,
        "grad": proxy_grad,
        "lap": proxy_lap,
        "ssim": proxy_ssim,
        "stats": proxy_stats,
    }


def compute_inference_quality(
    final_img_bgr: np.ndarray,
    canvas_mask: np.ndarray,
    tray_mask_bool: np.ndarray = None,
    generate_mode: str = "shampoo",
    training_ref: dict = None,
):
    gray01 = _to_gray01(final_img_bgr)
    H, W = gray01.shape[:2]

    # Always align semantic canvas mask to final generated image size
    if canvas_mask.shape[:2] != (H, W):
        canvas_mask = cv2.resize(canvas_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    shampoo_mask, tray_mask_from_canvas, overlap_mask = semantic_masks_from_canvas_mask(canvas_mask)

    # Force every mask to same size as final image
    shampoo_mask = _resize_bool_mask(shampoo_mask, (H, W))
    tray_mask_from_canvas = _resize_bool_mask(tray_mask_from_canvas, (H, W))
    overlap_mask = _resize_bool_mask(overlap_mask, (H, W))

    if tray_mask_bool is not None:
        tray_mask = _resize_bool_mask(tray_mask_bool, (H, W))
    else:
        tray_mask = tray_mask_from_canvas

    obj_mask = shampoo_mask
    img_nonzero = gray01 > (8.0 / 255.0)

    metrics = {}

    global_mean = float(gray01.mean())
    global_std = float(gray01.std())
    lap_var = float(cv2.Laplacian((gray01 * 255.0).astype(np.uint8), cv2.CV_32F).var())
    nonblank_ratio = float(img_nonzero.mean())

    metrics["global_mean"] = global_mean
    metrics["global_std"] = global_std
    metrics["laplacian_var"] = lap_var
    metrics["nonblank_ratio"] = nonblank_ratio

    # Keep the old low-level image checks because they are useful building blocks,
    # but convert them into training-aligned proxy scores below.
    score_nonblank = _score_from_distance(
        nonblank_ratio,
        0.18 if generate_mode == "shampoo" else 0.10,
        0.18,
    )
    score_contrast = _score_from_distance(global_std, 0.22, 0.18)
    score_exposure = _score_from_distance(global_mean, 0.30, 0.22)
    score_sharpness = min(1.0, lap_var / 250.0)

    metrics["score_nonblank"] = score_nonblank
    metrics["score_contrast"] = score_contrast
    metrics["score_exposure"] = score_exposure
    metrics["score_sharpness"] = score_sharpness

    if generate_mode == "tray":
        tray_fill = float(gray01[tray_mask].mean()) if tray_mask.any() else 0.0
        tray_bg = float(gray01[~tray_mask].mean()) if (~tray_mask).any() else 0.0
        tray_contrast = max(0.0, min(1.0, abs(tray_fill - tray_bg) / 0.35))

        metrics["tray_fill_mean"] = tray_fill
        metrics["tray_bg_mean"] = tray_bg
        metrics["tray_contrast"] = tray_contrast
    else:
        obj_area_ratio = float(obj_mask.mean()) if obj_mask.size else 0.0
        obj_mean = float(gray01[obj_mask].mean()) if obj_mask.any() else 0.0
        tray_mean = float(gray01[tray_mask].mean()) if tray_mask.any() else 0.0
        bg_mean = float(gray01[~tray_mask].mean()) if (~tray_mask).any() else 0.0
        inside_tray_ratio = float((obj_mask & tray_mask).sum() / max(1, obj_mask.sum()))
        overlap_ratio = float(overlap_mask.sum() / max(1, obj_mask.sum()))

        score_area = _score_from_distance(obj_area_ratio, 0.05, 0.05)
        score_inside = inside_tray_ratio
        score_overlap = _score_from_distance(overlap_ratio, 1.0, 0.25)
        score_obj_contrast = max(0.0, min(1.0, abs(obj_mean - tray_mean) / 0.25))

        metrics["object_area_ratio"] = obj_area_ratio
        metrics["object_mean"] = obj_mean
        metrics["tray_mean"] = tray_mean
        metrics["bg_mean"] = bg_mean
        metrics["inside_tray_ratio"] = inside_tray_ratio
        metrics["overlap_ratio"] = overlap_ratio
        metrics["score_area"] = score_area
        metrics["score_inside"] = score_inside
        metrics["score_overlap"] = score_overlap
        metrics["score_obj_contrast"] = score_obj_contrast

    # Convert inference heuristics into proxies for the same training-side Q_score parts:
    # GAN, L1, grad, lap, SSIM, stats.
    proxy = _compute_train_aligned_proxy_scores(metrics, generate_mode=generate_mode, training_ref=training_ref)
    metrics["proxy_scores"] = proxy
    metrics["score_version"] = "train_aligned_qscore_proxy_v1"
    metrics["score_weights"] = TRAIN_SCORE_WEIGHTS.copy()

    q = (
        TRAIN_SCORE_WEIGHTS["gan"] * proxy["gan"] +
        TRAIN_SCORE_WEIGHTS["l1"] * proxy["l1"] +
        TRAIN_SCORE_WEIGHTS["grad"] * proxy["grad"] +
        TRAIN_SCORE_WEIGHTS["lap"] * proxy["lap"] +
        TRAIN_SCORE_WEIGHTS["ssim"] * proxy["ssim"] +
        TRAIN_SCORE_WEIGHTS["stats"] * proxy["stats"]
    )

    metrics["quality_score"] = float(np.clip(q, 0.0, 1.0) * 100.0)
    metrics["quality_label"] = (
        "GOOD" if metrics["quality_score"] >= 75 else
        "OKAY" if metrics["quality_score"] >= 60 else
        "WEAK"
    )

    # Keep a readable breakdown that mirrors training metric categories.
    metrics["contributors"] = [
        ("GAN-like", TRAIN_SCORE_WEIGHTS["gan"], proxy["gan"]),
        ("L1-like", TRAIN_SCORE_WEIGHTS["l1"], proxy["l1"]),
        ("Grad-like", TRAIN_SCORE_WEIGHTS["grad"], proxy["grad"]),
        ("Lap-like", TRAIN_SCORE_WEIGHTS["lap"], proxy["lap"]),
        ("SSIM-like", TRAIN_SCORE_WEIGHTS["ssim"], proxy["ssim"]),
        ("Stats-like", TRAIN_SCORE_WEIGHTS["stats"], proxy["stats"]),
    ]

    return metrics




def _fit_lines_to_width(lines, font, target_w, font_scale, thickness, pad):
    out = []
    max_w = max(40, target_w - 2 * pad)
    for line in lines:
        words = line.split()
        if not words:
            out.append("")
            continue
        cur = words[0]
        for word in words[1:]:
            test = cur + " " + word
            tw = cv2.getTextSize(test, font, font_scale, thickness)[0][0]
            if tw <= max_w:
                cur = test
            else:
                out.append(cur)
                cur = word
        out.append(cur)
    return out




def strip_existing_top_banner(img_bgr: np.ndarray) -> np.ndarray:
    """
    Remove an existing dark metrics banner already baked into fake_B output.
    This avoids stacking training-banner + inference-banner together.
    """
    if img_bgr is None or img_bgr.ndim != 3:
        return img_bgr

    h, w = img_bgr.shape[:2]
    if h < 80:
        return img_bgr

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    row_mean = gray.mean(axis=1)

    max_scan = min(h // 3, 220)
    cut = 0
    started = False
    for y in range(max_scan):
        if row_mean[y] < 45:
            started = True
            cut = y + 1
        else:
            if started and y > 20:
                break

    if cut >= 35:
        return img_bgr[cut:, :].copy()
    return img_bgr

def overlay_metric_panel(img_bgr: np.ndarray, lines, anchor="top_banner") -> np.ndarray:
    if img_bgr is None:
        return None
    base = img_bgr.copy()
    h, w = base.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Use a dedicated banner above the image so text never covers the generated result.
    font_scale = max(0.42, min(0.72, w / 1500.0))
    thickness = max(1, int(round(font_scale * 2)))
    pad_x = max(10, int(round(w * 0.018)))
    pad_y = max(8, int(round(h * 0.012)))

    wrapped = _fit_lines_to_width(lines, font, w, font_scale, thickness, pad_x)
    line_h = max(20, int(round(cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] * 1.7)))
    banner_h = pad_y * 2 + max(1, len(wrapped)) * line_h

    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (18, 18, 18)
    cv2.line(banner, (0, banner_h - 1), (w, banner_h - 1), (70, 70, 70), 1)

    y = pad_y + line_h - 6
    for i, line in enumerate(wrapped):
        color = (0, 255, 255) if i == 0 else (235, 235, 235)
        cv2.putText(banner, line, (pad_x, y), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(banner, line, (pad_x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        y += line_h

    return np.vstack([banner, base])


def build_inference_overlay_lines(metrics: dict):
    line1 = f"Quality {metrics['quality_score']:.1f}/100 | {metrics['quality_label'].upper()}"
    line2 = "Training-aligned proxy score"
    p = metrics.get("proxy_scores", {})
    line3 = (
        f"GAN-like 10%x{p.get('gan', 0.0):.2f} | "
        f"L1-like 35%x{p.get('l1', 0.0):.2f} | "
        f"Grad-like 15%x{p.get('grad', 0.0):.2f} | "
        f"Lap-like 10%x{p.get('lap', 0.0):.2f} | "
        f"SSIM-like 20%x{p.get('ssim', 0.0):.2f} | "
        f"Stats-like 10%x{p.get('stats', 0.0):.2f}"
    )
    return [line1, line2, line3]


def save_quality_metrics(metrics_path: Path, metrics: dict):
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[metric] wrote inference quality metrics to: {metrics_path}")


def print_quality_metrics(metrics: dict):
    p = metrics.get("proxy_scores", {})
    print(
        f"[metric] inference quality = {metrics['quality_score']:.2f}/100 | "
        f"label={metrics['quality_label']} | "
        f"GAN-like={p.get('gan', 0.0):.3f} | "
        f"L1-like={p.get('l1', 0.0):.3f} | "
        f"Grad-like={p.get('grad', 0.0):.3f} | "
        f"Lap-like={p.get('lap', 0.0):.3f} | "
        f"SSIM-like={p.get('ssim', 0.0):.3f} | "
        f"Stats-like={p.get('stats', 0.0):.3f}"
    )
# =============================================================================
# X-ray postprocess / debug helpers
# =============================================================================

def force_grayscale_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Force any generated image back to 3-channel grayscale.
    This removes color speckles / chroma artifacts while preserving structure.
    """
    if img_bgr is None:
        return None
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def canvas_mask_to_debug_gray(canvas_mask: np.ndarray) -> np.ndarray:
    """
    Human-friendly visualization only.
    Does NOT change the actual conditioning image used by pix2pix.
    """
    shampoo_mask = np.all(canvas_mask == np.array(PALETTE_BGR[1], dtype=np.uint8), axis=2)
    tray_mask = np.all(canvas_mask == np.array(PALETTE_BGR[2], dtype=np.uint8), axis=2)
    blade_mask = np.all(canvas_mask == np.array(PALETTE_BGR[3], dtype=np.uint8), axis=2)
    overlap_mask = np.all(canvas_mask == np.array(OVERLAP_BGR, dtype=np.uint8), axis=2)
    overlap_blade_mask = np.all(canvas_mask == np.array(OVERLAP_BLADE_BGR, dtype=np.uint8), axis=2)

    vis = np.zeros(canvas_mask.shape[:2], dtype=np.uint8)
    vis[tray_mask] = 140
    vis[shampoo_mask] = 210
    vis[blade_mask] = 180
    vis[overlap_mask] = 235
    vis[overlap_blade_mask] = 250
    return cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate_mode", choices=["shampoo", "blade", "tray"], required=True)

    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--coco_json", type=str, default="")
    ap.add_argument("--classes", type=str, default="Shampoo")
    ap.add_argument("--count", type=str, default="1")
    ap.add_argument("--blade_mask_dir", type=str, default="")
    ap.add_argument("--blade_pick_mode", choices=["random", "first"], default="random")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dataset", type=str, default="datasets/_gen_stage13_shampoo_tray")
    ap.add_argument("--epoch", type=str, default="latest")

    # Keep args for compatibility, but force actual generation to training size
    ap.add_argument("--canvas_h", type=int, default=1024)
    ap.add_argument("--canvas_w", type=int, default=1024)

    ap.add_argument("--rand_scale_min", type=float, default=0.85)
    ap.add_argument("--rand_scale_max", type=float, default=0.85)
    ap.add_argument("--rand_rot_min", type=float, default=0.0)
    ap.add_argument("--rand_rot_max", type=float, default=40.0)
    ap.add_argument("--rand_max_tries_per_obj", type=int, default=30)

    ap.add_argument("--no_overlap", action="store_true")
    ap.add_argument("--horizontal_shift_only", action="store_true")
    ap.add_argument("--max_horizontal_shift", type=int, default=150)
    ap.add_argument("--x_search_step", type=int, default=12)
    ap.add_argument("--max_transform_candidates", type=int, default=8)
    ap.add_argument("--tray_horizontal_margin_px", type=int, default=0,
                    help="Keep the object this many pixels away from the tray left/right border during placement.")

    ap.add_argument("--tray_mask_path", type=str, default="")
    ap.add_argument("--tray_mask_thr", type=float, default=0.5)
    ap.add_argument("--tray_mask_invert", action="store_true")
    ap.add_argument("--tray_cc_close_px", type=int, default=2)
    ap.add_argument("--tray_mask_dilate_px", type=int, default=0)
    ap.add_argument("--tray_mask_dir", type=str, default="")
    ap.add_argument("--tray_vertical_margin_px", type=int, default=0,
                    help="Extra top/bottom tray padding. Usually keep this at 0 when you only want minor vertical shift.")

    ap.add_argument("--keep_intermediates", action="store_true")
    ap.add_argument("--keep_preview", action="store_true")
    ap.add_argument("--skip_pix2pix", action="store_true")
    ap.add_argument("--max_vertical_shift", type=int, default=0,
                    help="Small vertical placement freedom in pixels around the original y position.")
    ap.add_argument("--training_score_ref_json", type=str, default="",
                    help="Optional JSON exported from training to calibrate inference score to real-image training statistics.")

    args = ap.parse_args()
    training_ref = load_training_score_reference(args.training_score_ref_json)

    if args.horizontal_shift_only:
        print(f"[placement] horizontal_shift_only=True | max_horizontal_shift={args.max_horizontal_shift} | max_vertical_shift={args.max_vertical_shift} | tray_vertical_margin_px={args.tray_vertical_margin_px}")

    # FORCE match training size
    canvas_h = TRAIN_CFG["canvas_h"]
    canvas_w = TRAIN_CFG["canvas_w"]
    if args.canvas_h != canvas_h or args.canvas_w != canvas_w:
        print(f"[warn] forcing canvas from ({args.canvas_h},{args.canvas_w}) to training size ({canvas_h},{canvas_w})")

    rng = random.Random(args.seed)
    out_root = Path(args.out_dataset)
    tray_mask_bool = None

    if args.generate_mode in {"shampoo", "blade"} and not args.tray_mask_dir:
        raise SystemExit("--tray_mask_dir is required for shampoo/blade mode")

    if args.generate_mode == "shampoo":
        if not args.images_dir or not args.coco_json:
            raise SystemExit("--images_dir and --coco_json are required for shampoo mode")

        tray_masks, _ = load_tray_masks(
            Path(args.tray_mask_dir),
            out_h=canvas_h,
            out_w=canvas_w,
            thr=args.tray_mask_thr,
            invert=args.tray_mask_invert,
            close_px=args.tray_cc_close_px,
            dilate_px=args.tray_mask_dilate_px,
        )

        images_dir = Path(args.images_dir)
        coco = json.loads(Path(args.coco_json).read_text())
        want_classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]

        real_cutouts = build_real_cutout_library(coco, images_dir)

        if want_classes:
            targets = ([int(counts_raw[0])] * len(want_classes)
                       if len(counts_raw) == 1 else [int(x) for x in counts_raw])
            selected = []
            for cls, n in zip(want_classes, targets):
                cands = [c for c in real_cutouts if c["class_name"].lower() == cls.lower()]
                if not cands:
                    avail = sorted({c["class_name"] for c in real_cutouts})
                    raise SystemExit(f"Class '{cls}' not found. Available: {avail}")
                if len(cands) < n:
                    raise SystemExit(f"Class '{cls}' only has {len(cands)} cutouts, but requested {n}")
                selected.extend(rng.sample(cands, n))
        else:
            selected = real_cutouts[: int(counts_raw[0])]

        print("[selected]")
        for i, s in enumerate(selected, 1):
            print(
                f"  item {i}: class={s['class_name']} file={s['file_name']} "
                f"orig_x0={s['orig_x0']} orig_y0={s['orig_y0']} "
                f"orig_w={s['orig_w']} orig_h={s['orig_h']}"
            )

        canvas_mask, canvas_app, pseudo_B_bgr, placed_count, used_tray = build_scene(
            rng=rng,
            tray_masks=tray_masks,
            cutouts=selected,
            scale_min=args.rand_scale_min,
            scale_max=args.rand_scale_max,
            rot_min=args.rand_rot_min,
            rot_max=args.rand_rot_max,
            horizontal_shift_only=args.horizontal_shift_only,
            max_horizontal_shift=args.max_horizontal_shift,
            max_vertical_shift=max(0, int(args.max_vertical_shift)),
            no_overlap=args.no_overlap,
            x_search_step=max(1, int(args.x_search_step)),
            max_transform_candidates=max(1, int(args.max_transform_candidates)),
            tray_horizontal_margin_px=max(0, int(args.tray_horizontal_margin_px)),
            tray_vertical_margin_px=max(0, int(args.tray_vertical_margin_px)),
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )

        tray_mask_bool = used_tray
        print(f"[summary] shampoo mode | requested {len(selected)} items, placed {placed_count}")

    elif args.generate_mode == "blade":
        if not args.blade_mask_dir:
            raise SystemExit("--blade_mask_dir is required for blade mode")

        tray_masks, _ = load_tray_masks(
            Path(args.tray_mask_dir),
            out_h=canvas_h,
            out_w=canvas_w,
            thr=args.tray_mask_thr,
            invert=args.tray_mask_invert,
            close_px=args.tray_cc_close_px,
            dilate_px=args.tray_mask_dilate_px,
        )

        blade_cutouts = build_blade_mask_library(Path(args.blade_mask_dir))
        if args.blade_pick_mode == "first":
            selected = [blade_cutouts[0]]
        else:
            selected = [rng.choice(blade_cutouts)]

        print("[selected]")
        for i, s in enumerate(selected, 1):
            print(
                f"  item {i}: class={s['class_name']} file={s['file_name']} "
                f"orig_x0={s['orig_x0']} orig_y0={s['orig_y0']} "
                f"orig_w={s['orig_w']} orig_h={s['orig_h']}"
            )

        canvas_mask, canvas_app, pseudo_B_bgr, placed_count, used_tray = build_scene(
            rng=rng,
            tray_masks=tray_masks,
            cutouts=selected,
            scale_min=args.rand_scale_min,
            scale_max=args.rand_scale_max,
            rot_min=args.rand_rot_min,
            rot_max=args.rand_rot_max,
            horizontal_shift_only=args.horizontal_shift_only,
            max_horizontal_shift=args.max_horizontal_shift,
            max_vertical_shift=max(0, int(args.max_vertical_shift)),
            no_overlap=args.no_overlap,
            x_search_step=max(1, int(args.x_search_step)),
            max_transform_candidates=max(1, int(args.max_transform_candidates)),
            tray_horizontal_margin_px=max(0, int(args.tray_horizontal_margin_px)),
            tray_vertical_margin_px=max(0, int(args.tray_vertical_margin_px)),
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )

        tray_mask_bool = used_tray
        print(f"[summary] blade mode | requested {len(selected)} items, placed {placed_count}")

    else:
        if args.tray_mask_path:
            tray_mask_path = Path(args.tray_mask_path)
        else:
            if not args.tray_mask_dir:
                raise SystemExit("Provide either --tray_mask_path or --tray_mask_dir for tray mode.")
            tray_mask_path = choose_tray_mask_path(Path(args.tray_mask_dir), args.seed)

        print(f"[tray] using mask: {tray_mask_path}")
        canvas_mask, pseudo_B_bgr, tray_mask_bool = build_tray_scene_from_mask(
            tray_mask_path,
            out_w=canvas_w,
            out_h=canvas_h,
            thr=args.tray_mask_thr,
            invert=args.tray_mask_invert,
            close_px=args.tray_cc_close_px,
            dilate_px=args.tray_mask_dilate_px,
        )
        canvas_app = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        print("[summary] tray mode | generated tray conditioning image from mask")

    if args.keep_preview or args.keep_intermediates:
        preview_dir = out_root / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)

        # raw conditioning actually used for pix2pix
        cv2.imwrite(str(preview_dir / f"mask_seed{args.seed}.png"), canvas_mask)

        # human-friendly visualization so the label colors do not confuse debugging
        cv2.imwrite(
            str(preview_dir / f"mask_seed{args.seed}_debug_gray.png"),
            canvas_mask_to_debug_gray(canvas_mask)
        )

        cv2.imwrite(str(preview_dir / f"pseudo_B_seed{args.seed}.png"), pseudo_B_bgr)
        cv2.imwrite(str(preview_dir / f"canvas_app_seed{args.seed}.png"), canvas_app)
        if tray_mask_bool is not None:
            cv2.imwrite(str(preview_dir / f"tray_seed{args.seed}.png"), tray_mask_bool.astype(np.uint8) * 255)

    test_dir = out_root / "test"
    stem = write_single_test_image(
        test_dir,
        canvas_mask,
        pseudo_B_bgr,
        tray_mask_bool=tray_mask_bool,
    )

    if args.skip_pix2pix:
        print("[summary] skip_pix2pix enabled: saving pseudo_B directly for debugging.")
        raw_final_img = pseudo_B_bgr.copy()
        final_img = export_smooth_2x(raw_final_img.copy())
        results_root = None
    else:
        results_root = run_pix2pix_test(
            temp_dataset_dir=out_root,
            epoch=args.epoch,
            num_test=1,
            tray_mask_dir=str(test_dir) if (tray_mask_bool is not None) else "",
            use_tray_mask=(tray_mask_bool is not None),
        )

        fake_path = resolve_fake_image_path(results_root / "images", stem)
        if fake_path is None:
            raise RuntimeError(f"Could not find fake_B output for stem '{stem}' in {results_root / 'images'}")
        
        raw_final_img = cv2.imread(str(fake_path), cv2.IMREAD_COLOR)
        if raw_final_img is None:
            raise RuntimeError(f"Could not read generated image: {fake_path}")

        # Remove color artifacts from generated X-ray output
        raw_final_img = force_grayscale_bgr(raw_final_img)

        final_img = export_smooth_2x(raw_final_img)

    # Remove any pre-existing training banner baked into the pix2pix output,
    # then compute/display inference-only quality.
    raw_final_img = strip_existing_top_banner(raw_final_img)
    raw_final_img = force_grayscale_bgr(raw_final_img)

    metrics = compute_inference_quality(
        final_img_bgr=raw_final_img,
        canvas_mask=canvas_mask,
        tray_mask_bool=tray_mask_bool,
        generate_mode=args.generate_mode,
        training_ref=training_ref,
    )
    print_quality_metrics(metrics)

    raw_annotated = overlay_metric_panel(raw_final_img, build_inference_overlay_lines(metrics), anchor="top_banner")
    final_img = export_smooth_2x(raw_annotated)

    out_dir = out_root / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.generate_mode
    out_path = out_dir / f"generated_{suffix}_seed{args.seed}_smooth2x.png"
    metrics_path = out_dir / f"generated_{suffix}_seed{args.seed}_metrics.json"
    cv2.imwrite(str(out_path), final_img)
    save_quality_metrics(metrics_path, metrics)
    print(f"Saved generated result to: {out_path}")

    if (not args.keep_intermediates) and (results_root is not None):
        cleanup_intermediate_outputs(out_root, results_root, keep_preview=args.keep_preview)

    print(f"\nDone. Final generated result: {out_path}")
    print(f"Done. Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
"""W

python notebooks/StyleGan/generate_pix2pix.py \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --classes Blade,Penknife \
  --count 1,1 \
  --seed 125 \
  --out_dataset datasets/_gen_real \
  --epoch latest 


  
python notebooks/StyleGan/generate_pix2pix.py \
  --images_dir data/raw/Non-Contraband \
  --coco_json data/raw/Non-Contraband/result.json \
  --classes Shampoo,Hairgel \
  --count 1,1 \
  --seed 125 \
  --out_dataset datasets/_gen_real \
  --epoch latest 
  

python notebooks/StyleGan/generate_pix2pix.py \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --mode real_scene \
  --classes Blade,Vape \
  --seed 125 \
  --out_dataset datasets/_gen_real \
  --epoch latest

  
  Paste mode with strict count enforcement (each placed instance must create a new blob):
  python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode paste \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --classes Blade,Vape \
  --count 2,1 \
  --strict_count --no_overlap --morph none \
  --seed 125 \
  --out_dataset datasets/_gen_real \
  --epoch latest

  Hyper-realistic real_scene_count mode (start from real image, enforce counts by drop/add):
 python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --classes Nail \
  --count 1 \
  --morph dilate \
  --morph_k 3 \
  --morph_iter 1 \
  --scale_min 0.90 \
  --scale_max 1.10 \
  --rot_deg 1.0 \
  --seed 1 \
  --out_dataset datasets/_gen_real \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance

python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Non-Contraband \
  --coco_json data/raw/Non-Contraband/result.json \
  --classes Shampoo,Book \
  --count 2,3 \
  --morph none \
  --seed 88 \
  --out_dataset datasets/_gen_real \
  --epoch latest \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance

  

  python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode random_mask \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --seed 1 \
  --out_dataset datasets/_gen_random \
  --epoch latest \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance \
  --no_overlap


  python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode random_mask \
  --images_dir data/raw/Shampoo_Blade \
  --coco_json data/raw/Shampoo_Blade/result.json \
  --seed 1 \
  --out_dataset datasets/_gen_random \
  --epoch latest \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance \
  --no_overlap
  

   python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Shampoo_Blade \
  --coco_json data/raw/Shampoo_Blade/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 1 --norm instance \
  --out_dataset datasets/_gen_real \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance

  python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Shampoo_Blade \
  --coco_json data/raw/Shampoo_Blade/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 1 \
  --out_dataset datasets/_gen_test \
  --epoch latest \
  --norm instance \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty
  
python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Shampoo_Blade \
  --coco_json data/raw/Shampoo_Blade/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 789 \
  --out_dataset datasets/_gen_test \
  --epoch latest \
  --norm instance \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --no_use_display_mapper \
  --mask_blur_ksize 0 \
  --morph none \
  --remove_halo \
  --halo_erode_px 6 \
  --halo_blend_width 10

  python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode random_mask \
  --images_dir data/raw/Shampoo_Blade \
  --coco_json data/raw/Shampoo_Blade/result.json \
  --seed 456 \
  --out_dataset datasets/_gen_test \
  --epoch latest \
  --norm instance \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --cutout_dir data/raw/Shampoo_Blade/Cropped \
  --tray_mask_dir data/interim/GAN/Empty_Tray_mask/Mask \
  --no_overlap \
  --no_use_display_mapper

  python notebooks/Pix2Pix/generate_pix2pix.py \
    --mode random_mask \
    --images_dir data/raw/Shampoo_Blade \
    --coco_json data/raw/Shampoo_Blade/result.json \
    --seed 1 \
    --out_dataset datasets/_gen_test \
    --epoch latest \
    --norm instance \
    --use_delta_comp \
    --empty_dir data/interim/GAN/Empty \
    --tray_mask_dir data/interim/GAN/Empty_Tray_mask/Mask \
    --no_overlap \
    --classes Shampoo,Blade \
    --count 2,1

    GUIDED
python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode random_mask \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --seed 121 \
  --out_dataset datasets/_gen_test \
  --epoch latest \
  --norm instance \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_dir data/interim/GAN/Empty_Tray_mask/Mask \
  --no_overlap \
  --classes Shampoo \
  --count 1 \
  --rand_n_min 1 \
  --rand_n_max 2 \
  --rand_scale_min 0.9 \
  --rand_scale_max 1.1 \
  --rand_rot_min 0 \
  --rand_rot_max 20 \
  --test_appearance_mode real

  NONGUIDED
python notebooks/Pix2Pix/generate_pix2pix.py \
  --mode random_mask \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --seed 124 \
  --out_dataset datasets/_gen_test \
  --epoch latest \
  --norm instance \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_dir data/interim/GAN/Empty_Tray_mask/Mask \
  --no_overlap \
  --classes Shampoo \
  --count 2 \
  --rand_n_min 1 \
  --rand_n_max 2 \
  --rand_scale_min 0.75 \
  --rand_scale_max 1.25 \
  --rand_rot_min 0 \
  --rand_rot_max 45 \
  --test_appearance_mode zero --disable_test_appearance

python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --mode random_mask \
  --classes Shampoo \
  --count 3 \
  --seed 777 \
  --out_dataset datasets/_gen_normal_V8 \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --rand_scale_min 8.5 \
  --rand_scale_max 8.5 \
  --rand_rot_min 0.0 \
  --rand_rot_max 2.0 \
  --rand_max_tries_per_obj 60 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --horizontal_shift_only \
  --max_horizontal_shift 120 \
  --no_overlap

  

  TRAY
  python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --mode random_mask \
  --classes Shampoo \
  --count 3 \
  --seed 669 \
  --out_dataset datasets/_gen_empty_tray \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --rand_scale_min 1.0 \
  --rand_scale_max 1.0 \
  --rand_rot_min -40.0 \
  --rand_rot_max 40.0 \
  --rand_max_tries_per_obj 60 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --horizontal_shift_only \
  --max_horizontal_shift 900 \
  --no_overlap

  

GENERATE SHAMPOO
python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --generate_mode shampoo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 4 \
  --seed 152 \
  --out_dataset datasets/_gen_shampoo_only \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --rand_scale_min 0.85 \
  --rand_scale_max 0.85 \
  --rand_rot_min 0 \
  --rand_rot_max 40 \
  --rand_max_tries_per_obj 30 \
  --horizontal_shift_only \
  --max_horizontal_shift 150 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --no_overlap \
  --ignore_tray \
  --keep_intermediates


GENERATE SHAMPOO & TRAY
python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --generate_mode shampoo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 1230 \
  --out_dataset datasets/_gen_stage16_shampoo_tray \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
  --tray_mask_thr 0.5 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --rand_scale_min 0.95 \
  --rand_scale_max 0.95 \
  --rand_rot_min -5 \
  --rand_rot_max 5 \
  --rand_max_tries_per_obj 60 \
  --horizontal_shift_only \
  --max_horizontal_shift 150 \
  --tray_horizontal_margin_px 45 \
  --tray_vertical_margin_px 0 \
  --max_vertical_shift 8 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --no_overlap \
  --keep_intermediates  


BLADE WITH TRAY:

python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --generate_mode blade \
  --blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY/matched_masks/train/blade \
  --seed 888 \
  --out_dataset datasets/_gen_stage16_blade_tray \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --tray_mask_dir datasets/SHAMPOOWITHTRAY/matched_masks/train/tray \
  --tray_mask_thr 0.5 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --rand_scale_min 0.95 \
  --rand_scale_max 0.95 \
  --rand_rot_min -5 \
  --rand_rot_max 5 \
  --horizontal_shift_only \
  --max_horizontal_shift 150 \
  --tray_horizontal_margin_px 45 \
  --tray_vertical_margin_px 0 \
  --max_vertical_shift 8 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --no_overlap \
  --keep_intermediates

  

  
EMPTY TRAY:
python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --generate_mode tray \
  --tray_mask_dir data/interim/Empty/masks \
  --tray_mask_thr 0 \
  --tray_cc_close_px 2 \
  --tray_mask_dilate_px 0 \
  --seed 15 \
  --out_dataset datasets/_gen_stage10_tray_only \
  --epoch latest \
  --canvas_h 1152 \
  --canvas_w 1584 \
  --keep_intermediates

  To print out classes that you have:
  python - <<'PY'
    import json
    c=json.load(open("data/raw/Non-Contraband/result.json"))
    print(sorted([x["name"] for x in c["categories"]]))
    PY
"""

