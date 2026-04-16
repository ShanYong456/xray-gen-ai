from pathlib import Path
import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime

import cv2
import numpy as np


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

# overlap colors for conditioning visualization
OVERLAP_BGR = (255, 255, 0)         # shampoo + tray
OVERLAP_BLADE_BGR = (255, 0, 255)   # blade + tray

MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage18_BladeMaskSyn"
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

X_SEARCH_STEP = 12
MAX_TRANSFORM_CANDIDATES = 8


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
        return m01.astype(np.uint8)
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
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

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
                "item_type": "shampoo",
            })

    if not lib:
        raise SystemExit("No valid cutouts from COCO + images_dir")

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
            "item_type": "blade",
        })

    if not lib:
        raise SystemExit("No valid blade mask cutouts found.")

    print(f"[cutouts] built {len(lib)} blade masks")
    return lib


def select_shampoo_cutouts(real_cutouts, want_classes, counts_raw, rng):
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
        return selected

    n = int(counts_raw[0])
    if len(real_cutouts) < n:
        raise SystemExit(f"Only {len(real_cutouts)} shampoo cutouts available, but requested {n}")
    return rng.sample(real_cutouts, n)


def select_blade_cutouts(blade_cutouts, count, pick_mode, rng):
    count = max(1, int(count))
    if len(blade_cutouts) < count:
        raise SystemExit(f"Only {len(blade_cutouts)} blade masks available, but requested {count}")
    if pick_mode == "first":
        return blade_cutouts[:count]
    return rng.sample(blade_cutouts, count)


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
# Scene builder
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

    canvas_mask = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_mask[tray] = PALETTE_BGR[2]

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
            item_type = item.get("item_type", "shampoo")

            y_base = int(round(float(item.get("fit_cy", item["orig_cy"])) - h / 2.0))
            y_base = max(0, min(canvas_h - h, y_base))

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
                    region_B = canvas_app[y:y + h, x:x + w].astype(np.float32)

                    if int(item.get("train_id", 1)) == 3 or item_type == "blade":
                        region_A[obj_mask] = OVERLAP_BLADE_BGR
                        canvas_app[y:y + h, x:x + w] = render_blade_on_tray(region_B, obj_soft, obj_mask)
                    else:
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
    tray_preview_dir: Path = None,
):
    if canvas_mask.shape[:2] != pseudo_B_bgr.shape[:2]:
        raise RuntimeError(
            f"canvas_mask and pseudo_B must have same size. "
            f"Got mask={canvas_mask.shape[:2]} pseudo_B={pseudo_B_bgr.shape[:2]}"
        )

    ab = np.concatenate([canvas_mask, pseudo_B_bgr], axis=1)
    out_path = test_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), ab)

    if tray_mask_bool is not None and tray_preview_dir is not None:
        tray_preview_dir.mkdir(parents=True, exist_ok=True)
        T_path = tray_preview_dir / f"{stem}_T.png"
        cv2.imwrite(str(T_path), (tray_mask_bool.astype(np.uint8) * 255))
        print(f"[T] wrote tray mask preview: {T_path}")

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
# NIQE helpers
# =============================================================================

def _prepare_niqe_gray(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        raise ValueError("img_bgr is None")
    if img_bgr.ndim == 2:
        gray = img_bgr
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint8)


def init_niqe_backend():
    try:
        import torch
        import pyiqa
    except Exception as e:
        return {
            "name": None,
            "score_fn": None,
            "error": f"pyiqa NIQE unavailable: {e}",
        }

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        metric = pyiqa.create_metric("niqe", device=device)

        def _score(gray_u8: np.ndarray) -> float:
            rgb = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                val = metric(tensor)
            return float(val.detach().cpu().item())

        return {
            "name": "pyiqa",
            "score_fn": _score,
            "error": None,
        }
    except Exception as e:
        return {
            "name": None,
            "score_fn": None,
            "error": f"pyiqa NIQE init failed: {e}",
        }


def compute_niqe_for_images(images_by_stem, enabled=True):
    if not enabled:
        return {
            "enabled": False,
            "backend": None,
            "available": False,
            "error": "NIQE disabled by user",
            "per_image": {},
            "mean": None,
            "min": None,
            "max": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    backend = init_niqe_backend()
    if backend["score_fn"] is None:
        return {
            "enabled": True,
            "backend": None,
            "available": False,
            "error": backend["error"],
            "per_image": {},
            "mean": None,
            "min": None,
            "max": None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    per_image = {}
    vals = []

    for stem, img_bgr in images_by_stem.items():
        try:
            gray = _prepare_niqe_gray(img_bgr)
            score = backend["score_fn"](gray)
            per_image[stem] = {
                "niqe": float(score),
                "status": "ok",
            }
            vals.append(float(score))
        except Exception as e:
            per_image[stem] = {
                "niqe": None,
                "status": f"failed: {e}",
            }

    return {
        "enabled": True,
        "backend": backend["name"],
        "available": True,
        "error": None,
        "per_image": per_image,
        "mean": float(np.mean(vals)) if vals else None,
        "min": float(np.min(vals)) if vals else None,
        "max": float(np.max(vals)) if vals else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def print_niqe_metrics(niqe_info: dict):
    if not niqe_info.get("enabled", False):
        print(f"[NIQE] disabled | reason={niqe_info.get('error')}")
        return

    if not niqe_info.get("available", False):
        print(f"[NIQE] unavailable | reason={niqe_info.get('error')}")
        return

    print(
        f"[NIQE] backend={niqe_info.get('backend')} | "
        f"mean={niqe_info.get('mean')} | "
        f"min={niqe_info.get('min')} | "
        f"max={niqe_info.get('max')}"
    )

    for stem, info in niqe_info.get("per_image", {}).items():
        print(
            f"[NIQE] {stem} | "
            f"niqe={info.get('niqe')} | "
            f"status={info.get('status')}"
        )


def save_niqe_metrics(metrics_path: Path, niqe_info: dict):
    metrics_path.write_text(json.dumps(niqe_info, indent=2))
    print(f"[NIQE] wrote metrics to: {metrics_path}")


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
        sys.executable, str(PIX2PIX_DIR / "test.py"),
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

    results_root = Path("results") / MODEL_NAME / f"test_{epoch}"
    images_dir = results_root / "images"
    if not images_dir.exists():
        raise RuntimeError(f"pix2pix finished but images dir was not created: {images_dir}")

    return results_root


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
# Display helpers
# =============================================================================

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


def overlay_metric_panel(img_bgr: np.ndarray, lines) -> np.ndarray:
    if img_bgr is None:
        return None
    base = img_bgr.copy()
    h, w = base.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

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


def build_overlay_lines_for_image(stem: str, niqe_info: dict):
    lines = []

    per_image = niqe_info.get("per_image", {}) if niqe_info else {}
    stem_niqe = per_image.get(stem, {})
    niqe_val = stem_niqe.get("niqe")

    if niqe_val is not None:
        lines.append(f"NIQE {niqe_val:.4f} | lower is better")
    else:
        if niqe_info and niqe_info.get("enabled", False):
            lines.append("NIQE unavailable")
            if niqe_info.get("error"):
                lines.append(str(niqe_info["error"]))
        else:
            lines.append("NIQE disabled")

    return lines


def force_grayscale_bgr(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr is None:
        return None
    if img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def canvas_mask_to_debug_gray(canvas_mask: np.ndarray) -> np.ndarray:
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
# Batch helpers
# =============================================================================

def make_scene_stem(index: int) -> str:
    return f"scene_{index:04d}"


def choose_cutouts_for_mode(args, rng, real_cutouts=None, blade_cutouts=None):
    if args.generate_mode == "shampoo":
        want_classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]
        selected = select_shampoo_cutouts(real_cutouts, want_classes, counts_raw, rng)
        return selected

    if args.generate_mode == "blade":
        selected = select_blade_cutouts(
            blade_cutouts=blade_cutouts,
            count=1,
            pick_mode=args.blade_pick_mode,
            rng=rng,
        )
        return selected

    if args.generate_mode == "combo":
        want_classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
        counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]
        shampoo_selected = select_shampoo_cutouts(real_cutouts, want_classes, counts_raw, rng)
        blade_selected = select_blade_cutouts(
            blade_cutouts=blade_cutouts,
            count=1,
            pick_mode=args.blade_pick_mode,
            rng=rng,
        )
        return shampoo_selected + blade_selected

    return None


def print_selected_items(selected):
    print("[selected]")
    for i, s in enumerate(selected, 1):
        print(
            f"  item {i}: class={s['class_name']} file={s['file_name']} "
            f"orig_x0={s['orig_x0']} orig_y0={s['orig_y0']} "
            f"orig_w={s['orig_w']} orig_h={s['orig_h']}"
        )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate_mode", choices=["shampoo", "blade", "combo", "tray"], required=True)

    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--coco_json", type=str, default="")
    ap.add_argument("--classes", type=str, default="Shampoo")
    ap.add_argument("--count", type=str, default="1")
    ap.add_argument("--blade_mask_dir", type=str, default="")
    ap.add_argument("--blade_pick_mode", choices=["random", "first"], default="random")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dataset", type=str, default="datasets/_gen_stage18_shampoo_tray")
    ap.add_argument("--epoch", type=str, default="latest")
    ap.add_argument("--num_scenes", type=int, default=1,
                    help="How many scenes to generate in one run. FID becomes meaningful when this is >= 2.")

    ap.add_argument("--canvas_h", type=int, default=1024)
    ap.add_argument("--canvas_w", type=int, default=1024)

    ap.add_argument("--rand_scale_min", type=float, default=0.85)
    ap.add_argument("--rand_scale_max", type=float, default=0.85)
    ap.add_argument("--rand_rot_min", type=float, default=0.0)
    ap.add_argument("--rand_rot_max", type=float, default=40.0)

    ap.set_defaults(no_overlap=True)
    overlap_group = ap.add_mutually_exclusive_group()
    overlap_group.add_argument("--no_overlap", dest="no_overlap", action="store_true",
                               help="Disallow shampoo/blade overlap with each other (default).")
    overlap_group.add_argument("--allow_overlap", dest="no_overlap", action="store_false",
                               help="Allow shampoo/blade overlap with each other.")

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
                    help="Extra top/bottom tray padding.")

    ap.add_argument("--keep_intermediates", action="store_true")
    ap.add_argument("--keep_preview", action="store_true")
    ap.add_argument("--skip_pix2pix", action="store_true")
    ap.add_argument("--max_vertical_shift", type=int, default=0,
                    help="Small vertical placement freedom in pixels around the original y position.")

    ap.add_argument("--disable_niqe", action="store_true",
                    help="Disable per-image NIQE scoring.")
    ap.add_argument("--niqe_json_name", type=str, default="generated_niqe.json",
                    help="Filename for NIQE metrics JSON inside output folder.")

    args = ap.parse_args()

    if args.num_scenes < 1:
        raise SystemExit("--num_scenes must be >= 1")

    if args.horizontal_shift_only:
        print(
            f"[placement] horizontal_shift_only=True | "
            f"max_horizontal_shift={args.max_horizontal_shift} | "
            f"max_vertical_shift={args.max_vertical_shift} | "
            f"tray_vertical_margin_px={args.tray_vertical_margin_px}"
        )

    canvas_h = TRAIN_CFG["canvas_h"]
    canvas_w = TRAIN_CFG["canvas_w"]
    if args.canvas_h != canvas_h or args.canvas_w != canvas_w:
        print(f"[warn] forcing canvas from ({args.canvas_h},{args.canvas_w}) to training size ({canvas_h},{canvas_w})")

    out_root = Path(args.out_dataset)

    if args.generate_mode in {"shampoo", "blade", "combo"} and not args.tray_mask_dir:
        raise SystemExit("--tray_mask_dir is required for shampoo/blade/combo mode")

    tray_masks = None
    if args.generate_mode in {"shampoo", "blade", "combo"}:
        tray_masks, _ = load_tray_masks(
            Path(args.tray_mask_dir),
            out_h=canvas_h,
            out_w=canvas_w,
            thr=args.tray_mask_thr,
            invert=args.tray_mask_invert,
            close_px=args.tray_cc_close_px,
            dilate_px=args.tray_mask_dilate_px,
        )

    real_cutouts = None
    blade_cutouts = None

    if args.generate_mode in {"shampoo", "combo"}:
        if not args.images_dir or not args.coco_json:
            raise SystemExit("--images_dir and --coco_json are required for shampoo/combo mode")
        images_dir = Path(args.images_dir)
        coco = json.loads(Path(args.coco_json).read_text())
        real_cutouts = build_real_cutout_library(coco, images_dir)

    if args.generate_mode in {"blade", "combo"}:
        if not args.blade_mask_dir:
            raise SystemExit("--blade_mask_dir is required for blade/combo mode")
        blade_cutouts = build_blade_mask_library(Path(args.blade_mask_dir))

    test_dir = out_root / "test"
    clean_test_dir(test_dir)

    preview_dir = out_root / "preview"
    if args.keep_preview or args.keep_intermediates:
        preview_dir.mkdir(parents=True, exist_ok=True)

    generated_scene_info = []
    tray_mask_by_stem = {}

    for scene_idx in range(args.num_scenes):
        scene_seed = args.seed + scene_idx
        rng = random.Random(scene_seed)
        stem = make_scene_stem(scene_idx)

        print(f"\n[scene] {scene_idx + 1}/{args.num_scenes} | stem={stem} | seed={scene_seed}")

        if args.generate_mode == "tray":
            if args.tray_mask_path:
                tray_mask_path = Path(args.tray_mask_path)
            else:
                if not args.tray_mask_dir:
                    raise SystemExit("Provide either --tray_mask_path or --tray_mask_dir for tray mode.")
                tray_mask_path = choose_tray_mask_path(Path(args.tray_mask_dir), scene_seed)

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
            placed_count = 0
            selected = []
            print("[summary] tray mode | generated tray conditioning image from mask")
        else:
            selected = choose_cutouts_for_mode(args, rng, real_cutouts=real_cutouts, blade_cutouts=blade_cutouts)
            print_selected_items(selected)

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

            if args.generate_mode == "shampoo":
                print(f"[summary] shampoo mode | requested {len(selected)} items, placed {placed_count}")
            elif args.generate_mode == "blade":
                print(f"[summary] blade mode | requested {len(selected)} items, placed {placed_count}")
            elif args.generate_mode == "combo":
                shampoo_n = sum(1 for s in selected if s.get("item_type") == "shampoo")
                blade_n = sum(1 for s in selected if s.get("item_type") == "blade")
                print(
                    f"[summary] combo mode | shampoo={shampoo_n} blade={blade_n} "
                    f"total={len(selected)} placed={placed_count} | no_overlap={args.no_overlap}"
                )

        write_single_test_image(
            test_dir,
            canvas_mask,
            pseudo_B_bgr,
            tray_mask_bool=tray_mask_bool,
            stem=stem,
            tray_preview_dir=preview_dir if (args.keep_preview or args.keep_intermediates) else None,
        )
        tray_mask_by_stem[stem] = tray_mask_bool

        if args.keep_preview or args.keep_intermediates:
            cv2.imwrite(str(preview_dir / f"{stem}_mask.png"), canvas_mask)
            cv2.imwrite(str(preview_dir / f"{stem}_mask_debug_gray.png"), canvas_mask_to_debug_gray(canvas_mask))
            cv2.imwrite(str(preview_dir / f"{stem}_pseudo_B.png"), pseudo_B_bgr)
            cv2.imwrite(str(preview_dir / f"{stem}_canvas_app.png"), canvas_app)
            if tray_mask_bool is not None:
                cv2.imwrite(str(preview_dir / f"{stem}_tray.png"), tray_mask_bool.astype(np.uint8) * 255)

        generated_scene_info.append({
            "stem": stem,
            "seed": scene_seed,
            "placed_count": placed_count,
            "selected_count": len(selected),
        })

    if args.skip_pix2pix:
        print("[summary] skip_pix2pix enabled: using pseudo_B directly as final image(s).")
        results_root = None
        final_images = {}
        for p in sorted(test_dir.glob("scene_*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            if w == 2 * canvas_w:
                b = img[:, canvas_w:]
            else:
                b = img
            final_images[p.stem] = force_grayscale_bgr(b)
    else:
        results_root = run_pix2pix_test(
            temp_dataset_dir=out_root,
            epoch=args.epoch,
            num_test=args.num_scenes,
            tray_mask_dir=str(test_dir),
            use_tray_mask=True,
        )

        final_images = {}
        for scene in generated_scene_info:
            stem = scene["stem"]
            fake_path = resolve_fake_image_path(results_root / "images", stem)
            if fake_path is None:
                raise RuntimeError(f"Could not find fake_B output for stem '{stem}' in {results_root / 'images'}")

            raw_final_img = cv2.imread(str(fake_path), cv2.IMREAD_COLOR)
            if raw_final_img is None:
                raise RuntimeError(f"Could not read generated image: {fake_path}")

            raw_final_img = strip_existing_top_banner(raw_final_img)
            raw_final_img = force_grayscale_bgr(raw_final_img)
            final_images[stem] = raw_final_img

    niqe_info = compute_niqe_for_images(
        images_by_stem=final_images,
        enabled=(not args.disable_niqe),
    )
    print_niqe_metrics(niqe_info)

    out_dir = out_root / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, raw_img in final_images.items():
        overlay_lines = build_overlay_lines_for_image(stem, niqe_info)
        raw_annotated = overlay_metric_panel(raw_img, overlay_lines)
        final_img = export_smooth_2x(raw_annotated)
        out_path = out_dir / f"{stem}_{args.generate_mode}_smooth2x.png"
        cv2.imwrite(str(out_path), final_img)
        print(f"Saved generated result to: {out_path}")

    niqe_path = out_dir / args.niqe_json_name
    save_niqe_metrics(niqe_path, niqe_info)

    summary_path = out_dir / f"generated_{args.generate_mode}_summary.json"
    summary = {
        "generate_mode": args.generate_mode,
        "num_scenes_requested": args.num_scenes,
        "num_scenes_generated": len(final_images),
        "seed_start": args.seed,
        "no_overlap": args.no_overlap,
        "niqe": niqe_info,
        "scenes": generated_scene_info,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote scene summary to: {summary_path}")

    if (not args.keep_intermediates) and (results_root is not None):
        cleanup_intermediate_outputs(out_root, results_root, keep_preview=args.keep_preview)

    print(f"\nDone. Output folder: {out_dir}")
    print(f"Done. NIQE JSON: {niqe_path}")
    print(f"Done. Summary JSON: {summary_path}")


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
python notebooks/Pix2Pix/generate_pix2pixV2_NIQE.py \
  --generate_mode shampoo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 166 \
  --out_dataset datasets/_gen_stage18_shampoo_tray \
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
  --num_scenes 20 \
  --keep_intermediates  


BLADE WITH TRAY:

python notebooks/Pix2Pix/generate_pix2pixV2_NIQE.py \
  --generate_mode blade \
  --blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY/matched_masks/train/blade \
  --seed 777 \
  --out_dataset datasets/_gen_stage18_blade_tray \
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
  --num_scenes 20 \
  --keep_intermediates


  
BLADE + SHAMPOO WITH TRAY:
python notebooks/Pix2Pix/generate_pix2pixV2_NIQE.py \
  --generate_mode combo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/train/blade \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_TGT/matched_masks/train/tray \
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
  --out_dataset datasets/_gen_stage18_combo_tray \
  --num_scenes 20 \
  --seed 77

  add this for overlap  --allow_overlap
  
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

