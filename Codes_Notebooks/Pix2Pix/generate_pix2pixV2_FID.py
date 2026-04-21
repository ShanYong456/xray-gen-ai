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
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms

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

MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage19_COMPLETESyn"
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


def build_real_b_from_ab(ab_bgr: np.ndarray, canvas_w: int) -> np.ndarray:
    if ab_bgr is None:
        raise RuntimeError("AB image is None")
    h, w = ab_bgr.shape[:2]
    if w < 2 * canvas_w:
        raise RuntimeError(f"Expected AB image width >= {2 * canvas_w}, got {w}")
    b = ab_bgr[:, canvas_w:canvas_w * 2]
    gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def materialize_dataset_eval_inputs(
    src_dataroot: Path,
    phase: str,
    out_root: Path,
    canvas_w: int,
    max_images: int = 0,
    tray_mask_dir: str = "",
    pick_mode: str = "first",
    seed: int = 123,
):
    src_phase_dir = src_dataroot / phase
    if not src_phase_dir.exists():
        raise SystemExit(f"dataset_eval phase dir not found: {src_phase_dir}")

    ab_paths = sorted([p for p in src_phase_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}])
    if not ab_paths:
        raise SystemExit(f"No aligned AB images found in: {src_phase_dir}")

    if max_images and max_images > 0:
        if pick_mode == "random":
            rng = random.Random(seed)
            if len(ab_paths) > max_images:
                ab_paths = sorted(rng.sample(ab_paths, max_images), key=lambda p: p.name)
            else:
                ab_paths = ab_paths[:max_images]
        else:
            ab_paths = ab_paths[:max_images]

    test_dir = out_root / phase
    clean_test_dir(test_dir)

    real_eval_dir = out_root / "real_eval"
    if real_eval_dir.exists():
        shutil.rmtree(real_eval_dir, ignore_errors=True)
    real_eval_dir.mkdir(parents=True, exist_ok=True)

    tray_out_dir = out_root / "dataset_eval_tray_masks"
    if tray_out_dir.exists():
        shutil.rmtree(tray_out_dir, ignore_errors=True)
    tray_out_dir.mkdir(parents=True, exist_ok=True)

    tray_src_dir = Path(tray_mask_dir) if tray_mask_dir else None

    selected_stems = []
    for p in ab_paths:
        stem = p.stem
        shutil.copy2(p, test_dir / p.name)

        ab = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if ab is None:
            raise RuntimeError(f"Could not read aligned AB image: {p}")
        real_b = build_real_b_from_ab(ab, canvas_w=canvas_w)
        cv2.imwrite(str(real_eval_dir / f"{stem}.png"), real_b)

        if tray_src_dir is not None:
            mask_path = tray_src_dir / f"{stem}.png"
            if not mask_path.exists():
                raise RuntimeError(f"Missing tray mask for dataset_eval image: {mask_path}")
            shutil.copy2(mask_path, tray_out_dir / mask_path.name)

        selected_stems.append(stem)

    return test_dir, real_eval_dir, tray_out_dir, selected_stems


def prepare_real_eval_dir(real_eval_dir: Path, out_root: Path, canvas_w: int) -> Path:
    """
    If real_eval_dir already contains pure real target images, return it unchanged.
    If it appears to contain aligned AB images, automatically extract the B-half into
    out_root/real_eval_from_aligned and return that folder.
    """
    real_paths = _list_image_files(real_eval_dir)
    if not real_paths:
        raise SystemExit(f"No real eval images found in: {real_eval_dir}")

    sample = cv2.imread(str(real_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"Could not read real eval sample: {real_paths[0]}")

    h, w = sample.shape[:2]
    looks_aligned_ab = w >= 2 * canvas_w
    if not looks_aligned_ab:
        print(f"[FID] using real eval dir as-is: {real_eval_dir}")
        return real_eval_dir

    auto_dir = out_root / "real_eval_from_aligned"
    if auto_dir.exists():
        shutil.rmtree(auto_dir, ignore_errors=True)
    auto_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FID] detected aligned AB images in {real_eval_dir}; extracting real B halves to {auto_dir}")
    for p in real_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        b = build_real_b_from_ab(img, canvas_w=canvas_w)
        cv2.imwrite(str(auto_dir / f"{p.stem}.png"), b)

    return auto_dir

# =============================================================================
# FID helpers
# =============================================================================

def _list_image_files(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _load_image_rgb_uint8(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


def _bgr_to_rgb_uint8(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return rgb


def _to_fid_tensor_uint8(img_rgb: np.ndarray) -> torch.Tensor:
    # torchmetrics FID expects uint8 tensor in [0,255], shape [N,C,H,W]
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)
    return t


def resize_and_pad_gray_to(gray: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h == target_h and w == target_w:
        return gray
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def stage_fid_dirs_like_training(images_by_stem, real_eval_dir: Path, out_root: Path):
    real_paths = _list_image_files(real_eval_dir)
    if not real_paths:
        raise RuntimeError(f"No images found in real_eval_dir: {real_eval_dir}")
    if not images_by_stem:
        raise RuntimeError("No fake images to stage for FID")

    sample_fake = next(iter(images_by_stem.values()))
    target_h, target_w = sample_fake.shape[:2]

    stage_root = out_root / "fid_staging"
    real_stage = stage_root / "real"
    fake_stage = stage_root / "fake"
    if stage_root.exists():
        shutil.rmtree(stage_root, ignore_errors=True)
    real_stage.mkdir(parents=True, exist_ok=True)
    fake_stage.mkdir(parents=True, exist_ok=True)

    for p in real_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = resize_and_pad_gray_to(img, target_h=target_h, target_w=target_w)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(real_stage / f"{p.stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    for stem, img_bgr in images_by_stem.items():
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = resize_and_pad_gray_to(gray, target_h=target_h, target_w=target_w)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(str(fake_stage / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    return real_stage, fake_stage


def compute_fid_for_images(images_by_stem, real_eval_dir: Path, enabled=True, out_root: Path = None):
    if not enabled:
        return {
            "enabled": False,
            "available": False,
            "error": "FID disabled by user",
            "num_real_images": 0,
            "num_fake_images": 0,
            "fid": None,
            "per_image": {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    if out_root is None:
        out_root = Path(".")

    real_paths = _list_image_files(real_eval_dir)
    if len(real_paths) < 2:
        return {
            "enabled": True,
            "available": False,
            "error": f"Need at least 2 real images for FID. Got {len(real_paths)}",
            "num_real_images": len(real_paths),
            "num_fake_images": len(images_by_stem),
            "fid": None,
            "per_image": {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    if len(images_by_stem) < 2:
        return {
            "enabled": True,
            "available": False,
            "error": f"Need at least 2 generated images for FID. Got {len(images_by_stem)}",
            "num_real_images": len(real_paths),
            "num_fake_images": len(images_by_stem),
            "fid": None,
            "per_image": {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_count = 0
    fake_count = 0

    try:
        real_stage, fake_stage = stage_fid_dirs_like_training(
            images_by_stem=images_by_stem,
            real_eval_dir=real_eval_dir,
            out_root=out_root,
        )
        real_count = len(_list_image_files(real_stage))
        fake_count = len(_list_image_files(fake_stage))

        fid_value = None
        backend = None
        backend_error = None

        try:
            from torch_fidelity import calculate_metrics
            metrics = calculate_metrics(
                input1=str(real_stage),
                input2=str(fake_stage),
                cuda=torch.cuda.is_available(),
                isc=False,
                fid=True,
                kid=False,
                verbose=False,
            )
            fid_value = float(metrics["frechet_inception_distance"])
            backend = "torch_fidelity"
        except Exception as e:
            backend_error = str(e)

        if fid_value is None:
            fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            for p in _list_image_files(real_stage):
                img_rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                t = _to_fid_tensor_uint8(img_rgb).to(device)
                fid_metric.update(t, real=True)
            for p in _list_image_files(fake_stage):
                img_rgb = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                t = _to_fid_tensor_uint8(img_rgb).to(device)
                fid_metric.update(t, real=False)
            fid_value = float(fid_metric.compute().item())
            backend = f"torchmetrics_fallback ({backend_error})" if backend_error else "torchmetrics_fallback"

        per_image = {
            stem: {
                "fid": None,
                "status": "FID is dataset-level, not per-image"
            }
            for stem in images_by_stem.keys()
        }

        return {
            "enabled": True,
            "available": True,
            "error": None,
            "backend": backend,
            "num_real_images": real_count,
            "num_fake_images": fake_count,
            "fid": fid_value,
            "per_image": per_image,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }

    except Exception as e:
        return {
            "enabled": True,
            "available": False,
            "error": str(e),
            "num_real_images": real_count,
            "num_fake_images": fake_count,
            "fid": None,
            "per_image": {},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }


def print_fid_metrics(fid_info: dict):
    if not fid_info.get("enabled", False):
        print(f"[FID] disabled | reason={fid_info.get('error')}")
        return

    if not fid_info.get("available", False):
        print(f"[FID] unavailable | reason={fid_info.get('error')}")
        return

    print(
        f"[FID] score={fid_info.get('fid')} | "
        f"real_count={fid_info.get('num_real_images')} | "
        f"fake_count={fid_info.get('num_fake_images')}"
    )


def save_fid_metrics(metrics_path: Path, fid_info: dict):
    metrics_path.write_text(json.dumps(fid_info, indent=2))
    print(f"[FID] wrote metrics to: {metrics_path}")


def compute_fid_via_fid_eval_script(args, out_root: Path):
    work_dir = out_root / "fid_eval_like_training"
    cmd = [
        sys.executable, "Codes_Notebooks/Pix2Pix/fid_eval.py",
        "--dataroot", args.dataset_eval_dataroot,
        "--name", MODEL_NAME,
        "--epoch", args.epoch,
        "--phase", args.dataset_eval_phase,
        "--work_dir", str(work_dir),
        "--max_images", str(args.dataset_eval_max_images if args.dataset_eval_max_images > 0 else 500),
        "--input_nc", str(TRAIN_CFG["input_nc"]),
        "--output_nc", "3",
        "--netG", "unet_256",
        "--netD", "n_layers",
        "--n_layers_D", "4",
        "--norm", TRAIN_CFG["norm"],
        "--class_nc", str(TRAIN_CFG["class_nc"]),
        "--thickness_nc", str(TRAIN_CFG["thickness_nc"]),
        "--preprocess", "none",
        "--load_size", "0",
        "--crop_size", "0",
        "--pad_to_canvas",
        "--canvas_w", str(TRAIN_CFG["canvas_w"]),
        "--canvas_h", str(TRAIN_CFG["canvas_h"]),
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--use_tray_mask",
        "--tray_mask_dir", args.tray_mask_dir,
    ]
    if args.blade_mask_dir:
        cmd += ["--synthetic_blade_mask_dir", args.blade_mask_dir]
    print("[FID] calling fid_eval.py:", " ".join(cmd))
    subprocess.check_call(cmd)

    metrics_json = work_dir / MODEL_NAME / f"epoch_{args.epoch}" / "metrics.json"
    if not metrics_json.exists():
        raise RuntimeError(f"fid_eval.py completed but metrics.json was not found: {metrics_json}")
    data = json.loads(metrics_json.read_text())
    return {
        "enabled": True,
        "available": True,
        "error": None,
        "backend": "fid_eval.py/torch_fidelity",
        "num_real_images": int(data.get("num_images", 0)),
        "num_fake_images": int(data.get("num_images", 0)),
        "fid": float(data.get("fid")) if data.get("fid") is not None else None,
        "per_image": {},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_metrics_json": str(metrics_json),
        "raw_metrics": data.get("raw_metrics", {}),
    }

# =============================================================================
# Pix2pix inference
# =============================================================================

def run_pix2pix_test(
    temp_dataset_dir,
    epoch="latest",
    num_test=None,
    tray_mask_dir="",
    use_tray_mask=True,
    phase="test",
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
        f"--phase={phase}",
        "--eval",
        "--class_nc=3",
        "--thickness_nc=1",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--mask_thr=0.05",
        f"--canvas_h={cfg['canvas_h']}",
        f"--canvas_w={cfg['canvas_w']}",
        "--pad_to_canvas",
        "--serial_batches",
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

    results_root = Path("results") / MODEL_NAME / f"{phase}_{epoch}"
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

def build_overlay_lines_for_image(stem: str, fid_info: dict):
    lines = []

    if fid_info and fid_info.get("enabled", False):
        if fid_info.get("available", False) and fid_info.get("fid") is not None:
            lines.append(f"FID {fid_info['fid']:.4f} | lower is better")
            lines.append("FID is dataset-level, not per-image")
        else:
            lines.append("FID unavailable")
            if fid_info.get("error"):
                lines.append(str(fid_info["error"]))
    else:
        lines.append("FID disabled")

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
    ap.add_argument("--generate_mode", choices=["shampoo", "blade", "combo", "tray", "dataset_eval", "real_dataset", "fit_real_dataset"], required=True)

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

    ap.add_argument("--real_eval_dir", type=str, default="",
            help="Directory of real X-ray images used as the real set for FID. Optional for dataset_eval mode.")

    ap.add_argument("--dataset_eval_dataroot", type=str, default="",
            help="For --generate_mode dataset_eval: aligned pix2pix dataroot to evaluate on.")
    ap.add_argument("--dataset_eval_phase", type=str, default="train",
            help="For --generate_mode dataset_eval: which phase folder inside dataroot to use (train/test/val).")
    ap.add_argument("--dataset_eval_max_images", type=int, default=0,
            help="For --generate_mode dataset_eval/real_dataset: limit number of aligned AB images. 0 means use all.")
    ap.add_argument("--dataset_eval_pick", choices=["first", "random"], default="first",
            help="For --generate_mode dataset_eval/real_dataset: choose the first N images or a random subset.")
    ap.add_argument("--disable_fid", action="store_true",
            help="Disable FID scoring.")
    ap.add_argument("--fid_use_fid_eval", action="store_true",
            help="For dataset-based modes, compute FID by calling fid_eval.py directly.")
    ap.add_argument("--strip_banner_for_preview", action="store_true",
            help="Only for saved preview images. FID always uses raw fake_B without banner stripping.")

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

    if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"}:
        if not args.dataset_eval_dataroot:
            raise SystemExit("--dataset_eval_dataroot is required for generate_mode=dataset_eval, real_dataset, or fit_real_dataset")
        if not args.tray_mask_dir:
            raise SystemExit("--tray_mask_dir is required for generate_mode=dataset_eval/real_dataset/fit_real_dataset so tray masks match the aligned inputs")
    elif args.generate_mode in {"shampoo", "blade", "combo"} and not args.tray_mask_dir:
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

    dataset_eval_real_dir = None
    dataset_eval_tray_dir = None

    runtime_phase = args.dataset_eval_phase if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else "test"
    test_dir = out_root / runtime_phase
    clean_test_dir(test_dir)

    preview_dir = out_root / "preview"
    if args.keep_preview or args.keep_intermediates:
        preview_dir.mkdir(parents=True, exist_ok=True)

    generated_scene_info = []
    tray_mask_by_stem = {}
    runtime_tray_dir = out_root / "runtime_tray_masks"
    runtime_tray_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"}:
        test_dir, dataset_eval_real_dir, dataset_eval_tray_dir, selected_stems = materialize_dataset_eval_inputs(
            src_dataroot=Path(args.dataset_eval_dataroot),
            phase=args.dataset_eval_phase,
            out_root=out_root,
            canvas_w=canvas_w,
            max_images=max(0, int(args.dataset_eval_max_images)),
            tray_mask_dir=args.tray_mask_dir,
            pick_mode=args.dataset_eval_pick,
            seed=args.seed,
        )
        generated_scene_info = [
            {
                "stem": stem,
                "seed": None,
                "placed_count": None,
                "selected_count": None,
            }
            for stem in selected_stems
        ]
    else:
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
                tray_preview_dir=runtime_tray_dir,
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
            num_test=(len(generated_scene_info) if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else args.num_scenes),
            tray_mask_dir=(str(dataset_eval_tray_dir) if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else str(runtime_tray_dir)),
            use_tray_mask=True,
            phase=(args.dataset_eval_phase if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else "test"),
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

            # IMPORTANT: FID must use the raw fake_B image.
            # Do not crop/strip anything here, otherwise the fake and real sets
            # are no longer geometrically comparable.
            final_images[stem] = force_grayscale_bgr(raw_final_img)

    fid_info = {
        "enabled": False,
        "available": False,
        "error": "FID not computed",
        "num_real_images": 0,
        "num_fake_images": 0,
        "fid": None,
        "per_image": {},
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fid_reference": (str(dataset_eval_real_dir) if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else args.real_eval_dir),
    }

    if not args.disable_fid:
        if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"}:
            real_eval_dir = dataset_eval_real_dir
        else:
            if not args.real_eval_dir:
                raise SystemExit("--real_eval_dir is required for FID unless using generate_mode=dataset_eval or --disable_fid")
            real_eval_dir = prepare_real_eval_dir(
                real_eval_dir=Path(args.real_eval_dir),
                out_root=out_root,
                canvas_w=canvas_w,
            )

        fid_info = compute_fid_for_images(
            images_by_stem=final_images,
            real_eval_dir=real_eval_dir,
            enabled=True,
            out_root=out_root,
        )

    print_fid_metrics(fid_info)

    out_dir = out_root / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    for stem, raw_img in final_images.items():
        preview_img = strip_existing_top_banner(raw_img) if args.strip_banner_for_preview else raw_img
        overlay_lines = build_overlay_lines_for_image(stem, fid_info)
        raw_annotated = overlay_metric_panel(preview_img, overlay_lines)
        final_img = export_smooth_2x(raw_annotated)
        out_path = out_dir / f"{stem}_{args.generate_mode}_smooth2x.png"
        cv2.imwrite(str(out_path), final_img)
        print(f"Saved generated result to: {out_path}")

    fid_path = out_dir / "generated_fid.json"
    save_fid_metrics(fid_path, fid_info)

    summary_path = out_dir / f"generated_{args.generate_mode}_summary.json"
    summary = {
        "generate_mode": args.generate_mode,
        "dataset_source": args.dataset_eval_dataroot if args.generate_mode in {"dataset_eval", "real_dataset", "fit_real_dataset"} else None,
        "num_scenes_requested": args.num_scenes,
        "num_scenes_generated": len(final_images),
        "seed_start": args.seed,
        "no_overlap": args.no_overlap,
        "fid": fid_info,
        "scenes": generated_scene_info,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[summary] wrote scene summary to: {summary_path}")

    if (not args.keep_intermediates) and (results_root is not None):
        cleanup_intermediate_outputs(out_root, results_root, keep_preview=args.keep_preview)

    print(f"\nDone. Output folder: {out_dir}")
    print(f"Done. FID JSON: {fid_path}")
    print(f"Done. Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
"""

GENERATE SHAMPOO & TRAY
python Codes_Notebooks/Pix2Pix/generate_pix2pixV2_FID.py \
  --generate_mode shampoo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --seed 188 \
  --out_dataset results/_gen_stage19_shampoo_tray \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
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
  --real_eval_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test \
  --keep_intermediates --mahal_pca_dim 32


BLADE WITH TRAY:

python Codes_Notebooks/Pix2Pix/generate_pix2pixV2_FID.py \
  --generate_mode blade \
  --blade_mask_dir results/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --seed 777 \
  --out_dataset results/_gen_stage19_blade_tray \
  --epoch latest \
  --canvas_h 1024 \
  --canvas_w 1024 \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
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
  --real_eval_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test \
  --keep_intermediates --mahal_pca_dim 32


  
BLADE + SHAMPOO WITH TRAY:
python Codes_Notebooks/Pix2Pix/generate_pix2pixV2_FID.py \
  --generate_mode combo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
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
  --out_dataset results/_gen_stage19_combo_tray \
  --num_scenes 20 \
  --seed 17 --allow_overlap \
  --real_eval_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test \
  --mahal_pca_dim 32


  python Codes_Notebooks/Pix2Pix/generate_pix2pixV2_FID.py \
  --generate_mode combo \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --classes Shampoo \
  --count 1 \
  --blade_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/blade \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
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
  --allow_overlap \
  --num_scenes 100 \
  --seed 188 \
  --real_eval_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test

  add this for overlap  --allow_overlap  --no_overlap


python Codes_Notebooks/Pix2Pix/generate_pix2pixV2_FID.py \
  --generate_mode dataset_eval \
  --dataset_eval_dataroot datasets/SHAMPOOBLADEWITHTRAY_COMPLETE \
  --dataset_eval_phase train \
  --dataset_eval_max_images 100 \
  --dataset_eval_pick random \
  --seed 188 \
  --tray_mask_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/matched_masks/train/tray \
  --out_dataset results/_gen_fit_real_dataset \
  --epoch latest \
  --fid_use_fid_eval

  To print out classes that you have:
  python - <<'PY'
    import json
    c=json.load(open("data/raw/Non-Contraband/result.json"))
    print(sorted([x["name"] for x in c["categories"]]))
    PY
"""
