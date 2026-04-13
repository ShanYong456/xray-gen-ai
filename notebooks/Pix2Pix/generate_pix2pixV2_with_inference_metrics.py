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
}
OVERLAP_BGR = (255, 255, 0)  # cyan in RGB after cv2 write

MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage14TEST"
PIX2PIX_DIR = Path("external/pix2pix")

TRAIN_CFG = dict(
    input_nc=6,
    norm="instance",
    use_appearance_channel=False,
    use_tray_mask=True,
    class_nc=2,
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


# =============================================================================
# Cutout transforms
# =============================================================================

def transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max):
    mask_bin = (item["mask_bin"] > 127).astype(np.uint8) * 255
    gray = item["gray"].copy()

    base_fit_scale = float(item.get("fit_scale", 1.0))
    s = rng.uniform(scale_min, scale_max) * base_fit_scale
    base_h, base_w = mask_bin.shape[:2]
    aug_w = max(1, int(round(base_w * s)))
    aug_h = max(1, int(round(base_h * s)))

    mask_soft = cv2.resize(mask_bin.astype(np.float32), (aug_w, aug_h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.resize(gray, (aug_w, aug_h), interpolation=cv2.INTER_LINEAR)

    ang = rng.uniform(rot_min, rot_max)

    h0, w0 = mask_soft.shape[:2]
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
    M[0, 2] += nw / 2 - w0 / 2
    M[1, 2] += nh / 2 - h0 / 2

    mask_soft = cv2.warpAffine(mask_soft, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    gray = cv2.warpAffine(gray, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)

    mask_soft = cv2.GaussianBlur(mask_soft, (0, 0), 1.2)
    mask_u8 = np.clip(mask_soft, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    _, m = cv2.threshold(mask_u8, 96, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(m > 0)
    if not len(xs):
        return None

    gray[m == 0] = 0

    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out_gray = gray[y1:y2, x1:x2].copy()
    out_soft = (mask_soft[y1:y2, x1:x2] / 255.0).astype(np.float32)
    out_hard = (m[y1:y2, x1:x2] > 0)

    return {
        "gray": out_gray,
        "soft_mask": out_soft,
        "hard_mask": out_hard,
        "h": out_gray.shape[0],
        "w": out_gray.shape[1],
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
    ys.append(y_min)
    ys.append(y_max)

    ys = sorted(set(ys), key=lambda yy: abs(yy - y_base))
    chunks = [ys[i:i + 8] for i in range(0, len(ys), 8)]
    for chunk in chunks:
        rng.shuffle(chunk)
    return [y for chunk in chunks for y in chunk]


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
    no_overlap=True,
    x_search_step=X_SEARCH_STEP,
    max_transform_candidates=MAX_TRANSFORM_CANDIDATES,
    canvas_h=1024,
    canvas_w=1024,
):
    tray = rng.choice(tray_masks)

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
            obj_gray = t["gray"].astype(np.float32)

            y_base = int(round(float(item.get("fit_cy", item["orig_cy"])) - h / 2.0))
            y_base = max(0, min(canvas_h - h, y_base))

            if horizontal_shift_only:
                y_candidates = build_soft_y_candidates(
                    canvas_h=canvas_h,
                    obj_h=h,
                    y_base=y_base,
                    radius=0,
                    step=1,
                    rng=rng,
                )
            else:
                y_candidates = list(range(0, canvas_h - h + 1, max(1, x_search_step)))
                if y_base not in y_candidates:
                    y_candidates.append(y_base)
                rng.shuffle(y_candidates)

            x_base_scaled = int(round(float(item.get("fit_x0", item["orig_x0"]))))
            x_base_scaled = max(0, min(canvas_w - w, x_base_scaled))
            x_min = max(0, x_base_scaled - max_horizontal_shift)
            x_max = min(canvas_w - w, x_base_scaled + max_horizontal_shift)
            if x_min > x_max:
                continue

            candidate_xs = build_candidate_xs(x_min, x_max, x_base_scaled, max(1, x_search_step), rng)

            for y in y_candidates:
                for x in candidate_xs:
                    roi_tray = tray[y:y + h, x:x + w]
                    if roi_tray.shape[:2] != obj_mask.shape or not roi_tray.any():
                        continue
                    if not np.all(roi_tray[obj_mask]):
                        continue

                    roi_occ = occ[y:y + h, x:x + w]
                    if no_overlap and np.any(roi_occ & obj_mask):
                        continue

                    # IMPORTANT:
                    # Stage13 training expects shampoo-over-tray overlap as CYAN in A
                    region_A = canvas_mask[y:y + h, x:x + w]
                    region_A[obj_mask] = OVERLAP_BGR

                    # pseudo_B preview only
                    region_B = canvas_app[y:y + h, x:x + w].astype(np.float32)

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
        "--class_nc=2",
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
    overlap = np.all(canvas_mask == np.array(OVERLAP_BGR, dtype=np.uint8), axis=2)

    shampoo_mask = shampoo_only | overlap
    tray_mask = tray_only | overlap
    return shampoo_mask.astype(bool), tray_mask.astype(bool), overlap.astype(bool)


def _score_from_distance(value: float, target: float, tolerance: float) -> float:
    tolerance = max(float(tolerance), 1e-6)
    return float(max(0.0, 1.0 - abs(float(value) - float(target)) / tolerance))


def _to_gray01(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return np.clip(gray, 0.0, 1.0)


def compute_inference_quality(final_img_bgr: np.ndarray, canvas_mask: np.ndarray, tray_mask_bool: np.ndarray = None, generate_mode: str = "shampoo"):
    gray01 = _to_gray01(final_img_bgr)
    shampoo_mask, tray_mask_from_canvas, overlap_mask = semantic_masks_from_canvas_mask(canvas_mask)

    if tray_mask_bool is not None:
        tray_mask = tray_mask_bool.astype(bool)
    else:
        tray_mask = tray_mask_from_canvas.astype(bool)

    obj_mask = shampoo_mask.astype(bool)
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

    score_nonblank = _score_from_distance(nonblank_ratio, 0.18 if generate_mode == "shampoo" else 0.10, 0.18)
    score_contrast = _score_from_distance(global_std, 0.22, 0.18)
    score_exposure = _score_from_distance(global_mean, 0.30, 0.22)
    score_sharpness = min(1.0, lap_var / 250.0)

    if generate_mode == "tray":
        tray_fill = float(gray01[tray_mask].mean()) if tray_mask.any() else 0.0
        tray_bg = float(gray01[~tray_mask].mean()) if (~tray_mask).any() else 0.0
        tray_contrast = max(0.0, min(1.0, abs(tray_fill - tray_bg) / 0.35))

        metrics["tray_fill_mean"] = tray_fill
        metrics["tray_bg_mean"] = tray_bg
        metrics["tray_contrast"] = tray_contrast

        score_raw = (
            0.30 * score_nonblank +
            0.25 * score_contrast +
            0.20 * score_exposure +
            0.10 * score_sharpness +
            0.15 * tray_contrast
        )
    else:
        obj_area_ratio = float(obj_mask.mean()) if obj_mask.size else 0.0
        obj_mean = float(gray01[obj_mask].mean()) if obj_mask.any() else 0.0
        tray_mean = float(gray01[tray_mask].mean()) if tray_mask.any() else 0.0
        bg_mean = float(gray01[~tray_mask].mean()) if (~tray_mask).any() else 0.0
        inside_tray_ratio = float((obj_mask & tray_mask).sum() / max(1, obj_mask.sum()))
        overlap_ratio = float(overlap_mask.sum() / max(1, obj_mask.sum()))

        metrics["object_area_ratio"] = obj_area_ratio
        metrics["object_mean"] = obj_mean
        metrics["tray_mean"] = tray_mean
        metrics["bg_mean"] = bg_mean
        metrics["inside_tray_ratio"] = inside_tray_ratio
        metrics["overlap_ratio"] = overlap_ratio

        score_area = _score_from_distance(obj_area_ratio, 0.05, 0.05)
        score_inside = inside_tray_ratio
        score_overlap = _score_from_distance(overlap_ratio, 1.0, 0.25)
        score_obj_contrast = max(0.0, min(1.0, abs(obj_mean - tray_mean) / 0.25))

        score_raw = (
            0.20 * score_nonblank +
            0.15 * score_contrast +
            0.10 * score_exposure +
            0.10 * score_sharpness +
            0.15 * score_area +
            0.20 * score_inside +
            0.05 * score_overlap +
            0.05 * score_obj_contrast
        )

        metrics["score_area"] = score_area
        metrics["score_inside"] = score_inside
        metrics["score_overlap"] = score_overlap
        metrics["score_obj_contrast"] = score_obj_contrast

    metrics["score_nonblank"] = score_nonblank
    metrics["score_contrast"] = score_contrast
    metrics["score_exposure"] = score_exposure
    metrics["score_sharpness"] = score_sharpness

    final_score = float(np.clip(score_raw, 0.0, 1.0) * 100.0)
    metrics["quality_score"] = final_score
    metrics["quality_label"] = (
        "good" if final_score >= 75 else
        "okay" if final_score >= 60 else
        "weak"
    )
    return metrics


def save_quality_metrics(metrics_path: Path, metrics: dict):
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[metric] wrote inference quality metrics to: {metrics_path}")


def print_quality_metrics(metrics: dict):
    print(
        f"[metric] inference quality = {metrics['quality_score']:.2f}/100 | "
        f"label={metrics['quality_label']} | "
        f"nonblank={metrics['score_nonblank']:.3f} | "
        f"contrast={metrics['score_contrast']:.3f} | "
        f"exposure={metrics['score_exposure']:.3f} | "
        f"sharpness={metrics['score_sharpness']:.3f}"
    )


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate_mode", choices=["shampoo", "tray"], required=True)

    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--coco_json", type=str, default="")
    ap.add_argument("--classes", type=str, default="Shampoo")
    ap.add_argument("--count", type=str, default="1")

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

    ap.add_argument("--tray_mask_path", type=str, default="")
    ap.add_argument("--tray_mask_thr", type=float, default=0.5)
    ap.add_argument("--tray_mask_invert", action="store_true")
    ap.add_argument("--tray_cc_close_px", type=int, default=2)
    ap.add_argument("--tray_mask_dilate_px", type=int, default=0)
    ap.add_argument("--tray_mask_dir", type=str, default="")

    ap.add_argument("--keep_intermediates", action="store_true")
    ap.add_argument("--keep_preview", action="store_true")
    ap.add_argument("--skip_pix2pix", action="store_true")

    args = ap.parse_args()

    # FORCE match training size
    canvas_h = TRAIN_CFG["canvas_h"]
    canvas_w = TRAIN_CFG["canvas_w"]
    if args.canvas_h != canvas_h or args.canvas_w != canvas_w:
        print(f"[warn] forcing canvas from ({args.canvas_h},{args.canvas_w}) to training size ({canvas_h},{canvas_w})")

    rng = random.Random(args.seed)
    out_root = Path(args.out_dataset)
    tray_mask_bool = None

    if args.generate_mode == "shampoo":
        if not args.images_dir or not args.coco_json:
            raise SystemExit("--images_dir and --coco_json are required for shampoo mode")
        if not args.tray_mask_dir:
            raise SystemExit("--tray_mask_dir is required for shampoo mode")

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
            no_overlap=args.no_overlap,
            x_search_step=max(1, int(args.x_search_step)),
            max_transform_candidates=max(1, int(args.max_transform_candidates)),
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )

        tray_mask_bool = used_tray
        print(f"[summary] shampoo mode | requested {len(selected)} items, placed {placed_count}")

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
        cv2.imwrite(str(preview_dir / f"mask_seed{args.seed}.png"), canvas_mask)
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

        final_img = export_smooth_2x(raw_final_img)

    metrics = compute_inference_quality(
        final_img_bgr=raw_final_img,
        canvas_mask=canvas_mask,
        tray_mask_bool=tray_mask_bool,
        generate_mode=args.generate_mode,
    )
    print_quality_metrics(metrics)

    out_dir = out_root / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "tray" if args.generate_mode == "tray" else "shampoo"
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