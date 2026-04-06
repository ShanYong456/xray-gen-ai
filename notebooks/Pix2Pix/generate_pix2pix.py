from pathlib import Path
import argparse
import json
import random
import subprocess
import shutil
import os

import cv2
import numpy as np
from PIL import Image

try:
    from halo_remover import remove_halo_with_mask
    HAS_HALO_REMOVER = True
except ImportError:
    HAS_HALO_REMOVER = False


# =============================================================================
# CONFIG — must match your training options exactly
# =============================================================================
SIZE = 1024                    # model/test size
CANVAS_H = 1024                # generation canvas height (overridable by CLI)
CANVAS_W = 1024                # generation canvas width  (overridable by CLI)

MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage9_SynTray"
PIX2PIX_DIR = Path("external/pix2pix")

TRAY_MASK_DIR = Path("data/interim/GAN/Empty_Tray_mask/Mask")

RAND_MAX_TRIES_PER_OBJ = 120
RAND_ALLOW_OVERLAP = False
RAND_SCALE_MIN, RAND_SCALE_MAX = 1.0, 1.0
RAND_ROT_MIN, RAND_ROT_MAX = 0.0, 25.0

# Fast placement defaults
X_SEARCH_STEP = 8
MAX_TRANSFORM_CANDIDATES = 12
Y_FALLBACK_OFFSETS = (0, -64, 64, -128, 128, -256, 256, -384, 384, -512, 512)
Y_SOFT_SEARCH_RADIUS = 320
Y_SOFT_SEARCH_STEP = 24

# Must match training palette exactly
PALETTE_BGR = {
    0: (0, 0, 0),
    1: (0, 255, 0),
}

# =============================================================================
# Training config mirror
# IMPORTANT: this checkpoint was trained with 5 channels:
# mask + edge + thickness + coord_x + coord_y
# =============================================================================
TRAIN_CFG = dict(
    input_nc=5,
    norm="instance",
    use_appearance_channel=False,
    use_tray_mask=False,
)


# =============================================================================
# Geometry helpers
# =============================================================================

def rotate_preserve_bgr(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - cx
    M[1, 2] += nh / 2 - cy
    return cv2.warpAffine(
        img,
        M,
        (nw, nh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


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
# Tray mask helpers
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


def dilate_bin(m01, px):
    if px <= 0:
        return m01.astype(np.uint8)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(m01.astype(np.uint8), k, iterations=1)


def preprocess_tray_mask(mask_gray, thr=0.5, invert=False, close_px=2, dilate_px=3):
    T = (mask_gray > int(np.clip(thr * 255, 0, 255))).astype(np.uint8)
    if invert:
        T = 1 - T
    T = largest_cc(T)
    T = morph_close(T, close_px)
    return dilate_bin(T, dilate_px).astype(bool)


def load_tray_masks(tray_dir: Path, thr=0.5, invert=False, close_px=2, dilate_px=3):
    paths = sorted(tray_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No tray mask PNGs in {tray_dir}")
    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (CANVAS_H, CANVAS_W):
            m = cv2.resize(m, (CANVAS_W, CANVAS_H), interpolation=cv2.INTER_NEAREST)
        masks.append(preprocess_tray_mask(m, thr, invert, close_px, dilate_px))
    if not masks:
        raise SystemExit(f"Could not read tray masks from {tray_dir}")
    print(f"[tray] loaded {len(masks)} masks")
    return masks


# =============================================================================
# Cutout library
# =============================================================================

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

            cat_id = ann.get("category_id")
            train_id = int(cat_to_train.get(cat_id, 1))
            color_bgr = np.array(PALETTE_BGR.get(train_id, (255, 255, 255)), dtype=np.uint8)

            mask_bgr = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            mask_bgr[mask_crop > 0] = color_bgr

            orig_w = int(x1 - x0 + 1)
            orig_h = int(y1 - y0 + 1)
            orig_cx = 0.5 * (x0 + x1)
            orig_cy = 0.5 * (y0 + y1)

            lib.append({
                "train_id": train_id,
                "class_name": cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}"),
                "image_id": img_id,
                "file_name": Path(im.get("file_name", "")).name,
                "mask_bin": mask_crop,
                "mask_bgr": mask_bgr,
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
            })

    if not lib:
        raise SystemExit("No valid cutouts from COCO + images_dir")

    print(f"[cutouts] built {len(lib)} instances")
    return lib


# =============================================================================
# Cutout transform
# =============================================================================

def transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max):
    base_w = int(item["mask_bgr"].shape[1])
    base_h = int(item["mask_bgr"].shape[0])

    mask_bin = (item["mask_bin"] > 127).astype(np.uint8) * 255
    gray = item["gray"].copy()

    s = rng.uniform(scale_min, scale_max)
    aug_w = max(1, int(round(base_w * s)))
    aug_h = max(1, int(round(base_h * s)))

    # Use a soft mask during resize/rotation so the contour is smoother.
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

    # Slight blur + close reduces staircase edges and tiny holes.
    mask_soft = cv2.GaussianBlur(mask_soft, (0, 0), 1.2)
    mask_u8 = np.clip(mask_soft, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Lower threshold keeps more of the rotated boundary.
    _, m = cv2.threshold(mask_u8, 96, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(m > 0)
    if not len(xs):
        return None

    dom = np.array(PALETTE_BGR.get(item["train_id"], (255, 255, 255)), dtype=np.uint8)
    clean_bgr = np.zeros((m.shape[0], m.shape[1], 3), dtype=np.uint8)
    clean_bgr[m > 0] = dom

    gray[m == 0] = 0

    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out_mask = clean_bgr[y1:y2, x1:x2].copy()
    out_gray = gray[y1:y2, x1:x2].copy()

    return {
        "mask_bgr": out_mask,
        "gray": out_gray,
        "h": out_mask.shape[0],
        "w": out_mask.shape[1],
    }


# =============================================================================
# Scene builder
# =============================================================================

def build_transform_candidates(item, rng, scale_min, scale_max, rot_min, rot_max, num_candidates):
    candidates = []
    for _ in range(num_candidates):
        t = transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max)
        if t is not None:
            candidates.append(t)
    return candidates


def build_candidate_xs(x_min, x_max, x_base, step, rng, randomize=True):
    candidate_xs = list(range(x_min, x_max + 1, step))
    x_base = max(x_min, min(x_max, x_base))
    candidate_xs.append(x_base)
    candidate_xs = sorted(set(candidate_xs), key=lambda xx: abs(xx - x_base))

    if randomize:
        chunks = [candidate_xs[i:i + 8] for i in range(0, len(candidate_xs), 8)]
        for chunk in chunks:
            rng.shuffle(chunk)
        candidate_xs = [x for chunk in chunks for x in chunk]

    return candidate_xs


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


def build_scene(rng, tray_masks, cutouts,
                n_min, n_max, allow_overlap, max_tries,
                scale_min, scale_max, rot_min, rot_max,
                use_real_appearance=True,
                horizontal_shift_only=True,
                max_horizontal_shift=200,
                require_all=True,
                y_fallback_offsets=Y_FALLBACK_OFFSETS,
                y_soft_search_radius=Y_SOFT_SEARCH_RADIUS,
                y_soft_search_step=Y_SOFT_SEARCH_STEP,
                x_search_step=X_SEARCH_STEP,
                max_transform_candidates=MAX_TRANSFORM_CANDIDATES,
                preserve_original_y_x_only=True,
                ignore_tray=True):
    if ignore_tray:
        tray = np.ones((CANVAS_H, CANVAS_W), dtype=bool)
    else:
        tray = rng.choice(tray_masks)
        if tray.shape[:2] != (CANVAS_H, CANVAS_W):
            tray = cv2.resize(
                tray.astype(np.uint8) * 255,
                (CANVAS_W, CANVAS_H),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

    canvas_mask = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas_app = np.zeros((CANVAS_H, CANVAS_W), dtype=np.uint8)
    occ = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)

    n_obj = rng.randint(n_min, n_max + 1) if n_min != n_max else n_min
    items = cutouts[:n_obj] if n_obj <= len(cutouts) else [rng.choice(cutouts) for _ in range(n_obj)]

    placed_count = 0
    for idx, item in enumerate(items):
        placed = False
        transformed_candidates = build_transform_candidates(
            item=item,
            rng=rng,
            scale_min=scale_min,
            scale_max=scale_max,
            rot_min=rot_min,
            rot_max=rot_max,
            num_candidates=max(1, min(max_tries, max_transform_candidates)),
        )

        for t in transformed_candidates:
            h, w = t["h"], t["w"]
            if h >= CANVAS_H or w >= CANVAS_W or h < 2 or w < 2:
                continue

            obj_mask = np.any(t["mask_bgr"] > 0, axis=2)

            y_base = int(round(float(item["orig_cy"]) - h / 2.0))
            y_base = max(0, min(CANVAS_H - h, y_base))

            if preserve_original_y_x_only:
                y_candidates = [y_base]
            elif horizontal_shift_only:
                y_candidates = build_soft_y_candidates(
                    canvas_h=CANVAS_H,
                    obj_h=h,
                    y_base=y_base,
                    radius=y_soft_search_radius,
                    step=y_soft_search_step,
                    rng=rng,
                )
                for dy in y_fallback_offsets:
                    yy = max(0, min(CANVAS_H - h, y_base + dy))
                    if yy not in y_candidates:
                        y_candidates.append(yy)
            else:
                row_step = max(4, x_search_step)
                y_candidates = list(range(0, CANVAS_H - h + 1, row_step))
                if y_base not in y_candidates:
                    y_candidates.append(y_base)
                rng.shuffle(y_candidates)

            x_base = max(0, min(CANVAS_W - w, int(round(float(item["orig_x0"])))))

            if preserve_original_y_x_only:
                x_min = 0
                x_max = CANVAS_W - w
                candidate_xs = build_candidate_xs(x_min, x_max, x_base, max(1, x_search_step), rng, True)
            elif horizontal_shift_only:
                x_base_scaled = int(round(float(item["orig_x0"]) * CANVAS_W / max(1, item["src_W"])))
                x_base_scaled = max(0, min(CANVAS_W - w, x_base_scaled))
                x_min = max(0, x_base_scaled - max_horizontal_shift)
                x_max = min(CANVAS_W - w, x_base_scaled + max_horizontal_shift)
                if x_min > x_max:
                    continue
                candidate_xs = build_candidate_xs(x_min, x_max, x_base_scaled, max(1, x_search_step), rng, True)
            else:
                candidate_xs = list(range(0, CANVAS_W - w + 1, max(1, x_search_step)))
                if x_base not in candidate_xs:
                    candidate_xs.append(x_base)
                rng.shuffle(candidate_xs)

            for y in y_candidates:
                for x in candidate_xs:
                    if not ignore_tray:
                        roi_tray = tray[y:y + h, x:x + w]
                        if roi_tray.shape[:2] != obj_mask.shape or not roi_tray.any():
                            continue
                        if not np.all(roi_tray[obj_mask]):
                            continue

                    roi_occ = occ[y:y + h, x:x + w]
                    if not allow_overlap and np.any(roi_occ & obj_mask):
                        continue

                    canvas_mask[y:y + h, x:x + w][obj_mask] = t["mask_bgr"][obj_mask]
                    if use_real_appearance:
                        canvas_app[y:y + h, x:x + w][obj_mask] = t["gray"][obj_mask]
                    occ[y:y + h, x:x + w][obj_mask] = True

                    placed = True
                    placed_count += 1
                    print(f"[place] item {idx + 1}/{len(items)} placed at x={x}, y={y}, w={w}, h={h}")
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            print(
                f"[place] item {idx+1}/{len(items)} FAILED | "
                f"file={item.get('file_name')} "
                f"orig_x0={item.get('orig_x0')} orig_y0={item.get('orig_y0')} "
                f"orig_w={item.get('orig_w')} orig_h={item.get('orig_h')}"
            )

    print(f"[place] placed {placed_count}/{len(items)} items")

    if require_all and placed_count < len(items):
        raise RuntimeError(
            f"Only placed {placed_count}/{len(items)} items. "
            f"Try increasing --canvas_w, --max_horizontal_shift, reducing rotation, or allowing overlap."
        )

    pseudo_B_bgr = cv2.cvtColor(canvas_app, cv2.COLOR_GRAY2BGR)
    return canvas_mask, canvas_app, pseudo_B_bgr, placed_count


# =============================================================================
# Normal single-pass test image writing
# =============================================================================

def clean_test_dir(test_dir: Path):
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.glob("*.png"):
        p.unlink()


def cleanup_intermediate_outputs(out_root: Path, results_root: Path, keep_preview=False):
    for path in [out_root / "test", results_root / "images"]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

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


def write_single_test_image(test_dir: Path, canvas_mask: np.ndarray, pseudo_B_bgr: np.ndarray, stem: str = "scene_0000"):
    clean_test_dir(test_dir)
    if canvas_mask.shape[:2] != (SIZE, SIZE):
        raise RuntimeError(
            f"Normal method requires a single-pass canvas of exactly {SIZE}x{SIZE}. "
            f"Got {canvas_mask.shape[1]}x{canvas_mask.shape[0]}."
        )
    if pseudo_B_bgr.shape[:2] != (SIZE, SIZE):
        raise RuntimeError(
            f"Pseudo-B must be exactly {SIZE}x{SIZE}. "
            f"Got {pseudo_B_bgr.shape[1]}x{pseudo_B_bgr.shape[0]}."
        )

    ab = np.concatenate([canvas_mask, pseudo_B_bgr], axis=1)
    out_path = test_dir / f"{stem}.png"
    cv2.imwrite(str(out_path), ab)
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
    """
    Export-only smoothing:
    1) upscale 2x with cubic interpolation
    2) apply a light Gaussian blur to soften staircase pixels
    This avoids destructive masking/cropping of the generated object.
    """
    h, w = img.shape[:2]
    up = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), 0.6)
    return up



# =============================================================================
# Pix2pix inference
# =============================================================================

def run_pix2pix_test(temp_dataset_dir, epoch="latest", use_display_mapper=True, num_test=None):
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
        f"--load_size={SIZE}",
        f"--crop_size={SIZE}",
        "--no_flip",
        f"--epoch={epoch}",
        "--eval",
        "--class_nc=1",
        "--thickness_nc=1",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--return_instance_masks",
        "--mask_thr=0.05",
    ]
    if num_test is not None:
        cmd.append(f"--num_test={num_test}")
    if cfg.get("use_appearance_channel", False):
        cmd += ["--appearance_nc=1", "--use_appearance_channel"]
    if not use_display_mapper:
        cmd.append("--no_use_display_mapper")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    results_dir = Path("results") / MODEL_NAME / f"test_{epoch}"
    print(f"\nResults: {results_dir.resolve()}")
    return results_dir


# =============================================================================
# Post-processing
# =============================================================================

def feather_from_canvas_mask(canvas_mask: np.ndarray, blur_sigma: float = 1.2, dilate_px: int = 1) -> np.ndarray:
    obj = (np.any(canvas_mask > 0, axis=2)).astype(np.uint8) * 255

    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        obj = cv2.dilate(obj, k, iterations=1)

    alpha = obj.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), blur_sigma)
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha

def smooth_generated_edges(fake_bgr: np.ndarray, canvas_mask: np.ndarray,
                           blur_sigma: float = 1.2,
                           dilate_px: int = 1,
                           do_supersample: bool = True) -> np.ndarray:

    alpha = feather_from_canvas_mask(canvas_mask, blur_sigma=blur_sigma, dilate_px=dilate_px)

    #FIX: match fake_B resolution
    h, w = fake_bgr.shape[:2]
    alpha = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)

    alpha3 = alpha[..., None]

    out = fake_bgr.astype(np.float32) * alpha3

    if do_supersample:
        up = cv2.resize(out, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        out = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def save_visuals(results_root: Path):
    for p in results_root.rglob("*_fake_B.png"):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(img_rgb).save(str(p.with_name(p.stem + "_rgb.png")))
        color = cv2.applyColorMap(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(p.with_name(p.stem + "_colormap.png")), color)
        print(f"Saved visuals for: {p.name}")


# =============================================================================
# Main
# =============================================================================

def main():
    global CANVAS_H, CANVAS_W

    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--coco_json", type=str, required=True)
    ap.add_argument("--classes", type=str, default="")
    ap.add_argument("--count", type=str, default="1")
    ap.add_argument("--seed", type=int, default=125)
    ap.add_argument("--out_dataset", type=str, default="datasets/_gen_real")
    ap.add_argument("--epoch", type=str, default="latest")
    ap.add_argument("--mode", type=str, default="random_mask", choices=["random_mask"])

    ap.add_argument("--canvas_h", type=int, default=CANVAS_H)
    ap.add_argument("--canvas_w", type=int, default=CANVAS_W)

    ap.add_argument("--rand_scale_min", type=float, default=RAND_SCALE_MIN)
    ap.add_argument("--rand_scale_max", type=float, default=RAND_SCALE_MAX)
    ap.add_argument("--rand_rot_min", type=float, default=RAND_ROT_MIN)
    ap.add_argument("--rand_rot_max", type=float, default=RAND_ROT_MAX)
    ap.add_argument("--rand_max_tries_per_obj", type=int, default=RAND_MAX_TRIES_PER_OBJ)
    ap.add_argument("--x_search_step", type=int, default=X_SEARCH_STEP)
    ap.add_argument("--max_transform_candidates", type=int, default=MAX_TRANSFORM_CANDIDATES)
    ap.add_argument("--no_overlap", action="store_true")
    ap.add_argument("--horizontal_shift_only", action="store_true")
    ap.add_argument("--max_horizontal_shift", type=int, default=200)
    ap.add_argument("--y_soft_search_radius", type=int, default=Y_SOFT_SEARCH_RADIUS)
    ap.add_argument("--y_soft_search_step", type=int, default=Y_SOFT_SEARCH_STEP)

    ap.add_argument("--tray_mask_dir", type=str, default=str(TRAY_MASK_DIR))
    ap.add_argument("--tray_mask_thr", type=float, default=0.5)
    ap.add_argument("--tray_mask_invert", action="store_true")
    ap.add_argument("--tray_cc_close_px", type=int, default=2)
    ap.add_argument("--tray_mask_dilate_px", type=int, default=3)

    ap.add_argument("--disable_test_appearance", action="store_true")
    ap.add_argument("--no_use_display_mapper", action="store_false", dest="use_display_mapper")
    ap.set_defaults(use_display_mapper=True)

    ap.add_argument("--preserve_original_y_x_only", action="store_true",
                    help="Keep each item at original-size/original-y and search only along x.")
    ap.add_argument("--ignore_tray", action="store_true",
                    help="Ignore tray-mask constraints during placement.")
    ap.add_argument("--keep_intermediates", action="store_true")
    ap.add_argument("--keep_preview", action="store_true")
    ap.add_argument("--skip_pix2pix", action="store_true",
                    help="Skip pix2pix and save the pseudo_B directly for placement debugging.")

    args = ap.parse_args()

    args.preserve_original_y_x_only = True
    args.ignore_tray = True

    CANVAS_H = int(args.canvas_h)
    CANVAS_W = int(args.canvas_w)

    if CANVAS_H != SIZE or CANVAS_W != SIZE:
        raise SystemExit(
            f"Normal single-pass method requires --canvas_h {SIZE} --canvas_w {SIZE}. "
            f"Got --canvas_h {CANVAS_H} --canvas_w {CANVAS_W}."
        )

    rng = random.Random(args.seed)

    images_dir = Path(args.images_dir)
    coco = json.loads(Path(args.coco_json).read_text())
    want_classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    out_root = Path(args.out_dataset)

    tray_masks = [] if args.ignore_tray else load_tray_masks(
        Path(args.tray_mask_dir),
        thr=args.tray_mask_thr,
        invert=args.tray_mask_invert,
        close_px=args.tray_cc_close_px,
        dilate_px=args.tray_mask_dilate_px,
    )

    allow_overlap = not args.no_overlap
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
            selected.extend(rng.sample(cands, n))
    else:
        selected = real_cutouts

    print("[selected]")
    for i, s in enumerate(selected, 1):
        print(
            f"  item {i}: class={s['class_name']} file={s['file_name']} "
            f"orig_x0={s['orig_x0']} orig_y0={s['orig_y0']} "
            f"orig_w={s['orig_w']} orig_h={s['orig_h']} src_W={s['src_W']} src_H={s['src_H']}"
        )

    canvas_mask, canvas_app, pseudo_B_bgr, placed_count = build_scene(
        rng=rng,
        tray_masks=tray_masks,
        cutouts=selected,
        n_min=len(selected),
        n_max=len(selected),
        allow_overlap=allow_overlap,
        max_tries=args.rand_max_tries_per_obj,
        scale_min=args.rand_scale_min,
        scale_max=args.rand_scale_max,
        rot_min=args.rand_rot_min,
        rot_max=args.rand_rot_max,
        use_real_appearance=(not args.disable_test_appearance),
        horizontal_shift_only=args.horizontal_shift_only,
        max_horizontal_shift=args.max_horizontal_shift,
        require_all=True,
        y_soft_search_radius=max(1, int(args.y_soft_search_radius)),
        y_soft_search_step=max(1, int(args.y_soft_search_step)),
        x_search_step=max(1, int(args.x_search_step)),
        max_transform_candidates=max(1, int(args.max_transform_candidates)),
        preserve_original_y_x_only=args.preserve_original_y_x_only,
        ignore_tray=args.ignore_tray,
    )

    print(f"[summary] requested {len(selected)} items, placed {placed_count}")
    print("[summary] normal single-pass mode enabled: no tiling, no ROI splitting, no stitching")

    if args.keep_preview or args.keep_intermediates:
        preview_dir = out_root / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(preview_dir / f"mask_seed{args.seed}.png"), canvas_mask)
        cv2.imwrite(str(preview_dir / f"pseudo_B_seed{args.seed}.png"), pseudo_B_bgr)

    test_dir = out_root / "test"
    stem = write_single_test_image(test_dir, canvas_mask, pseudo_B_bgr, stem="scene_0000")

    if args.skip_pix2pix:
        print("[summary] skip_pix2pix enabled: saving pseudo_B directly for debugging.")
        final_img = export_smooth_2x(pseudo_B_bgr.copy())
        results_root = None
    else:
        results_root = run_pix2pix_test(
            temp_dataset_dir=out_root,
            epoch=args.epoch,
            use_display_mapper=args.use_display_mapper,
            num_test=1,
        )

        if args.keep_intermediates:
            save_visuals(results_root)

        fake_path = resolve_fake_image_path(results_root / "images", stem)
        if fake_path is None:
            raise RuntimeError(f"Could not find fake_B output for stem '{stem}' in {results_root / 'images'}")

        final_img = cv2.imread(str(fake_path), cv2.IMREAD_COLOR)
        if final_img is None:
            raise RuntimeError(f"Could not read generated image: {fake_path}")

        final_img = export_smooth_2x(final_img)

    out_dir = out_root / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"generated_seed{args.seed}_smooth2x.png"
    cv2.imwrite(str(out_path), final_img)
    print(f"Saved generated result to: {out_path}")

    if (not args.keep_intermediates) and (results_root is not None):
        cleanup_intermediate_outputs(out_root, results_root, keep_preview=args.keep_preview)

    print(f"\nDone. Final generated result: {out_path}")


if __name__ == "__main__":
    main()
"""

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

python notebooks/Pix2Pix/generate_pix2pix.py \
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
  --rand_scale_min 1.0 \
  --rand_scale_max 1.0 \
  --rand_rot_min 0.0 \
  --rand_rot_max 2.0 \
  --rand_max_tries_per_obj 60 \
  --x_search_step 12 \
  --max_transform_candidates 8 \
  --horizontal_shift_only \
  --max_horizontal_shift 900 \
  --no_overlap

  

  TRAY
  python notebooks/Pix2Pix/generate_pix2pix.py \
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

  python notebooks/Pix2Pix/generate_pix2pixV2.py \
  --generate_mode tray \
  --tray_mask_dir data/interim/Empty/masks \
  --tray_mask_dilate_px 0 \
  --tray_cc_close_px 2 \
  --tray_mask_thr 0.5 \
  --seed 777 \
  --out_dataset datasets/_gen_tray_only \
  --epoch latest \
  --keep_intermediates


python notebooks/Pix2Pix/generate_pix2pix.py \
  --images_dir data/raw/Shampoo \
  --coco_json data/raw/Shampoo/result.json \
  --mode random_mask_od_ab \
  --classes Shampoo \
  --count 1 \
  --seed 100 \
  --out_dataset datasets/_gen_real \
  --epoch latest \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_dir data/interim/GAN/Empty/Mask \
  --ab_train_dir datasets/Shampoo/train \
  --empty_train_dir data/interim/GAN/Empty \
  --pseudo_lib_max 500 \
  --rand_scale_min 0.90 \
  --rand_scale_max 1.00 \
  --rand_rot_min 0.0 \
  --rand_rot_max 10.0 \
  --rand_max_tries_per_obj 500

  --disable_test_appearance


  To print out classes that you have:
  python - <<'PY'
    import json
    c=json.load(open("data/raw/Non-Contraband/result.json"))
    print(sorted([x["name"] for x in c["categories"]]))
    PY
"""