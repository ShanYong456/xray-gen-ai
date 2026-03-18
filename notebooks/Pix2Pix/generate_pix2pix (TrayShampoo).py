from pathlib import Path
import argparse, json, random, subprocess
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
SIZE = 1024
MODEL_NAME = "Shampoo_pix2pix_StructCond_V5_unguided"
PIX2PIX_DIR = Path("external/pix2pix")

TRAY_MASK_DIR = Path("data/interim/GAN/Empty_Tray_mask/Mask")
CUTOUT_DIR    = Path("data/raw/Shampoo/Cropped")

RAND_N_MIN, RAND_N_MAX     = 1, 2
RAND_MAX_TRIES_PER_OBJ     = 300
RAND_ALLOW_OVERLAP         = False
RAND_SCALE_MIN, RAND_SCALE_MAX = 0.85, 1.15   # match training
RAND_ROT_MIN,   RAND_ROT_MAX   = 0.0, 25.0    # match training

# Must match training palette exactly
PALETTE_BGR = {
    0: (0, 0, 0),
    1: (0, 255, 0),
}

# =============================================================================
# Training config mirror
# These values MUST match what was used during training.
# Edit here if you change training options.
# =============================================================================
TRAIN_CFG = dict(
    input_nc            = 9,       # mask+edge+thickness+coord_x+coord_y+appearance + E(3)
    norm                = "instance",
    delta_scale         = 0.8,
    delta_max           = 5.0,     # raise from 3 if objects were clipped
    delta_positive      = True,
    use_tray_mask       = True,
    tray_mask_path      = "data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png",
    tray_bbox_margin    = 2,
    tray_obj_dilate_px  = 5,
    tray_mask_dilate_px = 3,
    tray_nudge_iters    = 8,
    tray_nudge_max_step = 20,
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
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


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
        if m.shape != (SIZE, SIZE):
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        masks.append(preprocess_tray_mask(m, thr, invert, close_px, dilate_px))
    if not masks:
        raise SystemExit(f"Could not read tray masks from {tray_dir}")
    print(f"[tray] loaded {len(masks)} masks")
    return masks


# =============================================================================
# OD helpers
# =============================================================================

def rgb_to_od(img_rgb, eps=1e-6):
    return -np.log(np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0) + eps)


def od_to_rgb(od):
    return (np.clip(np.exp(-od), 0, 1) * 255).round().astype(np.uint8)


# =============================================================================
# Cutout library
# =============================================================================

def infer_train_id(bgr: np.ndarray) -> int:
    m = np.any(bgr > 0, axis=2)
    if not np.any(m):
        return 0
    pix = bgr[m].reshape(-1, 3)
    uniq, counts = np.unique(pix, axis=0, return_counts=True)
    dom = tuple(uniq[np.argmax(counts)].tolist())
    for tid, col in PALETTE_BGR.items():
        if tuple(col) == dom:
            return tid
    return 0


def train_id_to_bgr(train_id):
    return np.array(PALETTE_BGR.get(int(train_id), (255, 255, 255)), dtype=np.uint8)


def build_real_cutout_library(coco, images_dir: Path):
    """
    Build grayscale crops from real COCO annotations.
    These are used as appearance cues at inference time to close the
    train/test domain gap (model was trained on real B interiors).
    """
    images_by_id  = {im["id"]: im for im in coco.get("images", [])}
    cats          = coco.get("categories", [])
    cats_by_id    = {c["id"]: c for c in cats}
    sorted_cats   = sorted(cats, key=lambda c: c["id"])
    cat_to_train  = {c["id"]: i + 1 for i, c in enumerate(sorted_cats)}
    ann_by_img    = {}
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

            cat_id   = ann.get("category_id")
            train_id = int(cat_to_train.get(cat_id, 1))
            color_bgr = np.array(PALETTE_BGR.get(train_id, (255, 255, 255)), dtype=np.uint8)
            mask_bgr = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
            mask_bgr[mask_crop > 0] = color_bgr

            lib.append({
                "train_id": train_id,
                "class_name": cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}"),
                "mask_bin":  mask_crop,
                "mask_bgr":  mask_bgr,
                "gray":      gray_crop,
            })

    if not lib:
        raise SystemExit("No valid cutouts from COCO + images_dir")
    print(f"[cutouts] built {len(lib)} instances")
    return lib


# =============================================================================
# Cutout transform
# =============================================================================

def transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max):
    s = rng.uniform(scale_min, scale_max)
    mask_bgr = cv2.resize(item["mask_bgr"], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    mask_bin = cv2.resize(item["mask_bin"], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    gray     = cv2.resize(item["gray"],     None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

    ang = rng.uniform(rot_min, rot_max)
    mask_bgr = rotate_preserve_bgr(mask_bgr, ang)

    h0, w0 = mask_bin.shape[:2]
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
    M[0, 2] += nw / 2 - w0 / 2
    M[1, 2] += nh / 2 - h0 / 2
    mask_bin = cv2.warpAffine(mask_bin, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    gray     = cv2.warpAffine(gray,     M, (nw, nh), flags=cv2.INTER_LINEAR,  borderValue=0)

    m = cv2.GaussianBlur((mask_bin > 127).astype(np.uint8) * 255, (5, 5), 0.8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(m > 0)
    if not len(xs):
        return None

    # Recover dominant colour from rotated semantic mask for clean palette
    valid = np.any(mask_bgr > 0, axis=2)
    if valid.sum() == 0:
        return None
    pix = mask_bgr[valid].reshape(-1, 3)
    uniq, counts = np.unique(pix, axis=0, return_counts=True)
    dom = uniq[np.argmax(counts)].astype(np.uint8)
    clean_bgr = np.zeros_like(mask_bgr)
    clean_bgr[m > 0] = dom
    gray[m == 0] = 0

    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return {
        "mask_bgr": clean_bgr[y1:y2, x1:x2].copy(),
        "gray":     gray[y1:y2, x1:x2].copy(),
    }


# =============================================================================
# Scene builder
# Key fix: appearance channel is populated with REAL grayscale interior
# instead of empty-tray gray, matching what the model saw during training.
# =============================================================================

def build_scene(rng, tray_masks, cutouts, empty_bgr,
                n_min, n_max, allow_overlap, max_tries,
                scale_min, scale_max, rot_min, rot_max,
                use_real_appearance=True):
    """
    Returns:
        canvas_mask  : HxWx3 BGR semantic mask  (the A conditioning image)
        canvas_app   : HxW   grayscale           (the appearance channel cue)
        pseudo_B_bgr : HxWx3 BGR                 (used as B in the AB pair)

    Appearance fix:
        When use_real_appearance=True (default), the object interior grayscale
        from real training images is pasted into canvas_app.  This is then fed
        as the appearance channel to the model, matching the training distribution.
        When False, appearance is empty-tray gray (unguided inference).
    """
    tray = rng.choice(tray_masks)
    canvas_mask = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    empty_gray  = cv2.cvtColor(empty_bgr, cv2.COLOR_BGR2GRAY)
    canvas_app  = empty_gray.copy()
    occ         = np.zeros((SIZE, SIZE), dtype=bool)

    n_obj = rng.randint(n_min, n_max + 1) if n_min != n_max else n_min

    for item in cutouts[:n_obj] if n_obj <= len(cutouts) else (
            [rng.choice(cutouts) for _ in range(n_obj)]):

        for _global in range(max_tries):
            t = transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max)
            if t is None:
                continue
            h, w = t["mask_bgr"].shape[:2]
            if h >= SIZE or w >= SIZE or h < 2 or w < 2:
                continue
            obj_mask = np.any(t["mask_bgr"] > 0, axis=2)

            placed = False
            for _ in range(max_tries):
                x = rng.randint(0, SIZE - w)
                y = rng.randint(0, SIZE - h)
                if not np.all(tray[y:y + h, x:x + w][obj_mask]):
                    continue
                if not allow_overlap and np.any(occ[y:y + h, x:x + w] & obj_mask):
                    continue
                canvas_mask[y:y + h, x:x + w][obj_mask] = t["mask_bgr"][obj_mask]
                if use_real_appearance:
                    canvas_app[y:y + h, x:x + w][obj_mask] = t["gray"][obj_mask]
                occ[y:y + h, x:x + w][obj_mask] = True
                placed = True
                break

            if placed:
                break

    # pseudo_B: just the appearance canvas as a 3-channel grayscale
    # The model uses this + appearance channel to guide generation.
    pseudo_B_bgr = cv2.cvtColor(canvas_app, cv2.COLOR_GRAY2BGR)
    return canvas_mask, canvas_app, pseudo_B_bgr


# =============================================================================
# AB-pair pseudo-object library (OD-based, physically correct)
# =============================================================================

def split_AB(ab_path):
    AB = Image.open(str(ab_path)).convert("RGB")
    w, h = AB.size
    return AB.crop((0, 0, w // 2, h)), AB.crop((w // 2, 0, w, h))


def find_matching_empty(ab_path, empty_dir):
    import datetime
    def parse_ts(name):
        try:
            return datetime.datetime.strptime(
                Path(name).stem, "%Y-%m-%d_%H-%M-%S-%f").timestamp()
        except Exception:
            return None

    bname = Path(ab_path).name
    direct = empty_dir / bname
    if direct.exists():
        return direct

    ts_str = bname.split("-", 1)[1].split("_tr")[0] if "-" in bname else None
    if ts_str:
        cands = list(empty_dir.glob(f"*{ts_str}*"))
        if cands:
            return cands[0]
        target = parse_ts(ts_str)
        best, best_diff = None, None
        for emp in empty_dir.iterdir():
            sec = parse_ts(emp.stem)
            if sec is None or target is None:
                continue
            d = abs(sec - target)
            if best_diff is None or d < best_diff:
                best_diff, best = d, emp
        if best:
            return best
    return None


def build_pseudo_lib_from_ab(ab_dir, empty_dir, max_items=500):
    lib = []
    for ab_path in sorted(ab_dir.glob("*.png"))[:max_items]:
        try:
            A_img, B_img = split_AB(ab_path)
        except Exception:
            continue
        e_path = find_matching_empty(ab_path, empty_dir)
        if e_path is None:
            continue
        try:
            E_img = Image.open(str(e_path)).convert("RGB")
        except Exception:
            continue
        if E_img.size != A_img.size:
            E_img = E_img.resize(A_img.size, Image.BICUBIC)

        A_rgb = np.array(A_img).astype(np.uint8)
        B_rgb = np.array(B_img).astype(np.uint8)
        E_rgb = np.array(E_img).astype(np.uint8)

        # Infer mask — support both green and blue-encoded
        best_area, best_mask, train_id = 0, None, 0
        for col_rgb, tid in [([0, 255, 0], 1), ([0, 0, 255], 1)]:
            m = np.all(np.abs(A_rgb.astype(np.int16) - np.array(col_rgb)) <= 10, axis=2).astype(np.uint8)
            if m.sum() > best_area:
                best_area, best_mask, train_id = m.sum(), m, tid
        if train_id == 0 or best_mask.sum() < 50:
            continue

        ys, xs = np.where(best_mask > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask_crop = best_mask[y0:y1, x0:x1]
        delta = np.maximum(rgb_to_od(B_rgb) - rgb_to_od(E_rgb), 0.0)
        delta_crop = delta[y0:y1, x0:x1].copy()
        delta_crop[mask_crop == 0] = 0.0

        color_bgr = train_id_to_bgr(train_id)
        mask_bgr = np.zeros((*mask_crop.shape, 3), dtype=np.uint8)
        mask_bgr[mask_crop > 0] = color_bgr

        lib.append({"train_id": int(train_id), "mask_bin": mask_crop,
                    "mask_bgr": mask_bgr, "delta": delta_crop.astype(np.float32)})
    if not lib:
        raise SystemExit("No pseudo objects built from AB pairs.")
    print(f"[pseudo_lib] built {len(lib)} items")
    return lib


def transform_pseudo(item, rng, scale_min, scale_max, rot_min, rot_max):
    s = rng.uniform(scale_min, scale_max)
    ang = rng.uniform(rot_min, rot_max)
    mask_bgr = cv2.resize(item["mask_bgr"], None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    mask_bin = cv2.resize(item["mask_bin"] * 255, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    delta    = cv2.resize(item["delta"],    None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

    mask_bgr = rotate_preserve_bgr(mask_bgr, ang)
    h0, w0 = mask_bin.shape[:2]
    M = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
    M[0, 2] += nw / 2 - w0 / 2
    M[1, 2] += nh / 2 - h0 / 2
    mask_bin = cv2.warpAffine(mask_bin, M, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0)
    delta    = cv2.warpAffine(delta,    M, (nw, nh), flags=cv2.INTER_LINEAR,  borderValue=0)

    m = (mask_bin > 127).astype(np.uint8)
    if m.sum() < 20:
        return None
    ys, xs = np.where(m > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = m[y0:y1, x0:x1]
    delta = delta[y0:y1, x0:x1]
    delta[m == 0] = 0.0

    valid = np.any(mask_bgr > 0, axis=2)
    dom_color = mask_bgr[valid].reshape(-1, 3)[0] if valid.sum() > 0 else np.array([255, 0, 0])
    clean = np.zeros((*m.shape, 3), dtype=np.uint8)
    clean[m > 0] = dom_color

    return {"mask_bgr": clean, "mask_bin": m, "delta": delta, "train_id": item["train_id"]}


def build_scene_od(rng, tray_masks, pseudo_objects, empty_bgr,
                   allow_overlap, max_tries, scale_min, scale_max, rot_min, rot_max):
    tray = rng.choice(tray_masks)
    canvas_mask = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    occ = np.zeros((SIZE, SIZE), dtype=bool)
    E_rgb = cv2.cvtColor(empty_bgr, cv2.COLOR_BGR2RGB)
    OD_total = rgb_to_od(E_rgb).copy()

    for item in pseudo_objects:
        for _ in range(max_tries):
            t = transform_pseudo(item, rng, scale_min, scale_max, rot_min, rot_max)
            if t is None:
                continue
            h, w = t["mask_bin"].shape[:2]
            if h >= SIZE or w >= SIZE or h < 2 or w < 2:
                continue
            obj = t["mask_bin"] > 0
            for _ in range(max_tries):
                x = rng.randint(0, SIZE - w)
                y = rng.randint(0, SIZE - h)
                if not np.all(tray[y:y + h, x:x + w][obj]):
                    continue
                if not allow_overlap and np.any(occ[y:y + h, x:x + w] & obj):
                    continue
                canvas_mask[y:y + h, x:x + w][obj] = t["mask_bgr"][obj]
                for c in range(3):
                    OD_total[y:y + h, x:x + w, c][obj] += t["delta"][..., c][obj]
                occ[y:y + h, x:x + w][obj] = True
                break
            else:
                continue
            break

    B_rgb = od_to_rgb(OD_total)
    B_rgb[~tray] = E_rgb[~tray]

    # Apply same scanner noise as training synthetic samples
    noise = np.random.normal(0, np.random.uniform(1.0, 3.5), B_rgb.shape).astype(np.float32)
    B_rgb = np.clip(B_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    B_rgb = cv2.GaussianBlur(B_rgb, (0, 0), sigmaX=np.random.uniform(0.3, 0.8))

    return canvas_mask, cv2.cvtColor(B_rgb, cv2.COLOR_RGB2BGR)


# =============================================================================
# IO helpers
# =============================================================================

def clean_test_dir(test_dir: Path):
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.glob("*.png"):
        p.unlink()


def load_empty_bgr(empty_dir, empty_path):
    if empty_path:
        img = cv2.imread(empty_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read --empty_path: {empty_path}")
        return cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
    if empty_dir:
        cands = sorted(p for p in Path(empty_dir).glob("*")
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        if not cands:
            raise FileNotFoundError(f"No images in --empty_dir: {empty_dir}")
        img = cv2.imread(str(cands[0]))
        if img is None:
            raise FileNotFoundError(f"Cannot read: {cands[0]}")
        return cv2.resize(img, (SIZE, SIZE), cv2.INTER_AREA)
    raise ValueError("Provide --empty_dir or --empty_path")


# =============================================================================
# Pix2pix inference
# Key fix: all args mirror TRAIN_CFG exactly — no more hardcoded mismatches.
# =============================================================================

def run_pix2pix_test(temp_dataset_dir, epoch="latest",
                     empty_dir="", empty_path="",
                     use_display_mapper=True,
                     disable_test_appearance=False):
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
        "--num_test=1",
        f"--epoch={epoch}",
        "--eval",
        "--class_nc=1",
        "--thickness_nc=1",
        "--appearance_nc=1",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--use_appearance_channel",
        "--use_delta_comp",
        "--use_delta_prior",
        f"--prior_shampoo={1.1}",
        "--return_instance_masks",
        "--mask_thr=0.05",
        "--compose_eps=1e-6",
        "--od_gamma=1.0",
        f"--delta_scale={cfg['delta_scale']}",
        f"--delta_max={cfg['delta_max']}",
        "--use_tray_mask",
        f"--tray_mask_path={cfg['tray_mask_path']}",
        "--tray_mask_autoshift",
        f"--tray_bbox_margin={cfg['tray_bbox_margin']}",
        f"--tray_obj_dilate_px={cfg['tray_obj_dilate_px']}",
        f"--tray_mask_dilate_px={cfg['tray_mask_dilate_px']}",
        f"--tray_nudge_iters={cfg['tray_nudge_iters']}",
        f"--tray_nudge_max_step={cfg['tray_nudge_max_step']}",
    ]
    if cfg["delta_positive"]:
        cmd.append("--delta_positive")
    if empty_dir:
        cmd.append(f"--empty_dir={empty_dir}")
    if empty_path:
        cmd.append(f"--empty_path={empty_path}")
    if disable_test_appearance:
        cmd.append("--disable_test_appearance")
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--coco_json",  type=str, required=True)
    ap.add_argument("--classes",    type=str, default="")
    ap.add_argument("--count",      type=str, default="1")
    ap.add_argument("--seed",       type=int, default=125)
    ap.add_argument("--out_dataset",type=str, default="datasets/_gen_real")
    ap.add_argument("--epoch",      type=str, default="latest")
    ap.add_argument("--mode",       type=str, default="random_mask",
                    choices=["random_mask", "random_mask_od_ab"])

    # Tray mask
    ap.add_argument("--tray_mask_dir",    type=str, default=str(TRAY_MASK_DIR))
    ap.add_argument("--tray_mask_thr",    type=float, default=0.5)
    ap.add_argument("--tray_mask_invert", action="store_true")
    ap.add_argument("--tray_cc_close_px", type=int, default=2)
    ap.add_argument("--tray_mask_dilate_px", type=int, default=3)

    # Empty tray
    ap.add_argument("--empty_dir",  type=str, default="")
    ap.add_argument("--empty_path", type=str, default="")

    # Cutout placement
    ap.add_argument("--rand_scale_min",          type=float, default=RAND_SCALE_MIN)
    ap.add_argument("--rand_scale_max",          type=float, default=RAND_SCALE_MAX)
    ap.add_argument("--rand_rot_min",            type=float, default=RAND_ROT_MIN)
    ap.add_argument("--rand_rot_max",            type=float, default=RAND_ROT_MAX)
    ap.add_argument("--rand_max_tries_per_obj",  type=int,   default=RAND_MAX_TRIES_PER_OBJ)
    ap.add_argument("--no_overlap",              action="store_true")

    # Appearance
    ap.add_argument("--disable_test_appearance", action="store_true",
                    help="Zero-out appearance channel (fully unguided). "
                         "Default: use real grayscale interior (matches training).")

    # AB library (od_ab mode)
    ap.add_argument("--ab_train_dir",   type=str, default="datasets/Shampoo/train")
    ap.add_argument("--empty_train_dir",type=str, default="data/interim/GAN/Empty")
    ap.add_argument("--pseudo_lib_max", type=int, default=500)

    # Display mapper
    ap.add_argument("--no_use_display_mapper", action="store_false", dest="use_display_mapper")
    ap.set_defaults(use_display_mapper=True)

    args = ap.parse_args()
    rng  = random.Random(args.seed)

    images_dir   = Path(args.images_dir)
    coco         = json.loads(Path(args.coco_json).read_text())
    want_classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    out_root     = Path(args.out_dataset)

    tray_masks = load_tray_masks(
        Path(args.tray_mask_dir),
        thr=args.tray_mask_thr,
        invert=args.tray_mask_invert,
        close_px=args.tray_cc_close_px,
        dilate_px=args.tray_mask_dilate_px,
    )
    empty_bgr = load_empty_bgr(args.empty_dir, args.empty_path)

    allow_overlap = not args.no_overlap
    counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]

    # ── Build mask + pseudo-B ──────────────────────────────────────────────
    if args.mode == "random_mask":
        real_cutouts = build_real_cutout_library(coco, images_dir)

        if want_classes:
            targets = ([int(counts_raw[0])] * len(want_classes) if len(counts_raw) == 1
                       else [int(x) for x in counts_raw])
            selected = []
            for cls, n in zip(want_classes, targets):
                cands = [c for c in real_cutouts if c["class_name"].lower() == cls.lower()]
                if not cands:
                    avail = sorted({c["class_name"] for c in real_cutouts})
                    raise SystemExit(f"Class '{cls}' not found. Available: {avail}")
                selected.extend(rng.choice(cands) for _ in range(n))
        else:
            selected = real_cutouts

        canvas_mask, canvas_app, pseudo_B_bgr = build_scene(
            rng=rng, tray_masks=tray_masks, cutouts=selected,
            empty_bgr=empty_bgr,
            n_min=len(selected), n_max=len(selected),
            allow_overlap=allow_overlap,
            max_tries=args.rand_max_tries_per_obj,
            scale_min=args.rand_scale_min, scale_max=args.rand_scale_max,
            rot_min=args.rand_rot_min,     rot_max=args.rand_rot_max,
            use_real_appearance=(not args.disable_test_appearance),
        )

    elif args.mode == "random_mask_od_ab":
        pseudo_lib = build_pseudo_lib_from_ab(
            Path(args.ab_train_dir), Path(args.empty_train_dir),
            max_items=args.pseudo_lib_max)

        if want_classes:
            targets = ([int(counts_raw[0])] * len(want_classes) if len(counts_raw) == 1
                       else [int(x) for x in counts_raw])
            n_total = sum(targets)
        else:
            n_total = rng.randint(RAND_N_MIN, RAND_N_MAX)

        selected_pseudo = [rng.choice(pseudo_lib) for _ in range(n_total)]
        canvas_mask, pseudo_B_bgr = build_scene_od(
            rng=rng, tray_masks=tray_masks, pseudo_objects=selected_pseudo,
            empty_bgr=empty_bgr,
            allow_overlap=allow_overlap,
            max_tries=args.rand_max_tries_per_obj,
            scale_min=args.rand_scale_min, scale_max=args.rand_scale_max,
            rot_min=args.rand_rot_min,     rot_max=args.rand_rot_max,
        )
        canvas_app = cv2.cvtColor(pseudo_B_bgr, cv2.COLOR_BGR2GRAY)
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    # ── Save preview ───────────────────────────────────────────────────────
    preview_dir = out_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_dir / f"mask_seed{args.seed}.png"), canvas_mask)
    cv2.imwrite(str(preview_dir / f"pseudo_B_seed{args.seed}.png"), pseudo_B_bgr)

    # ── Write AB pair for pix2pix ─────────────────────────────────────────
    test_dir = out_root / "test"
    clean_test_dir(test_dir)
    tag = "_".join(want_classes) if want_classes else args.mode
    ab_path = test_dir / f"gen_{tag}_seed{args.seed}.png"

    # Copy empty to tmp dir so pix2pix --empty_dir can find it by filename
    tmp_empty = out_root / "empty_for_test"
    tmp_empty.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(tmp_empty / ab_path.name), empty_bgr)

    # B side of AB pair = pseudo_B so E-matching in dataset loader works
    AB_bgr = np.concatenate([canvas_mask, pseudo_B_bgr], axis=1)
    cv2.imwrite(str(ab_path), AB_bgr)
    print(f"Wrote AB: {ab_path}")

    # ── Run inference ──────────────────────────────────────────────────────
    results_root = run_pix2pix_test(
        temp_dataset_dir=out_root,
        epoch=args.epoch,
        empty_dir=str(tmp_empty),
        empty_path=args.empty_path,
        use_display_mapper=args.use_display_mapper,
        disable_test_appearance=args.disable_test_appearance,
    )

    save_visuals(results_root)
    print(f"\nDone. Results: {results_root}/images/")


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
  --seed 87 \
  --out_dataset datasets/_gen_real \
  --epoch latest \
  --empty_dir data/interim/GAN/Empty \
  --tray_mask_dir data/interim/GAN/Empty_Tray_mask/Mask \
  --rand_scale_min 0.85 \
  --rand_scale_max 1.15 \
  --rand_rot_min 0.0 \
  --rand_rot_max 25.0  \
  --rand_max_tries_per_obj 300 \
  --no_overlap --disable_test_appearance

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