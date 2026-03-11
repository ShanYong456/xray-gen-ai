from pathlib import Path
import argparse, json, random, subprocess
import cv2
import numpy as np

# Import halo remover
import sys
sys.path.insert(0, str(Path(__file__).parent))
try:
    from halo_remover import remove_halo_with_mask
    HAS_HALO_REMOVER = True
except ImportError:
    HAS_HALO_REMOVER = False

# =========================
# CONFIG YOU MUST MATCH
# =========================
SIZE = 1024  # must match your training load_size/crop_size
#MODEL_NAME = "contraband_metal_pix2pix_phys_fixDull_v4"   # your pix2pix --name
MODEL_NAME = "Shampoo_Blade_pix2pix_AppearanceV3"       # your pix2pix --name
PIX2PIX_DIR = Path("external/pix2pix")

# =========================
# NEW (ONLY WHAT'S NEEDED)
# =========================
# Folder of tray masks (white=tray, black=outside). Put multiple PNGs here.
TRAY_MASK_DIR = Path("data/interim/GAN/Empty_Tray_mask/Mask")

# Folder of colored cutouts (your object library). Each PNG is category-color filled.
#Contraband Metal
#CUTOUT_DIR = Path("data/raw/Contraband/Metal/Cropped")
#Non-Contraband
CUTOUT_DIR = Path("data/raw/Shampoo_Blade/Cropped")

# For random mask generation
RAND_N_MIN, RAND_N_MAX = 1, 3
RAND_MAX_TRIES_PER_OBJ = 300
RAND_ALLOW_OVERLAP = False
RAND_SCALE_MIN, RAND_SCALE_MAX = 0.6, 1.4
RAND_ROT_MIN, RAND_ROT_MAX = 0.0, 360.0

# IMPORTANT:
# Build the canvas in RGB (model conditioning), then convert to BGR ONLY when saving with cv2.imwrite().
# (In this script we keep masks in BGR consistently because OpenCV uses BGR.)
# Your palette MUST match whatever you used to generate training masks.
"""
# Contraband METAL:
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
"""
# Shampoo:
PALETTE_BGR = {
    0:  (0, 0, 0),
    1: (255, 0, 0),   
}
"""

# Shampoo_Blade:
PALETTE_BGR = {
    0:  (0, 0, 0),
    1: (0, 255, 0),       # green
    2: (255, 0, 0),       # blue
}


"""s
# Non-Contraband:
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


# -------------------------
# Geometry helpers
# -------------------------
def poly_to_pts_list(seg, W, H):
    pts_list = []
    if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
        polys = seg
    else:
        polys = [seg]

    for poly in polys:
        if not poly or len(poly) < 6:
            continue
        arr = np.array(poly, dtype=np.float32).reshape(-1, 2)

        # normalized polygons support (0..1)
        if arr[:, 0].max() <= 1.5 and arr[:, 1].max() <= 1.5:
            arr[:, 0] *= W
            arr[:, 1] *= H

        arr[:, 0] = np.clip(arr[:, 0], 0, W - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, H - 1)
        pts_list.append(arr.astype(np.int32))
    return pts_list


def rasterize_instance_mask(pts_list, W, H):
    m = np.zeros((H, W), dtype=np.uint8)
    for pts in pts_list:
        cv2.fillPoly(m, [pts], 255)
    return m


def tight_crop(mask_bin):
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return x0, y0, x1, y1


def vary_instance_mask(mask_bin, rng, scale_var=0.08, blur_sigma=0.5):
    """Subtle shape variation to avoid identical duplicates."""
    h, w = mask_bin.shape[:2]
    scale = rng.uniform(1.0 - scale_var, 1.0 + scale_var)
    new_h, new_w = max(2, int(h * scale)), max(2, int(w * scale))

    m = cv2.resize(mask_bin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if blur_sigma > 0:
        m = cv2.GaussianBlur(m, (3, 3), blur_sigma)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def add_contour_bgr(mask_bgr):
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    out = mask_bgr.copy()
    out[edges > 0] = (255, 255, 255)
    return out


# -------------------------
# random_mask mode (tray + colored cutouts)
# -------------------------
def load_tray_masks(tray_dir: Path):
    paths = sorted(list(tray_dir.glob("*.png")))
    if not paths:
        raise SystemExit(f"No tray mask PNGs found in {tray_dir}")

    masks = []
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (SIZE, SIZE):
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        masks.append(m > 127)

    if not masks:
        raise SystemExit(f"Could not read any tray masks from {tray_dir}")
    print(f"[random_mask] Loaded tray masks: {len(masks)} from {tray_dir}")
    return masks


def infer_train_id_from_cutout_bgr(cut_bgr: np.ndarray) -> int:
    """Cutouts are filled with category color (from PALETTE_BGR). Infer train_id by dominant non-zero color."""
    m = np.any(cut_bgr > 0, axis=2)
    if not np.any(m):
        return 0
    pix = cut_bgr[m].reshape(-1, 3)
    uniq, counts = np.unique(pix, axis=0, return_counts=True)
    bgr = tuple(uniq[np.argmax(counts)].tolist())
    for tid, col in PALETTE_BGR.items():
        if tuple(col) == bgr:
            return int(tid)
    return 0


def load_cutouts(cutout_root: Path):
    """
    Kept only for backward compatibility.
    Not used anymore once random_mask switches to REAL grayscale cutout library.
    """
    items = []
    for cls_dir in cutout_root.iterdir():
        if not cls_dir.is_dir():
            continue
        for p in cls_dir.glob("*.png"):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None or img.ndim != 3:
                continue

            if img.shape[2] == 4:
                bgr = img[:, :, :3].copy()
                a = img[:, :, 3] > 0
                bgr[~a] = 0
            else:
                bgr = img[:, :, :3].copy()

            tid = infer_train_id_from_cutout_bgr(bgr)
            if tid == 0:
                continue

            m = np.any(bgr > 0, axis=2)
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue

            y1, y2 = ys.min(), ys.max() + 1
            x1, x2 = xs.min(), xs.max() + 1
            bgr = bgr[y1:y2, x1:x2].copy()

            items.append({
                "bgr": bgr,
                "train_id": tid,
                "path": str(p),
            })

    if not items:
        print(f"[random_mask] Warning: no valid colored cutouts in {cutout_root}")
    else:
        print(f"[random_mask] Loaded colored cutouts: {len(items)} from {cutout_root}")
    return items

def rotate_preserve_bgr(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy
    out = cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return out


def transform_cutout(item, rng: random.Random,
                     scale_min: float, scale_max: float,
                     rot_min: float, rot_max: float):
    mask_bgr = item["mask_bgr"]
    mask_bin = item["mask_bin"]
    gray = item["gray"]

    s = rng.uniform(scale_min, scale_max)

    out_mask_bgr = cv2.resize(mask_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    out_mask_bin = cv2.resize(mask_bin, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    out_gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

    ang = rng.uniform(rot_min, rot_max)

    out_mask_bgr = rotate_preserve_bgr(out_mask_bgr, ang)

    h0, w0 = out_mask_bin.shape[:2]
    cx, cy = w0 / 2.0, h0 / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int(h0 * sin + w0 * cos)
    new_h = int(h0 * cos + w0 * sin)
    M[0, 2] += (new_w / 2) - cx
    M[1, 2] += (new_h / 2) - cy

    out_mask_bin = cv2.warpAffine(
        out_mask_bin, M, (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    out_gray = cv2.warpAffine(
        out_gray, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    m = (out_mask_bin > 127).astype(np.uint8) * 255
    m = cv2.GaussianBlur(m, (5, 5), 0.8)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    ys, xs = np.where(m > 0)
    if len(xs) == 0:
        return None

    # clean semantic mask using dominant class color from rotated semantic mask
    orig_mask = np.any(out_mask_bgr > 0, axis=2)
    pix = out_mask_bgr[orig_mask].reshape(-1, 3)
    uniq, counts = np.unique(pix, axis=0, return_counts=True)
    dom_color = uniq[np.argmax(counts)].astype(np.uint8)

    clean_bgr = np.zeros_like(out_mask_bgr)
    clean_bgr[m > 0] = dom_color

    out_gray[m == 0] = 0

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1

    return {
        "mask_bgr": clean_bgr[y1:y2, x1:x2].copy(),
        "gray": out_gray[y1:y2, x1:x2].copy(),
    }

def build_random_mask_and_app_canvas(
    rng: random.Random,
    tray_masks,
    cutouts,
    empty_bgr,
    n_min: int,
    n_max: int,
    allow_overlap: bool,
    max_tries_per_obj: int,
    scale_min: float,
    scale_max: float,
    rot_min: float,
    rot_max: float,
):
    tray = rng.choice(tray_masks)
    canvas_mask = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    empty_gray = cv2.cvtColor(empty_bgr, cv2.COLOR_BGR2GRAY)
    canvas_app = empty_gray.copy()

    occ = np.zeros((SIZE, SIZE), dtype=bool)

    # if cutouts already represent the exact requested objects, use them directly
    requested_items = list(cutouts)

    placed_summary = {}

    for item in requested_items:
        cls_name = item.get("class_name", f"id_{item.get('train_id', -1)}")
        placed = False

        for _global_try in range(max_tries_per_obj):
            transformed = transform_cutout(item, rng, scale_min, scale_max, rot_min, rot_max)
            if transformed is None:
                continue

            cut_mask = transformed["mask_bgr"]
            cut_gray = transformed["gray"]

            h, w = cut_mask.shape[:2]
            if h >= SIZE or w >= SIZE or h < 2 or w < 2:
                continue

            obj_mask = np.any(cut_mask > 0, axis=2)

            for _t in range(max_tries_per_obj):
                x = rng.randint(0, SIZE - w)
                y = rng.randint(0, SIZE - h)

                tray_region = tray[y:y+h, x:x+w]
                if not np.all(tray_region[obj_mask]):
                    continue

                if not allow_overlap:
                    if np.any(occ[y:y+h, x:x+w] & obj_mask):
                        continue

                region_mask = canvas_mask[y:y+h, x:x+w]
                region_app = canvas_app[y:y+h, x:x+w]

                region_mask[obj_mask] = cut_mask[obj_mask]
                region_app[obj_mask] = cut_gray[obj_mask]

                canvas_mask[y:y+h, x:x+w] = region_mask
                canvas_app[y:y+h, x:x+w] = region_app
                occ[y:y+h, x:x+w][obj_mask] = True

                placed = True
                placed_summary[cls_name] = placed_summary.get(cls_name, 0) + 1
                break

            if placed:
                break

        if not placed:
            print(f"[random_mask] WARNING: could not place requested object of class {cls_name}")

    pseudo_B_bgr = cv2.cvtColor(canvas_app, cv2.COLOR_GRAY2BGR)
    print(f"[random_mask] placed summary: {placed_summary}")
    return canvas_mask, pseudo_B_bgr


# -------------------------
# COCO parsing (for real_scene / paste modes)
# -------------------------
def build_indices(coco):
    images_by_id = {im["id"]: im for im in coco.get("images", [])}
    cats = coco.get("categories", [])
    cats_by_id = {c["id"]: c for c in cats}

    sorted_cats = sorted(cats, key=lambda c: c["id"])
    cat_id_to_train = {c["id"]: i + 1 for i, c in enumerate(sorted_cats)}
    cat_name_to_train = {c["name"]: cat_id_to_train[c["id"]] for c in sorted_cats}

    ann_by_img = {}
    skipped_rle = 0
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        seg = ann.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, dict):
            skipped_rle += 1
            continue
        ann_by_img.setdefault(img_id, []).append(ann)

    return images_by_id, cats_by_id, ann_by_img, cat_id_to_train, cat_name_to_train, skipped_rle

def build_real_cutout_library(coco, images_dir: Path):
    """
    Build a library of REAL grayscale object crops + masks from COCO polygons.
    This is what you want for training-like pseudo-B generation.
    """
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, _, skipped_rle = build_indices(coco)
    lib = []

    for img_id, anns in ann_by_img.items():
        im = images_by_id.get(img_id)
        if not im:
            continue

        file_name = Path(im.get("file_name", "")).name
        img_path = images_dir / file_name

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        H, W = img_gray.shape[:2]

        for ann in anns:
            seg = ann.get("segmentation")
            if not seg or isinstance(seg, dict):
                continue

            pts_list = poly_to_pts_list(seg, W, H)
            if not pts_list:
                continue

            inst = rasterize_instance_mask(pts_list, W, H)
            bbox = tight_crop(inst)
            if bbox is None:
                continue

            x0, y0, x1, y1 = bbox
            mask_crop = inst[y0:y1 + 1, x0:x1 + 1].copy()
            gray_crop = img_gray[y0:y1 + 1, x0:x1 + 1].copy()

            if mask_crop.size == 0:
                continue

            gray_crop[mask_crop == 0] = 0

            cat_id = ann.get("category_id")
            train_id = int(cat_id_to_train.get(cat_id, 1))
            class_name = cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}")

            # build semantic mask crop in palette color
            color_bgr = np.array(PALETTE_BGR.get(train_id, (255, 255, 255)), dtype=np.uint8)
            mask_bgr = np.zeros((mask_crop.shape[0], mask_crop.shape[1], 3), dtype=np.uint8)
            mask_bgr[mask_crop > 0] = color_bgr

            lib.append({
                "train_id": train_id,
                "class_name": class_name,
                "mask_bin": mask_crop,
                "mask_bgr": mask_bgr,
                "gray": gray_crop,
                "file_name": file_name,
            })
            
    if not lib:
        raise SystemExit("No valid REAL grayscale cutouts could be built from COCO + images_dir")

    print(f"[random_mask] Built real cutout library: {len(lib)} instances")
    return lib

def render_full_scene_mask(coco, images_dir: Path, rng, want_classes=None, edit_drop_prob=0.0, force_size=SIZE):
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, _, skipped_rle = build_indices(coco)

    valid_img_ids = [img_id for img_id, anns in ann_by_img.items() if len(anns) > 0]
    if not valid_img_ids:
        raise SystemExit("No polygon annotations found (or only RLE).")

    if want_classes:
        want_set = set(want_classes)
        filtered = []
        for img_id in valid_img_ids:
            ok = any(cats_by_id.get(a.get("category_id"), {}).get("name", "") in want_set
                     for a in ann_by_img.get(img_id, []))
            if ok:
                filtered.append(img_id)
        if filtered:
            valid_img_ids = filtered

    img_id = rng.choice(valid_img_ids)
    im = images_by_id[img_id]
    file_name = Path(im.get("file_name", "")).name
    img_path = images_dir / file_name

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image referenced by COCO: {img_path}")
    H, W = img.shape[:2]

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    for ann in ann_by_img.get(img_id, []):
        cat_id = ann.get("category_id")
        cname = cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}")

        if want_classes and cname not in want_classes:
            continue
        if edit_drop_prob > 0.0 and rng.random() < edit_drop_prob:
            continue

        seg = ann.get("segmentation")
        pts_list = poly_to_pts_list(seg, W, H)
        if not pts_list:
            continue

        inst = rasterize_instance_mask(pts_list, W, H)
        train_id = int(cat_id_to_train.get(cat_id, 1))
        color_bgr = PALETTE_BGR.get(train_id, (255, 255, 255))
        canvas[inst > 0] = color_bgr

    canvas_resized = cv2.resize(canvas, (force_size, force_size), interpolation=cv2.INTER_NEAREST)
    return canvas_resized, img_id, file_name


# -------------------------
# PASTE helpers (used by paste + real_scene_count)
# -------------------------
def apply_mask_style(m, morph="dilate", morph_k=3, morph_iter=1, soft_edges=False):
    if soft_edges:
        m = cv2.GaussianBlur(m, (3, 3), 0)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    if morph and morph.lower() != "none":
        k = max(1, int(morph_k))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        it = max(1, int(morph_iter))
        if morph.lower() == "dilate":
            m = cv2.dilate(m, kernel, iterations=it)
        elif morph.lower() == "erode":
            m = cv2.erode(m, kernel, iterations=it)
    return m


def build_shape_library_for_paste(coco, images_dir: Path):
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, cat_name_to_train, skipped_rle = build_indices(coco)
    lib = {}
    for img_id, anns in ann_by_img.items():
        im = images_by_id.get(img_id)
        if not im:
            continue
        file_name = Path(im.get("file_name", "")).name
        img_path = images_dir / file_name
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        for ann in anns:
            seg = ann.get("segmentation")
            pts_list = poly_to_pts_list(seg, W, H)
            if not pts_list:
                continue
            inst = rasterize_instance_mask(pts_list, W, H)
            bbox = tight_crop(inst)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            crop = inst[y0:y1 + 1, x0:x1 + 1]
            if crop.size == 0:
                continue
            cat_id = ann.get("category_id")
            train_id = int(cat_id_to_train.get(cat_id, 1))
            cat_name = cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}")
            lib.setdefault(cat_name, []).append({"train_id": train_id, "mask": crop})
    return lib, cat_id_to_train, cat_name_to_train, skipped_rle


def place_instance(canvas_bgr, occ_mask, inst_mask_bin, color_bgr, rng,
                   scale_range=(0.95, 1.05), rot_deg=0.0, max_tries=80,
                   allow_overlap=False, morph="dilate", morph_k=3, morph_iter=1, soft_edges=False,
                   blend_width=0):
    Hc, Wc = canvas_bgr.shape[:2]
    h, w = inst_mask_bin.shape[:2]

    for _ in range(max_tries):
        scale = rng.uniform(*scale_range)
        angle = rng.uniform(-rot_deg, rot_deg) if rot_deg > 0 else 0.0

        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        if new_w >= Wc or new_h >= Hc:
            continue

        m = cv2.resize(inst_mask_bin, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        if angle != 0.0:
            M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
            m = cv2.warpAffine(
                m, M, (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        m = apply_mask_style(m, morph=morph, morph_k=morph_k, morph_iter=morph_iter, soft_edges=soft_edges)

        ys, xs = np.where(m > 0)
        if len(xs) < 50:
            continue

        x0 = rng.randint(0, Wc - new_w)
        y0 = rng.randint(0, Hc - new_h)

        if not allow_overlap:
            occ_region = occ_mask[y0:y0 + new_h, x0:x0 + new_w]
            if np.any((occ_region > 0) & (m > 0)):
                continue

        region = canvas_bgr[y0:y0 + new_h, x0:x0 + new_w].copy()
        region[m > 0] = np.array(color_bgr, dtype=np.uint8)
        canvas_bgr[y0:y0 + new_h, x0:x0 + new_w] = region

        occ_region = occ_mask[y0:y0 + new_h, x0:x0 + new_w]
        occ_region[m > 0] = 255
        occ_mask[y0:y0 + new_h, x0:x0 + new_w] = occ_region

        return True

    return False


def count_components_for_train_id(canvas_bgr, train_id, min_area=50):
    color = np.array(PALETTE_BGR.get(int(train_id), (255, 255, 255)), dtype=np.uint8)
    m = cv2.inRange(canvas_bgr, color, color)
    m = (m > 0).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return int(np.sum(areas >= int(min_area)))


# -------------------------
# real_scene_count (same as your logic, kept)
# -------------------------
def get_instances_for_image(coco, images_dir: Path, img_id):
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, _, skipped_rle = build_indices(coco)
    im = images_by_id.get(img_id)
    if im is None:
        raise SystemExit(f"image_id {img_id} not found in COCO")
    file_name = Path(im.get("file_name", "")).name
    img_path = images_dir / file_name
    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")
    H, W = img.shape[:2]

    instances = []
    for ann in ann_by_img.get(img_id, []):
        seg = ann.get("segmentation")
        if not seg or isinstance(seg, dict):
            continue
        cat_id = ann.get("category_id")
        cname = cats_by_id.get(cat_id, {}).get("name", f"class_{cat_id}")
        pts_list = poly_to_pts_list(seg, W, H)
        if not pts_list:
            continue
        inst = rasterize_instance_mask(pts_list, W, H)
        train_id = int(cat_id_to_train.get(cat_id, 1))
        instances.append({"train_id": train_id, "cname": cname, "mask": inst})
    return img, instances, file_name


def render_from_instances(inst_list, size=SIZE):
    out = np.zeros((size, size, 3), dtype=np.uint8)
    for it in inst_list:
        color = PALETTE_BGR.get(int(it["train_id"]), (255, 255, 255))
        out[it["mask"] > 0] = color
    return out


def choose_best_real_image_for_targets(coco, want_classes, targets, rng, sample_k=200):
    images_by_id, cats_by_id, ann_by_img, _, _, _ = build_indices(coco)
    valid_img_ids = [img_id for img_id, anns in ann_by_img.items() if len(anns) > 0]
    if not valid_img_ids:
        raise SystemExit("No polygon annotations found (or only RLE).")

    want_set = set(want_classes)
    candidates = []
    for img_id in valid_img_ids:
        if any(cats_by_id.get(a.get("category_id"), {}).get("name", "") in want_set
               for a in ann_by_img.get(img_id, [])):
            candidates.append(img_id)
    if not candidates:
        candidates = valid_img_ids

    rng.shuffle(candidates)
    candidates = candidates[:max(1, min(len(candidates), int(sample_k)))]

    best = None
    best_score = None
    for img_id in candidates:
        counts = {c: 0 for c in want_classes}
        for ann in ann_by_img.get(img_id, []):
            cname = cats_by_id.get(ann.get("category_id"), {}).get("name", "")
            if cname in counts:
                seg = ann.get("segmentation")
                if seg and not isinstance(seg, dict):
                    counts[cname] += 1

        score = 0
        for c, t in zip(want_classes, targets):
            score += abs(counts.get(c, 0) - int(t))

        if best_score is None or score < best_score:
            best_score = score
            best = img_id

    return best


def build_real_scene_count_mask(
    coco,
    images_dir: Path,
    rng,
    want_classes,
    targets,
    keep_others=True,
    force_size=SIZE,
    scale_min=0.85,
    scale_max=1.15,
    rot_deg=0.0,
    morph="none",
    morph_k=3,
    morph_iter=1,
    soft_edges=False,
    min_comp_area=50,
    max_add_tries=1200,
    sample_k=200,
    var_scale=0.10,
    var_blur=0.8,
):
    if want_classes is None or targets is None:
        raise SystemExit("real_scene_count requires --classes and --count.")

    img_id = choose_best_real_image_for_targets(coco, want_classes, targets, rng, sample_k=sample_k)
    base_img, inst_full, fname = get_instances_for_image(coco, images_dir, img_id)

    tray_mask = np.any(base_img > 0, axis=2)
    tray_mask = cv2.resize(tray_mask.astype(np.uint8), (force_size, force_size), interpolation=cv2.INTER_NEAREST).astype(bool)

    insts = []
    for d in inst_full:
        m = cv2.resize(d["mask"], (force_size, force_size), interpolation=cv2.INTER_NEAREST)
        insts.append({"train_id": d["train_id"], "cname": d["cname"], "mask": m})

    want_set = set(want_classes)
    controllable = [it for it in insts if it["cname"] in want_set]
    others = [it for it in insts if it["cname"] not in want_set] if keep_others else []

    # keep closest matches first
    new_ctrl = []
    for cls_name, target in zip(want_classes, targets):
        cls_insts = [it for it in controllable if it["cname"] == cls_name]
        cls_insts.sort(key=lambda x: -np.sum(x["mask"] > 0))
        new_ctrl.extend(cls_insts[:max(0, int(target))])

    canvas_bgr = render_from_instances(others + new_ctrl, size=force_size)
    occ_mask = np.zeros((force_size, force_size), dtype=np.uint8)
    occ_mask[np.any(canvas_bgr > 0, axis=2)] = 255

    lib, _, _, _ = build_shape_library_for_paste(coco, images_dir)

    # add missing
    for cls_name, target in zip(want_classes, targets):
        target = int(target)
        if cls_name not in lib or len(lib[cls_name]) == 0:
            raise SystemExit(f"Class '{cls_name}' not found in COCO.")
        train_id = int(lib[cls_name][0]["train_id"])
        cur = count_components_for_train_id(canvas_bgr, train_id, min_area=min_comp_area)

        tries = 0
        num_added = 0

        while cur < target and tries < max_add_tries:
            tries += 1
            canvas_backup = canvas_bgr.copy()
            occ_backup = occ_mask.copy()

            inst = rng.choice(lib[cls_name])
            inst_mask = inst["mask"].copy()

            if num_added > 0:
                inst_mask = vary_instance_mask(inst_mask, rng, scale_var=var_scale, blur_sigma=var_blur)
                if rng.random() < 0.5:
                    k = rng.choice([3, 5])
                    inst_mask = cv2.erode(inst_mask, np.ones((k, k), np.uint8), 1)

            color_bgr = PALETTE_BGR.get(train_id, (255, 255, 255))

            ok = place_instance(
                canvas_bgr=canvas_bgr,
                occ_mask=occ_mask,
                inst_mask_bin=inst_mask,
                color_bgr=color_bgr,
                rng=rng,
                scale_range=(float(scale_min), float(scale_max)),
                rot_deg=float(rot_deg),
                allow_overlap=False,
                morph=morph,
                morph_k=morph_k,
                morph_iter=morph_iter,
                soft_edges=soft_edges,
                blend_width=0,
                max_tries=140,
            )
            if not ok:
                canvas_bgr[:] = canvas_backup
                occ_mask[:] = occ_backup
                continue

            mask_region = np.any(canvas_bgr > 0, axis=2)
            if not np.all(tray_mask[mask_region]):
                canvas_bgr[:] = canvas_backup
                occ_mask[:] = occ_backup
                continue

            new_cur = count_components_for_train_id(canvas_bgr, train_id, min_area=min_comp_area)
            if new_cur <= cur:
                canvas_bgr[:] = canvas_backup
                occ_mask[:] = occ_backup
                continue

            cur = new_cur
            num_added += 1

        if cur < target:
            print(f"[real_scene_count] Could not reach target for {cls_name}. Got {cur}, wanted {target}.")

    return canvas_bgr, img_id, fname


# -------------------------
# IO + pix2pix
# -------------------------
def clean_test_dir(test_dir: Path):
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.glob("*.png"):
        p.unlink()


def load_empty_tray_bgr(empty_dir: str, empty_path: str) -> np.ndarray:
    if empty_path:
        p = Path(empty_path)
        if not p.exists():
            raise FileNotFoundError(f"--empty_path not found: {p}")
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(f"Could not read --empty_path: {p}")
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    if empty_dir:
        d = Path(empty_dir)
        if not d.exists():
            raise FileNotFoundError(f"--empty_dir not found: {d}")
        cands = sorted([p for p in d.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}])
        if not cands:
            raise FileNotFoundError(f"No images found in --empty_dir: {d}")
        img = cv2.imread(str(cands[0]))
        if img is None:
            raise FileNotFoundError(f"Could not read empty tray image: {cands[0]}")
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    raise ValueError("use_delta_comp requires --empty_dir or --empty_path")


def run_pix2pix_test(
    temp_dataset_dir: Path,
    epoch: str = "latest",
    norm: str = "instance",
    use_delta_comp: bool = False,
    empty_dir: str = "",
    empty_path: str = "",
    use_display_mapper: bool = True,
    use_soft_mask: bool = False,
    mask_blur_ksize: int = 0,
    mask_soft_beta: float = 30.0,
):
    # Your trained model uses input_nc=6 when delta-comp is enabled (A=3 + E=3).
    input_nc = 6 if use_delta_comp else 3

    cmd = [
        "python", str(PIX2PIX_DIR / "test.py"),
        f"--dataroot={temp_dataset_dir}",
        f"--name={MODEL_NAME}",
        "--model=pix2pix",
        "--dataset_mode=aligned",
        "--direction=AtoB",
        f"--input_nc={input_nc}",
        "--output_nc=3",
        "--netG=unet_256",
        f"--norm={norm}",
        "--preprocess=none",
        f"--load_size={SIZE}",
        f"--crop_size={SIZE}",
        "--no_flip",
        "--num_test=1",
        f"--epoch={epoch}",
        "--eval",

          # must match training
        "--class_nc=1",
        "--appearance_nc=1",
        "--thickness_nc=1",
        "--use_appearance_channel",
        "--use_thickness_channel",
    ]

    if use_delta_comp:
        cmd += [
            "--use_delta_comp",
            "--delta_positive",
            "--delta_scale", "0.35",
            "--delta_max", "3",
            "--use_tray_mask",
            "--tray_mask_path=data/interim/GAN/Empty/Mask/2026-01-21_10-36-28-447_traymask.png",
            "--tray_mask_autoshift",
            "--tray_bbox_margin", "2",
            "--tray_obj_dilate_px", "5",
            "--tray_mask_dilate_px", "3",
            "--tray_nudge_iters", "8",
            "--tray_nudge_max_step", "20",
        ]

        if empty_dir:
            cmd.append(f"--empty_dir={empty_dir}")
        if empty_path:
            cmd.append(f"--empty_path={empty_path}")

    # IMPORTANT: Pass mask settings to match training
    if use_soft_mask:
        cmd.append("--use_soft_mask")
    cmd.append(f"--mask_blur_ksize={mask_blur_ksize}")
    cmd.append(f"--mask_blur_sigma=1.2")
    cmd.append(f"--mask_soft_beta={mask_soft_beta}")

    # If your checkpoint DOES NOT have net_display_mapper, disable it.
    if not use_display_mapper:
        cmd.append("--no_use_display_mapper")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    results_dir = Path("results") / MODEL_NAME / f"test_{epoch}"
    images_dir = results_dir / "images"
    print("\nPix2Pix results folder:", results_dir.resolve())
    print("Look for generated image (fake_B) in:", images_dir.resolve())
    print("   Files are like: *_fake_B.png")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--images_dir", type=str, required=True)
    ap.add_argument("--coco_json", type=str, required=True)

    ap.add_argument("--classes", type=str, default="")
    ap.add_argument("--count", type=str, default="1")
    ap.add_argument("--seed", type=int, default=125)

    ap.add_argument("--out_dataset", type=str, default="datasets/_gen_real")
    ap.add_argument("--epoch", type=str, default="latest")

    ap.add_argument("--add_contour", action="store_true")
    ap.add_argument("--no_overlap", action="store_true")

    ap.add_argument("--mode", type=str, default="real_scene",
                    choices=["real_scene", "paste", "real_scene_count", "random_mask"])

    # paste / real_scene_count knobs
    ap.add_argument("--scale_min", type=float, default=0.98)
    ap.add_argument("--scale_max", type=float, default=1.02)
    ap.add_argument("--rot_deg", type=float, default=0.0)
    ap.add_argument("--morph", type=str, default="none", choices=["dilate", "erode", "none"])
    ap.add_argument("--morph_k", type=int, default=1)
    ap.add_argument("--morph_iter", type=int, default=1)
    ap.add_argument("--soft_edges", action="store_true")
    ap.add_argument("--min_comp_area", type=int, default=50)
    ap.add_argument("--edit_drop_prob", type=float, default=0.0)
    ap.add_argument("--keep_others", action="store_true")
    ap.add_argument("--base_sample_k", type=int, default=200)
    ap.add_argument("--max_add_tries", type=int, default=1200)
    ap.add_argument("--var_scale", type=float, default=0.10)
    ap.add_argument("--var_blur", type=float, default=0.8)

    # delta-comp inference
    ap.add_argument("--use_delta_comp", action="store_true")
    ap.add_argument("--empty_dir", type=str, default="")
    ap.add_argument("--empty_path", type=str, default="")

    # Mask settings (must match training for consistent results)
    ap.add_argument("--use_soft_mask", action="store_true", help="Use soft mask (must match training)")
    ap.add_argument("--mask_blur_ksize", type=int, default=0, help="Gaussian blur kernel for mask (must match training)")
    ap.add_argument("--mask_soft_beta", type=float, default=30.0, help="Soft mask sharpness (must match training)")

    # Halo removal (post-processing fix)
    ap.add_argument("--remove_halo", action="store_true", help="Remove white halos from output")
    ap.add_argument("--halo_erode_px", type=int, default=5, help="Pixels to erode mask for halo removal (larger = more removal)")
    ap.add_argument("--halo_blend_width", type=int, default=8, help="Blend width for smooth halo removal transition")

    ap.add_argument("--norm", type=str, default="batch", choices=["batch", "instance", "none"])

    # random_mask knobs
    ap.add_argument("--tray_mask_dir", type=str, default=str(TRAY_MASK_DIR))
    ap.add_argument("--cutout_dir", type=str, default=str(CUTOUT_DIR))
    ap.add_argument("--rand_n_min", type=int, default=1)
    ap.add_argument("--rand_n_max", type=int, default=3)
    ap.add_argument("--rand_max_tries_per_obj", type=int, default=300)
    ap.add_argument("--rand_scale_min", type=float, default=0.6)
    ap.add_argument("--rand_scale_max", type=float, default=1.4)
    ap.add_argument("--rand_rot_min", type=float, default=0.0)
    ap.add_argument("--rand_rot_max", type=float, default=360.0)

    # allow disabling display mapper if checkpoint doesn't have it
    ap.add_argument("--no_use_display_mapper", action="store_false", dest="use_display_mapper",
                    help="disable display mapper (do not load *_net_display_mapper.pth)")
    ap.set_defaults(use_display_mapper=True)

    args = ap.parse_args()
    rng = random.Random(args.seed)

    images_dir = Path(args.images_dir)
    coco = json.loads(Path(args.coco_json).read_text())

    want_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    want_classes = want_classes if len(want_classes) > 0 else None

    out_root = Path(args.out_dataset)

    # -------------------------
    # BUILD MASK A (canvas_bgr)
    # -------------------------
    if args.mode == "real_scene":
        canvas_bgr, img_id, fname = render_full_scene_mask(
            coco=coco,
            images_dir=images_dir,
            rng=rng,
            want_classes=want_classes,
            edit_drop_prob=float(args.edit_drop_prob),
            force_size=SIZE,
        )
        print(f"real_scene picked image_id={img_id}, file={fname}")

    elif args.mode == "real_scene_count":
        if not want_classes:
            raise SystemExit("--mode real_scene_count requires --classes.")
        counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]
        if len(counts_raw) == 1:
            targets = [int(counts_raw[0])] * len(want_classes)
        elif len(counts_raw) == len(want_classes):
            targets = [int(x) for x in counts_raw]
        else:
            raise SystemExit("--count must be 1 number or same length as --classes")

        canvas_bgr, img_id, fname = build_real_scene_count_mask(
            coco=coco,
            images_dir=images_dir,
            rng=rng,
            want_classes=want_classes,
            targets=targets,
            keep_others=bool(args.keep_others),
            force_size=SIZE,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            rot_deg=args.rot_deg,
            morph=args.morph,
            morph_k=args.morph_k,
            morph_iter=args.morph_iter,
            soft_edges=args.soft_edges,
            min_comp_area=int(args.min_comp_area),
            max_add_tries=int(args.max_add_tries),
            sample_k=int(args.base_sample_k),
            var_scale=args.var_scale,
            var_blur=args.var_blur,
        )
        print(f"real_scene_count picked image_id={img_id}, file={fname}")

    elif args.mode == "random_mask":

        tray_masks = load_tray_masks(Path(args.tray_mask_dir))

        real_cutouts = build_real_cutout_library(coco, images_dir)

        allow_overlap = False if args.no_overlap else True

        # -------------------------
        # Build class filtered library
        # -------------------------
        if want_classes is not None:
            class_map = {}
            for item in real_cutouts:
                tid = item["train_id"]
                class_map.setdefault(tid, []).append(item)

            counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]

            if len(counts_raw) == 1:
                targets = [int(counts_raw[0])] * len(want_classes)
            elif len(counts_raw) == len(want_classes):
                targets = [int(x) for x in counts_raw]
            else:
                raise SystemExit("--count must match --classes")

            cutouts = []
            for cls, n in zip(want_classes, targets):
                candidates = [c for c in real_cutouts if c["class_name"].lower() == cls.lower()]

                if len(candidates) == 0:
                    avail = sorted({c["class_name"] for c in real_cutouts})
                    raise SystemExit(f"class {cls} not found. Available classes: {avail}")

                for _ in range(n):
                    cutouts.append(rng.choice(candidates))

        else:
            cutouts = real_cutouts

        if args.use_delta_comp:
            empty_bgr_for_scene = load_empty_tray_bgr(args.empty_dir, args.empty_path)
        else:
            empty_bgr_for_scene = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

        canvas_bgr, pseudo_B_bgr = build_random_mask_and_app_canvas(
            rng=rng,
            tray_masks=tray_masks,
            cutouts=cutouts,
            empty_bgr=empty_bgr_for_scene,
            n_min=len(cutouts) if want_classes else args.rand_n_min,
            n_max=len(cutouts) if want_classes else args.rand_n_max,
            allow_overlap=allow_overlap,
            max_tries_per_obj=args.rand_max_tries_per_obj,
            scale_min=args.rand_scale_min,
            scale_max=args.rand_scale_max,
            rot_min=args.rand_rot_min,
            rot_max=args.rand_rot_max,
        )

        img_id, fname = -1, "random_mask"
        print("random_mask: generated controlled synthetic scene")

    elif args.mode == "paste":
        if not want_classes:
            raise SystemExit("--mode paste requires --classes.")

        counts_raw = [x.strip() for x in args.count.split(",") if x.strip()]
        if len(counts_raw) == 1:
            targets = [int(counts_raw[0])] * len(want_classes)
        elif len(counts_raw) == len(want_classes):
            targets = [int(x) for x in counts_raw]
        else:
            raise SystemExit("--count must be 1 number or same length as --classes")

        lib, _, _, skipped_rle = build_shape_library_for_paste(coco, images_dir)
        if skipped_rle > 0:
            print(f"[paste] Skipped {skipped_rle} RLE annotations (polygons only supported).")

        canvas_bgr = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
        occ_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)

        allow_overlap = not args.no_overlap

        for cls_name, target in zip(want_classes, targets):
            if cls_name not in lib or len(lib[cls_name]) == 0:
                raise SystemExit(f"Class '{cls_name}' not found in dataset or empty.")

            train_id = lib[cls_name][0]["train_id"]
            color_bgr = PALETTE_BGR.get(train_id, (255, 255, 255))

            placed = 0
            tries = 0
            while placed < int(target) and tries < 2000:
                tries += 1
                inst = rng.choice(lib[cls_name])
                inst_mask = inst["mask"].copy()
                if placed > 0:
                    inst_mask = vary_instance_mask(inst_mask, rng, scale_var=args.var_scale, blur_sigma=args.var_blur)

                ok = place_instance(
                    canvas_bgr=canvas_bgr,
                    occ_mask=occ_mask,
                    inst_mask_bin=inst_mask,
                    color_bgr=color_bgr,
                    rng=rng,
                    scale_range=(float(args.scale_min), float(args.scale_max)),
                    rot_deg=float(args.rot_deg),
                    allow_overlap=allow_overlap,
                    morph=args.morph,
                    morph_k=args.morph_k,
                    morph_iter=args.morph_iter,
                    soft_edges=args.soft_edges,
                    blend_width=5,
                    max_tries=120,
                )
                if ok:
                    placed += 1

            if placed < int(target):
                print(f"[paste] Could not reach target for {cls_name}. Placed {placed}/{target}")

        img_id, fname = -1, "paste_synthetic"
        print("paste mode: generated synthetic mask from pasted instances")

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")

    if args.add_contour:
        canvas_bgr = add_contour_bgr(canvas_bgr)

    # -------------------------
    # WRITE PREVIEW MASK
    # -------------------------
    preview_dir = out_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    # Save preview in BGR (OpenCV expects BGR)
    cv2.imwrite(str(preview_dir / f"mask_color_seed{args.seed}.png"), canvas_bgr)
    if args.mode == "random_mask":
        cv2.imwrite(str(preview_dir / f"pseudo_realB_seed{args.seed}.png"), pseudo_B_bgr)
    # -------------------------
    # WRITE AB (aligned) into out_root/test
    # -------------------------
    test_dir = out_root / "test"
    clean_test_dir(test_dir)

    tag = "ALL" if not want_classes else "_".join(want_classes)
    ab_path = test_dir / f"gen_{args.mode}_{tag}_seed{args.seed}.png"

    effective_empty_dir = args.empty_dir

    if args.use_delta_comp:
        empty_bgr = load_empty_tray_bgr(args.empty_dir, args.empty_path)

        tmp_empty_dir = out_root / "empty_for_test"
        tmp_empty_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(tmp_empty_dir / ab_path.name), empty_bgr)
        effective_empty_dir = str(tmp_empty_dir)

        if args.mode == "random_mask":
            blank_B = pseudo_B_bgr
        else:
            blank_B = empty_bgr
    else:
        if args.mode == "random_mask":
            blank_B = pseudo_B_bgr
        else:
            blank_B = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # IMPORTANT: save AB in BGR because cv2.imwrite expects BGR.
    AB_bgr = np.concatenate([canvas_bgr, blank_B], axis=1)
    cv2.imwrite(str(ab_path), AB_bgr)
    print("Wrote AB to:", ab_path)

    # -------------------------
    # RUN PIX2PIX INFERENCE
    # -------------------------
    run_pix2pix_test(
        temp_dataset_dir=out_root,
        epoch=args.epoch,
        norm=args.norm,
        use_delta_comp=args.use_delta_comp,
        empty_dir=effective_empty_dir,
        empty_path=args.empty_path,
        use_display_mapper=bool(args.use_display_mapper),
        use_soft_mask=args.use_soft_mask,
        mask_blur_ksize=args.mask_blur_ksize,
        mask_soft_beta=args.mask_soft_beta,
    )

    # -------------------------
    # POST-PROCESS: REMOVE HALOS (optional)
    # -------------------------
        # -------------------------
    # POST-PROCESS: SAVE RGB COPY OF fake_B
    # -------------------------
    results_root = Path("results") / MODEL_NAME / f"test_{args.epoch}"

    if results_root.exists():
        fake_paths = list(results_root.rglob("*_fake_B.png"))
        print(f"[rgb-copy] searching in: {results_root}")
        print(f"[rgb-copy] found {len(fake_paths)} fake_B files")

        for p in fake_paths:
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue

            # -------------------------
            # 1 Save RGB version
            # -------------------------
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb_path = p.with_name(p.stem + "_rgb.png")
            cv2.imwrite(str(rgb_path), img_rgb)

            # -------------------------
            # 2 Create colored visualization
            # -------------------------
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

            color_path = p.with_name(p.stem + "_colormap.png")
            cv2.imwrite(str(color_path), color)

            print(f"Saved RGB: {rgb_path}")
            print(f"Saved colormap: {color_path}")
    else:
        print(f"[rgb-copy] results folder does not exist: {results_root}")

    # -------------------------
    # POST-PROCESS: REMOVE HALOS (optional)
    # -------------------------
    if args.remove_halo and HAS_HALO_REMOVER:
        print("\nRemoving white halos from output...")
        images_dir = results_root / "images"

        # Find AB files and fake_B files
        ab_files = sorted(images_dir.glob("*.png"))
        for ab_file in ab_files:
            # Skip if not AB (concatenated)
            AB_img = cv2.imread(str(ab_file))
            if AB_img is None or AB_img.shape[1] != 2048:
                continue

            # Extract A and fake_B
            mask_A = AB_img[:, :1024, :]

            # Find corresponding fake_B
            fake_B_name = ab_file.name
            fake_B_path = images_dir / fake_B_name.replace(".png", "") / f"{ab_file.stem}_fake_B.png"

            # Try alternative pattern (direct fake_B in same dir)
            if not fake_B_path.exists():
                possible_names = [
                    ab_file.name.replace(".png", "_fake_B.png"),
                    fake_B_name.replace(".png", "_fake_B.png"),
                ]
                for pname in possible_names:
                    ppath = images_dir / pname
                    if ppath.exists():
                        fake_B_path = ppath
                        break

            if fake_B_path.exists():
                out_path = images_dir / fake_B_path.name.replace("_fake_B", "_fake_B_dehalo")
                try:
                    remove_halo_with_mask(
                        str(fake_B_path),
                        mask=mask_A,
                        out_path=str(out_path),
                        erode_px=args.halo_erode_px,
                        blend_width=args.halo_blend_width,
                    )
                except Exception as e:
                    print(f"  Warning: Could not process {fake_B_path}: {e}")

        print("Halo-removed images saved with '_dehalo' suffix")
    elif args.remove_halo and not HAS_HALO_REMOVER:
        print("Warning: --remove_halo requested but halo_remover module not found")

    print("\nDone. Open:")
    print(f"results/{MODEL_NAME}/test_{args.epoch}/index.html")
    print("Generated image is *_fake_B.png")
    print("Additional RGB copy is *_fake_B_rgb.png")
    if args.remove_halo:
        print("Halo-removed images are *_fake_B_dehalo.png")


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

  To print out classes that you have:
  python - <<'PY'
    import json
    c=json.load(open("data/raw/Non-Contraband/result.json"))
    print(sorted([x["name"] for x in c["categories"]]))
    PY
"""