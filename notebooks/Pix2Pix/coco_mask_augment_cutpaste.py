"""
cut_paste_pix2pix_dataset_with_gui_roi.py

Build a Pix2Pix "aligned" AB dataset (A|B) by cut-and-paste:
- A = color mask (3ch) with training palette
- B = composite tray image: clean background + pasted real object patches

Key improvements:
Clean background using masked nan-median template (no blur inpaint smears)
Background library variants (grain/contrast jitter)
Balanced 1/2/3 objects, duplicates allowed
Random locations, OPTIONAL ROI restriction
Optional overlap control
GUI ROI drawer: draw allowed paste region once, save as PNG mask

Run:
  python cut_paste_pix2pix_dataset_with_gui_roi.py
"""

from pathlib import Path
import cv2
import numpy as np
import random
import math

# =========================
# INPUTS
# =========================
IMG_DIR  = Path("data/raw/Contraband/Metal")                  # B (real tray scans)
MASK_DIR = Path("data/interim/contraband_metal/masks_color")  # A (color masks)

# =========================
# OUTPUT
# =========================
OUT_ROOT  = Path("datasets/contraband_metal_cutpaste512")
TRAIN_DIR = OUT_ROOT / "train"
TEST_DIR  = OUT_ROOT / "test"
DBG_DIR   = OUT_ROOT / "_debug"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
DBG_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# MUST MATCH TRAINING
# =========================
SIZE = 512  # final output size (A and B resized to SIZE x SIZE)

PALETTE_RGB = {
    0: (0, 0, 0),         # background
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255),
    4: (255, 255, 0),
    5: (0, 255, 255),
    6: (255, 0, 255),
}

# =========================
# DATASET SIZE / BALANCE
# =========================
SEED = 123
TEST_RATIO = 0.2

SYN_PER_IMAGE_TRAIN = 80
SYN_PER_IMAGE_TEST  = 30

FULL_PER_IMAGE_TRAIN = 4
FULL_PER_IMAGE_TEST  = 2

COUNT_CHOICES = [1, 2, 3]
COUNT_PROBS   = [1/3, 1/3, 1/3]

ALLOW_DUPLICATES = True

# Placement constraints
ALLOW_OVERLAP = False
MIN_DIST_PX = 18

# Transform ranges for pasted object patches
ROT_DEG = 12
SCALE_MIN, SCALE_MAX = 0.80, 1.25

# Blending
FEATHER_PX = 3
COLOR_JITTER = 0.06

# =========================
# BACKGROUND (CLEAN)
# =========================
USE_MEDIAN_TEMPLATE_BG = True

INPAINT_RADIUS = 5
INPAINT_METHOD = cv2.INPAINT_TELEA

BG_DILATE_K = 13
BG_DILATE_IT = 2

BG_TEMPLATE_VARIANTS = 50
BG_JITTER_CONTRAST = 0.06
BG_JITTER_BRIGHTNESS = 6
BG_NOISE_SIGMA = 2.0

# =========================
# ROI (GUI)
# =========================
USE_TRAY_ROI = True
ROI_MASK_PATH = Path("data/roi/tray_roi_mask.png")  # saved mask (1ch) white=allowed
ROI_INSIDE_THRESH = 0.92  # % of pasted alpha that must fall inside ROI

# If you want the ROI to be a little "safer" (avoid edges), erode it:
ROI_ERODE_PX = 0  # e.g. 10 or 20; 0 disables

# =========================
# PATCH EXTRACTION
# =========================
PATCH_MIN_AREA = 120

# =========================
# RNG
# =========================
random.seed(SEED)
np.random.seed(SEED)

# =========================
# Helpers
# =========================
def list_pairs(mask_dir: Path, img_dir: Path):
    pairs = []
    for mp in sorted(mask_dir.glob("*.png")):
        ip = img_dir / mp.name
        if ip.exists():
            pairs.append((mp, ip))
    return pairs

def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def resize_mask_rgb(mask_rgb):
    return cv2.resize(mask_rgb, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)

def resize_img_rgb(img_rgb):
    return cv2.resize(img_rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

def union_object_mask(mask_rgb):
    gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    return (gray > 0).astype(np.uint8) * 255

def dilate_mask(mask_u8, k=11, it=2):
    k = int(k)
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask_u8, kernel, iterations=int(it))

def erode_mask(mask_u8, k=11, it=1):
    k = int(k)
    if k <= 0:
        return mask_u8
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.erode(mask_u8, kernel, iterations=int(it))

def feather_alpha(alpha_u8, feather_px=3):
    if feather_px <= 0:
        return alpha_u8
    k = feather_px * 2 + 1
    return cv2.GaussianBlur(alpha_u8, (k, k), 0)

def random_choice_weighted(items, probs, rng=random):
    r = rng.random()
    cum = 0.0
    for it, p in zip(items, probs):
        cum += p
        if r <= cum:
            return it
    return items[-1]

def exact_color_mask(mask_rgb, color_rgb):
    c = np.array(color_rgb, dtype=np.uint8).reshape(1, 1, 3)
    eq = np.all(mask_rgb == c, axis=2)
    return (eq.astype(np.uint8) * 255)

def connected_components(bin_u8):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_u8, connectivity=8)
    comps = []
    for i in range(1, num):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx, cy = centroids[i]
        comps.append((x, y, w, h, area, float(cx), float(cy)))
    return comps

def clip_bbox(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def apply_transform_rgba(patch_rgb, alpha_u8, angle_deg, scale):
    h, w = patch_rgb.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)

    corners = np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]], dtype=np.float32).T
    tc = (M @ corners).T
    minx, miny = tc[:, 0].min(), tc[:, 1].min()
    maxx, maxy = tc[:, 0].max(), tc[:, 1].max()
    out_w = int(math.ceil(maxx - minx))
    out_h = int(math.ceil(maxy - miny))

    M2 = M.copy()
    M2[:, 2] -= (minx, miny)

    out_rgb = cv2.warpAffine(
        patch_rgb, M2, (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    out_a = cv2.warpAffine(
        alpha_u8, M2, (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return out_rgb, out_a

def paste_rgba(dst_rgb, dst_mask_rgb, patch_rgb, patch_alpha_u8, color_rgb, x0, y0):
    H, W = dst_rgb.shape[:2]
    ph, pw = patch_rgb.shape[:2]
    if x0 < 0 or y0 < 0 or x0 + pw > W or y0 + ph > H:
        return False

    a = patch_alpha_u8.astype(np.float32) / 255.0
    if a.max() <= 0:
        return False
    a3 = a[..., None]

    if COLOR_JITTER > 0:
        jitter = 1.0 + random.uniform(-COLOR_JITTER, COLOR_JITTER)
        patch_rgb2 = np.clip(patch_rgb.astype(np.float32) * jitter, 0, 255).astype(np.uint8)
    else:
        patch_rgb2 = patch_rgb

    roi = dst_rgb[y0:y0+ph, x0:x0+pw].astype(np.float32)
    obj = patch_rgb2.astype(np.float32)
    out = obj * a3 + roi * (1.0 - a3)
    dst_rgb[y0:y0+ph, x0:x0+pw] = np.clip(out, 0, 255).astype(np.uint8)

    mroi = dst_mask_rgb[y0:y0+ph, x0:x0+pw]
    write = patch_alpha_u8 > 0
    mroi[write] = np.array(color_rgb, dtype=np.uint8)
    dst_mask_rgb[y0:y0+ph, x0:x0+pw] = mroi
    return True

# =========================
# GUI: draw ROI mask
# =========================
def draw_roi_gui(reference_bgr, out_mask_path: Path, brush_start=20):
    ref = reference_bgr.copy()
    H, W = ref.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    brush = int(brush_start)
    drawing = False
    erase_mode = False

    win = "Draw ROI (white=allowed) | LMB paint | [E] erase | [ ] brush | [C] clear | [S] save | [Q] quit"

    def render():
        overlay = ref.copy()
        allowed = mask > 0
        overlay[allowed] = (0.6 * overlay[allowed] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

        edges = cv2.Canny(mask, 50, 150)
        overlay[edges > 0] = (0, 0, 255)

        mode = "ERASE" if erase_mode else "PAINT"
        cv2.putText(overlay, f"Mode: {mode} | Brush: {brush}px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "White=allowed. S=save, Q=quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        return overlay

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, mask
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            val = 0 if erase_mode else 255
            cv2.circle(mask, (x, y), brush, val, -1)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, render())
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), ord('Q'), 27):
            break
        elif k in (ord('e'), ord('E')):
            erase_mode = not erase_mode
        elif k in (ord('c'), ord('C')):
            mask[:] = 0
        elif k in (ord('s'), ord('S')):
            out_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_mask_path), mask)
            print(f"✅ Saved ROI mask to: {out_mask_path.resolve()}")
        elif k == ord('['):
            brush = max(1, brush - 2)
        elif k == ord(']'):
            brush = min(200, brush + 2)

    cv2.destroyAllWindows()
    return mask

def load_or_create_roi_mask(pairs, bg_template_rgb):
    """
    ROI mask must match the resolution of synthesis backgrounds (bg_template).
    If ROI_MASK_PATH exists, load it. Otherwise open GUI to draw and save it.
    """
    H, W = bg_template_rgb.shape[:2]

    if ROI_MASK_PATH.exists():
        roi = cv2.imread(str(ROI_MASK_PATH), cv2.IMREAD_GRAYSCALE)
        if roi is None:
            raise RuntimeError(f"ROI mask exists but failed to load: {ROI_MASK_PATH}")
        if roi.shape[:2] != (H, W):
            roi = cv2.resize(roi, (W, H), interpolation=cv2.INTER_NEAREST)
        roi = (roi > 0).astype(np.uint8) * 255
    else:
        # Use the template as reference, so ROI matches the real tray geometry
        ref_bgr = rgb_to_bgr(bg_template_rgb)
        roi = draw_roi_gui(ref_bgr, ROI_MASK_PATH, brush_start=22)
        if roi.shape[:2] != (H, W):
            roi = cv2.resize(roi, (W, H), interpolation=cv2.INTER_NEAREST)
        roi = (roi > 0).astype(np.uint8) * 255

    if ROI_ERODE_PX > 0:
        roi = erode_mask(roi, k=ROI_ERODE_PX * 2 + 1, it=1)

    # Debug-save what ROI looks like
    cv2.imwrite(str(DBG_DIR / "tray_roi_mask_preview.png"), roi)
    print(f"ROI ready. White pixels allowed: {(roi>0).mean()*100:.2f}%")

    return roi

# =========================
# 1) Extract object patches
# =========================
def extract_object_library(pairs, min_area=120):
    lib = {tid: [] for tid in PALETTE_RGB.keys() if tid != 0}

    for mp, ip in pairs:
        mask_bgr = cv2.imread(str(mp))
        img_bgr  = cv2.imread(str(ip))
        if mask_bgr is None or img_bgr is None:
            continue

        mask_rgb = bgr_to_rgb(mask_bgr)
        img_rgb  = bgr_to_rgb(img_bgr)
        H, W = img_rgb.shape[:2]

        for tid, color_rgb in PALETTE_RGB.items():
            if tid == 0:
                continue

            binm = exact_color_mask(mask_rgb, color_rgb)
            comps = connected_components(binm)

            for (x, y, w, h, area, cx, cy) in comps:
                if area < min_area:
                    continue

                x, y, w, h = clip_bbox(x, y, w, h, W, H)
                alpha = binm[y:y+h, x:x+w].copy()
                if alpha.max() == 0:
                    continue

                patch = img_rgb[y:y+h, x:x+w].copy()
                alpha = feather_alpha(alpha, FEATHER_PX)

                lib[tid].append({"rgb": patch, "alpha": alpha, "color_rgb": color_rgb, "src": ip.name})

    lib = {tid: v for tid, v in lib.items() if len(v) > 0}
    print("Object library sizes:")
    for tid, v in sorted(lib.items()):
        print(f"  train_id={tid}  color={PALETTE_RGB[tid]}  patches={len(v)}")
    return lib

# =========================
# 2) Clean background template (FIXED indexing)
# =========================
def masked_nanmedian_background(pairs, target_size=None, dilate_k=13, dilate_it=2):
    """
    Build a clean tray background template using nan-median over multiple images,
    using only pixels that are background (mask==black).
    Uses np.where (broadcast-safe) to avoid boolean indexing shape errors.
    """
    imgs = []
    valids = []
    W = H = None

    for mp, ip in pairs:
        mask_bgr = cv2.imread(str(mp))
        img_bgr  = cv2.imread(str(ip))
        if mask_bgr is None or img_bgr is None:
            continue

        mask_rgb = bgr_to_rgb(mask_bgr)
        img_rgb  = bgr_to_rgb(img_bgr)

        if target_size is None:
            if W is None:
                H, W = img_rgb.shape[:2]
        else:
            W, H = target_size

        if (img_rgb.shape[1], img_rgb.shape[0]) != (W, H):
            img_rgb = cv2.resize(img_rgb, (W, H), interpolation=cv2.INTER_AREA)
        if (mask_rgb.shape[1], mask_rgb.shape[0]) != (W, H):
            mask_rgb = cv2.resize(mask_rgb, (W, H), interpolation=cv2.INTER_NEAREST)

        obj = union_object_mask(mask_rgb)           # 255 where objects exist
        obj = dilate_mask(obj, dilate_k, dilate_it) # expand to remove halos
        valid = (obj == 0)                          # True where pixel is safe background

        imgs.append(img_rgb.astype(np.float32))
        valids.append(valid.astype(bool))

    if len(imgs) < 2:
        raise RuntimeError("Need at least 2 images to build a median background template.")

    imgs = np.stack(imgs, axis=0)       # (N,H,W,3)
    valids = np.stack(valids, axis=0)   # (N,H,W)

    # Broadcast-safe masking (valids[...,None] becomes (N,H,W,1) and broadcasts to (N,H,W,3))
    imgs_nan = np.where(valids[..., None], imgs, np.nan)

    bg = np.nanmedian(imgs_nan, axis=0)  # (H,W,3) float with NaNs where always covered

    nan_mask = np.isnan(bg).any(axis=2)  # (H,W)
    bg_u8 = np.clip(np.nan_to_num(bg, nan=0.0), 0, 255).astype(np.uint8)

    # Fill rare holes with a small inpaint pass
    if nan_mask.any():
        holes = (nan_mask.astype(np.uint8) * 255)
        bg_bgr = rgb_to_bgr(bg_u8)
        bg_bgr = cv2.inpaint(bg_bgr, holes, 5, cv2.INPAINT_TELEA)
        bg_u8 = bgr_to_rgb(bg_bgr)

    return bg_u8

def make_background_library_from_template(bg_template_rgb, n_variants=40):
    bgs = []
    for i in range(int(n_variants)):
        x = bg_template_rgb.astype(np.float32)

        alpha = 1.0 + random.uniform(-BG_JITTER_CONTRAST, BG_JITTER_CONTRAST)
        beta  = random.uniform(-BG_JITTER_BRIGHTNESS, BG_JITTER_BRIGHTNESS)
        x = x * alpha + beta

        if BG_NOISE_SIGMA > 0:
            noise = np.random.normal(0, BG_NOISE_SIGMA, size=x.shape).astype(np.float32)
            x = x + noise

        x = np.clip(x, 0, 255).astype(np.uint8)
        bgs.append({"rgb": x, "src": f"median_template_var_{i:03d}"})
    return bgs

def build_backgrounds_fallback_inpaint(pairs):
    bgs = []
    for mp, ip in pairs:
        mask_bgr = cv2.imread(str(mp))
        img_bgr  = cv2.imread(str(ip))
        if mask_bgr is None or img_bgr is None:
            continue
        mask_rgb = bgr_to_rgb(mask_bgr)
        img_rgb  = bgr_to_rgb(img_bgr)

        m = union_object_mask(mask_rgb)
        m = dilate_mask(m, BG_DILATE_K, BG_DILATE_IT)
        bg_bgr = cv2.inpaint(rgb_to_bgr(img_rgb), m, INPAINT_RADIUS, INPAINT_METHOD)
        bg_rgb = bgr_to_rgb(bg_bgr)
        bgs.append({"rgb": bg_rgb, "src": ip.name})

    print(f"Fallback inpaint backgrounds built: {len(bgs)}")
    return bgs

# =========================
# 3) Synthesize one sample
# =========================
def synth_one(bg_rgb, lib, tray_roi=None):
    H, W = bg_rgb.shape[:2]
    out_img = bg_rgb.copy()
    out_msk = np.zeros((H, W, 3), dtype=np.uint8)

    n_obj = random_choice_weighted(COUNT_CHOICES, COUNT_PROBS)
    tids = sorted(list(lib.keys()))
    if len(tids) == 0:
        return out_msk, out_img

    if ALLOW_DUPLICATES:
        chosen = [random.choice(tids) for _ in range(n_obj)]
    else:
        chosen = random.sample(tids, k=min(n_obj, len(tids)))

    placed_centers = []

    for tid in chosen:
        patch = random.choice(lib[tid])
        prgb = patch["rgb"]
        palpha = patch["alpha"]
        color_rgb = patch["color_rgb"]

        ang = random.uniform(-ROT_DEG, ROT_DEG)
        sc  = random.uniform(SCALE_MIN, SCALE_MAX)
        trgb, ta = apply_transform_rgba(prgb, palpha, ang, sc)

        ph, pw = trgb.shape[:2]
        if ph >= H or pw >= W:
            continue

        success = False
        for _ in range(250):
            x0 = random.randint(0, W - pw)
            y0 = random.randint(0, H - ph)

            #ROI restriction (your requested "set region")
            if tray_roi is not None:
                roi_patch = tray_roi[y0:y0+ph, x0:x0+pw]
                inside = roi_patch[ta > 0]
                if inside.size == 0 or (inside > 0).mean() < ROI_INSIDE_THRESH:
                    continue

            if not ALLOW_OVERLAP:
                existing = out_msk[y0:y0+ph, x0:x0+pw]
                clash = np.any((ta > 0) & (np.any(existing != 0, axis=2)))
                if clash:
                    continue

            if MIN_DIST_PX > 0:
                cx = x0 + pw / 2
                cy = y0 + ph / 2
                ok = True
                for (px, py) in placed_centers:
                    if (cx - px) ** 2 + (cy - py) ** 2 < (MIN_DIST_PX ** 2):
                        ok = False
                        break
                if not ok:
                    continue

            if paste_rgba(out_img, out_msk, trgb, ta, color_rgb, x0, y0):
                placed_centers.append((x0 + pw / 2, y0 + ph / 2))
                success = True
                break

        if not success:
            continue

    return out_msk, out_img

def write_AB(A_rgb, B_rgb, out_path: Path):
    A2 = resize_mask_rgb(A_rgb)
    B2 = resize_img_rgb(B_rgb)
    AB = np.concatenate([A2, B2], axis=1)
    cv2.imwrite(str(out_path), rgb_to_bgr(AB))

# =========================
# Main build
# =========================
def process_split(pairs, out_dir, tag, syn_per_image, full_per_image, lib, bgs, tray_roi=None):
    idx = 0
    for mp, ip in pairs:
        mask_bgr = cv2.imread(str(mp))
        img_bgr  = cv2.imread(str(ip))
        if mask_bgr is None or img_bgr is None:
            continue
        mask_rgb = bgr_to_rgb(mask_bgr)
        img_rgb  = bgr_to_rgb(img_bgr)

        for _ in range(full_per_image):
            write_AB(mask_rgb, img_rgb, out_dir / f"{ip.stem}_{tag}_full_{idx:06d}.png")
            idx += 1

        for k in range(syn_per_image):
            bg = random.choice(bgs)["rgb"]
            A_syn, B_syn = synth_one(bg, lib, tray_roi=tray_roi)

            write_AB(A_syn, B_syn, out_dir / f"{ip.stem}_{tag}_syn_{k}_{idx:06d}.png")

            if tag == "tr" and k in (0, 1):
                dbgA = resize_mask_rgb(A_syn)
                dbgB = resize_img_rgb(B_syn)
                cv2.imwrite(str(DBG_DIR / f"dbg_{ip.stem}_{tag}_{k}_A.png"), rgb_to_bgr(dbgA))
                cv2.imwrite(str(DBG_DIR / f"dbg_{ip.stem}_{tag}_{k}_B.png"), rgb_to_bgr(dbgB))

            idx += 1

    print(f"[{tag}] wrote {idx} samples -> {out_dir}")

def main():
    pairs = list_pairs(MASK_DIR, IMG_DIR)
    if len(pairs) == 0:
        raise SystemExit("No pairs found. Check IMG_DIR / MASK_DIR and matching filenames.")

    random.shuffle(pairs)
    n_test = max(1, int(len(pairs) * TEST_RATIO))
    test_pairs = pairs[:n_test]
    train_pairs = pairs[n_test:]

    lib = extract_object_library(pairs, min_area=PATCH_MIN_AREA)
    if len(lib) == 0:
        raise SystemExit("Object library empty. Check palette colors match mask colors exactly.")

    # Backgrounds (clean template)
    tray_roi = None
    if USE_MEDIAN_TEMPLATE_BG:
        print("Building median background template (clean, no blur)...")
        bg_template = masked_nanmedian_background(
            pairs,
            target_size=None,
            dilate_k=BG_DILATE_K,
            dilate_it=BG_DILATE_IT,
        )
        cv2.imwrite(str(DBG_DIR / "median_bg_template.png"), rgb_to_bgr(bg_template))

        bgs = make_background_library_from_template(bg_template, n_variants=BG_TEMPLATE_VARIANTS)
        print(f"Background library from template: {len(bgs)} variants")

        # ROI mask (GUI draw once, then reuse)
        if USE_TRAY_ROI:
            tray_roi = load_or_create_roi_mask(pairs, bg_template)
    else:
        print("Building inpaint backgrounds (fallback, may blur)...")
        bgs = build_backgrounds_fallback_inpaint(pairs)
        if len(bgs) == 0:
            raise SystemExit("No backgrounds built.")

        # If you are using fallback backgrounds and still want ROI,
        # you can draw ROI against the first background too:
        if USE_TRAY_ROI:
            tray_roi = load_or_create_roi_mask(pairs, bgs[0]["rgb"])

    process_split(train_pairs, TRAIN_DIR, "tr", SYN_PER_IMAGE_TRAIN, FULL_PER_IMAGE_TRAIN, lib, bgs, tray_roi=tray_roi)
    process_split(test_pairs,  TEST_DIR,  "te", SYN_PER_IMAGE_TEST,  FULL_PER_IMAGE_TEST,  lib, bgs, tray_roi=tray_roi)

    print("\nDone:", OUT_ROOT.resolve())
    print("Train pix2pix with:")
    print(f"  python external/pix2pix/train.py \\")
    print(f"    --dataroot {OUT_ROOT} \\")
    print(f"    --name contraband_metal_pix2pix_cutpaste \\")
    print(f"    --model pix2pix --dataset_mode aligned --direction AtoB \\")
    print(f"    --preprocess none --load_size {SIZE} --crop_size {SIZE} \\")
    print(f"    --input_nc 3 --output_nc 3 --netG unet_256 --netD n_layers --n_layers_D 4 \\")
    print(f"    --norm instance --gan_mode lsgan --lambda_L1 60 --batch_size 1 --no_flip")

if __name__ == "__main__":
    main()