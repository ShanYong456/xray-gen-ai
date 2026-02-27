from pathlib import Path
import argparse, json, random, subprocess
import cv2
import numpy as np

# =========================
# CONFIG YOU MUST MATCH
# =========================
SIZE = 1024  # must match your training load_size/crop_size
MODEL_NAME = "contraband_metal_pix2pix_phys_fixDull_v3"   # your pix2pix --name
# MODEL_NAME = "non_contraband_pix2pix_physV2"       # your pix2pix --name
PIX2PIX_DIR = Path("external/pix2pix")

# IMPORTANT:
# Build the canvas in RGB (model conditioning), then convert to BGR ONLY when saving with cv2.imwrite().
# (In this script we keep masks in BGR consistently because OpenCV uses BGR.)
# Your palette MUST match whatever you used to generate training masks.

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

def add_contour_bgr(mask_bgr):
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    out = mask_bgr.copy()
    out[edges > 0] = (255, 255, 255)
    return out

# -------------------------
# Dataset parsing
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
# Mode 1: PASTE (unchanged)
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
                   allow_overlap=False, morph="dilate", morph_k=3, morph_iter=1, soft_edges=False):
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

        region = canvas_bgr[y0:y0 + new_h, x0:x0 + new_w]
        region[m > 0] = color_bgr
        canvas_bgr[y0:y0 + new_h, x0:x0 + new_w] = region

        occ_region = occ_mask[y0:y0 + new_h, x0:x0 + new_w]
        occ_region[m > 0] = 255
        occ_mask[y0:y0 + new_h, x0:x0 + new_w] = occ_region

        return True

    return False

# -------------------------
# Mode 2: REAL_SCENE (unchanged)
# -------------------------
def render_full_scene_mask(coco, images_dir: Path, rng,
                           want_classes=None,
                           edit_drop_prob=0.0,
                           force_size=SIZE):
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, cat_name_to_train, skipped_rle = build_indices(coco)

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
# real_scene_count helpers (unchanged)
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
    images_by_id, cats_by_id, ann_by_img, cat_id_to_train, _, skipped_rle = build_indices(coco)
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

def build_real_scene_count_mask(coco, images_dir: Path, rng,
                               want_classes, targets,
                               keep_others=True,
                               force_size=SIZE,
                               scale_min=0.85, scale_max=1.15,
                               rot_deg=0.0,
                               morph="none", morph_k=3, morph_iter=1,
                               soft_edges=False,
                               min_comp_area=50,
                               max_add_tries=1200,
                               sample_k=200):
    if want_classes is None or targets is None:
        raise SystemExit("real_scene_count requires --classes and --count.")

    img_id = choose_best_real_image_for_targets(coco, want_classes, targets, rng, sample_k=sample_k)
    _, inst_full, fname = get_instances_for_image(coco, images_dir, img_id)

    insts = []
    for d in inst_full:
        m = cv2.resize(d["mask"], (force_size, force_size), interpolation=cv2.INTER_NEAREST)
        insts.append({"train_id": d["train_id"], "cname": d["cname"], "mask": m})

    want_set = set(want_classes)
    controllable = [it for it in insts if it["cname"] in want_set]
    others = [it for it in insts if it["cname"] not in want_set] if keep_others else []

    new_ctrl = []
    for cls_name, target in zip(want_classes, targets):
        cls_insts = [it for it in controllable if it["cname"] == cls_name]
        rng.shuffle(cls_insts)
        new_ctrl.extend(cls_insts[:max(0, int(target))])

    canvas_bgr = render_from_instances(others + new_ctrl, size=force_size)

    occ_mask = np.zeros((force_size, force_size), dtype=np.uint8)
    occ_mask[np.any(canvas_bgr > 0, axis=2)] = 255

    lib, _, _, _ = build_shape_library_for_paste(coco, images_dir)

    for cls_name, target in zip(want_classes, targets):
        target = int(target)

        if cls_name not in lib or len(lib[cls_name]) == 0:
            raise SystemExit(f"Class '{cls_name}' not found or empty in COCO for paste adding.")

        train_id = int(lib[cls_name][0]["train_id"])
        cur = count_components_for_train_id(canvas_bgr, train_id, min_area=min_comp_area)

        tries = 0
        while cur < target and tries < max_add_tries:
            tries += 1
            canvas_backup = canvas_bgr.copy()
            occ_backup = occ_mask.copy()

            inst = rng.choice(lib[cls_name])
            color_bgr = PALETTE_BGR.get(train_id, (255, 255, 255))

            ok = place_instance(
                canvas_bgr=canvas_bgr,
                occ_mask=occ_mask,
                inst_mask_bin=inst["mask"],
                color_bgr=color_bgr,
                rng=rng,
                scale_range=(float(scale_min), float(scale_max)),
                rot_deg=float(rot_deg),
                allow_overlap=False,
                morph=morph,
                morph_k=morph_k,
                morph_iter=morph_iter,
                soft_edges=soft_edges,
                max_tries=140,
            )

            if not ok:
                canvas_bgr[:] = canvas_backup
                occ_mask[:] = occ_backup
                continue

            new_cur = count_components_for_train_id(canvas_bgr, train_id, min_area=min_comp_area)
            if new_cur <= cur:
                canvas_bgr[:] = canvas_backup
                occ_mask[:] = occ_backup
                continue

            cur = new_cur

        if cur < target:
            print(f"real_scene_count: Could not reach target for {cls_name}. Got {cur}, wanted {target}.")

    return canvas_bgr, img_id, fname

# -------------------------
# IO + pix2pix
# -------------------------
def clean_test_dir(test_dir: Path):
    test_dir.mkdir(parents=True, exist_ok=True)
    for p in test_dir.glob("*.png"):
        p.unlink()

# >>> EDIT: helper to pick/load ONE empty tray image
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
        img = cv2.imread(str(cands[0]))  # deterministic: first file
        if img is None:
            raise FileNotFoundError(f"Could not read empty tray image: {cands[0]}")
        return cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    raise ValueError("use_delta_comp requires --empty_dir or --empty_path")

def run_pix2pix_test(
    temp_dataset_dir: Path,
    epoch="latest",
    norm="batch",
    use_delta_comp=False,
    empty_dir="",
    empty_path="",
):
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
    ]

    if use_delta_comp:
        cmd.append("--use_delta_comp")
        if empty_dir:
            cmd.append(f"--empty_dir={empty_dir}")
        if empty_path:
            cmd.append(f"--empty_path={empty_path}")

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
                    choices=["real_scene", "paste", "real_scene_count"])

    ap.add_argument("--scale_min", type=float, default=0.98)
    ap.add_argument("--scale_max", type=float, default=1.02)
    ap.add_argument("--rot_deg", type=float, default=0.0)
    ap.add_argument("--morph", type=str, default="dilate", choices=["dilate", "erode", "none"])
    ap.add_argument("--morph_k", type=int, default=3)
    ap.add_argument("--morph_iter", type=int, default=1)
    ap.add_argument("--soft_edges", action="store_true")

    ap.add_argument("--strict_count", action="store_true")
    ap.add_argument("--min_comp_area", type=int, default=50)
    ap.add_argument("--strict_max_global_tries", type=int, default=500)
    ap.add_argument("--edit_drop_prob", type=float, default=0.0)

    ap.add_argument("--keep_others", action="store_true")
    ap.add_argument("--base_sample_k", type=int, default=200)
    ap.add_argument("--max_add_tries", type=int, default=1200)

    ap.add_argument("--use_delta_comp", action="store_true")
    ap.add_argument("--empty_dir", type=str, default="")
    ap.add_argument("--empty_path", type=str, default="")

    ap.add_argument("--norm", type=str, default="batch", choices=["batch", "instance", "none"])

    args = ap.parse_args()
    rng = random.Random(args.seed)

    images_dir = Path(args.images_dir)
    coco = json.loads(Path(args.coco_json).read_text())

    want_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    want_classes = want_classes if len(want_classes) > 0 else None

    out_root = Path(args.out_dataset)

    # BUILD MASK A
    if args.mode == "real_scene":
        canvas_bgr, img_id, fname = render_full_scene_mask(
            coco=coco, images_dir=images_dir, rng=rng,
            want_classes=want_classes, edit_drop_prob=float(args.edit_drop_prob), force_size=SIZE
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
            coco=coco, images_dir=images_dir, rng=rng,
            want_classes=want_classes, targets=targets,
            keep_others=bool(args.keep_others), force_size=SIZE,
            scale_min=args.scale_min, scale_max=args.scale_max, rot_deg=args.rot_deg,
            morph=args.morph, morph_k=args.morph_k, morph_iter=args.morph_iter,
            soft_edges=args.soft_edges, min_comp_area=int(args.min_comp_area),
            max_add_tries=int(args.max_add_tries), sample_k=int(args.base_sample_k),
        )
        print(f"real_scene_count picked image_id={img_id}, file={fname}")

    else:
        raise SystemExit("paste mode omitted here for brevity (unchanged in your original).")

    if args.add_contour:
        canvas_bgr = add_contour_bgr(canvas_bgr)

    preview_dir = out_root / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_dir / f"mask_color_seed{args.seed}.png"), canvas_bgr)

    # WRITE AB
    test_dir = out_root / "test"
    clean_test_dir(test_dir)

    tag = "ALL" if not want_classes else "_".join(want_classes)
    ab_path = test_dir / f"gen_{args.mode}_{tag}_seed{args.seed}.png"

    # >>> EDIT: if delta-comp, set B = empty tray AND write matching empty file name into a temp empty_dir
    effective_empty_dir = args.empty_dir
    if args.use_delta_comp:
        empty_bgr = load_empty_tray_bgr(args.empty_dir, args.empty_path)

        # Make a temp folder that WILL match filenames exactly
        tmp_empty_dir = out_root / "empty_for_test"
        tmp_empty_dir.mkdir(parents=True, exist_ok=True)

        # Save empty tray with SAME NAME as AB filename (critical!)
        cv2.imwrite(str(tmp_empty_dir / ab_path.name), empty_bgr)
        effective_empty_dir = str(tmp_empty_dir)

        blank_B = empty_bgr  # nicer eval view (real_B = empty tray)
    else:
        blank_B = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    AB = np.concatenate([canvas_bgr, blank_B], axis=1)
    cv2.imwrite(str(ab_path), AB)
    print("Wrote AB to:", ab_path)

    # RUN PIX2PIX INFERENCE
    run_pix2pix_test(
        out_root,
        epoch=args.epoch,
        norm=args.norm,
        use_delta_comp=args.use_delta_comp,
        empty_dir=effective_empty_dir,
        empty_path=args.empty_path,
    )

    print("\nDone. Open:")
    print(f"results/{MODEL_NAME}/test_{args.epoch}/index.html")
    print("Generated image is *_fake_B.png")

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
  python notebooks/StyleGan/generate_pix2pix.py \
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
  python notebooks/StyleGan/generate_pix2pix.py \
  --mode real_scene_count \
  --images_dir data/raw/Contraband/Metal \
  --coco_json data/raw/Contraband/Metal/result.json \
  --classes Blade,Vape \
  --count 2,1 \
  --morph none \
  --seed 88 \
  --out_dataset datasets/_gen_real \
  --epoch latest \
  --use_delta_comp \
  --empty_dir data/interim/GAN/Empty \
  --norm instance

python notebooks/StyleGan/generate_pix2pix.py \
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

  
  

  To print out classes that you have:
  python - <<'PY'
    import json
    c=json.load(open("data/raw/Non-Contraband/result.json"))
    print(sorted([x["name"] for x in c["categories"]]))
    PY
"""