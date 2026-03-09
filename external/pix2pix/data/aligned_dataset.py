import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """
    Paired {A,B} dataset. Optional E (empty tray) for delta compositing.
    Optional T (tray mask) to restrict generation inside tray.

    Also supports synthetic mask-only samples for mixed training:
      - real paired sample: returns A, B, (E), (T), is_synthetic=False
      - synthetic sample:   returns A,    (E), (T), is_synthetic=True

    IMPORTANT:
      - Synthetic samples currently assume batch_size=1, because some samples
        do not include key "B".
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            "--tray_mask_path",
            type=str,
            default="",
            help="Path to a tray mask PNG (white=inside tray, black=outside). Used with --use_tray_mask.",
        )
        parser.add_argument(
            "--tray_mask_invert",
            action="store_true",
            help="Invert tray mask if your PNG uses opposite convention.",
        )
        parser.add_argument(
            "--tray_mask_thr",
            type=float,
            default=0.5,
            help="Threshold (0..1) for binarizing tray mask after loading.",
        )

        parser.add_argument(
            "--tray_mask_autoshift",
            action="store_true",
            help="Shift A so it fits inside tray (bbox shift + pixel-level nudge).",
        )
        parser.add_argument(
            "--tray_shift_max_px",
            type=int,
            default=400,
            help="Max absolute pixels allowed for the total shift (dx,dy).",
        )
        parser.add_argument(
            "--tray_bbox_margin",
            type=int,
            default=2,
            help="Margin (pixels) inside tray bbox for initial bbox shift.",
        )
        parser.add_argument(
            "--tray_mask_dilate_px",
            type=int,
            default=0,
            help="Dilate tray mask by N pixels before computing bbox + nudging.",
        )
        parser.add_argument(
            "--tray_obj_dilate_px",
            type=int,
            default=0,
            help="Dilate object mask by N pixels when removing/pasting object in B.",
        )
        parser.add_argument(
            "--tray_nudge_iters",
            type=int,
            default=6,
            help="Extra refinement iters after bbox shift to ensure ALL object pixels are inside tray pixels.",
        )
        parser.add_argument(
            "--tray_nudge_max_step",
            type=int,
            default=25,
            help="Max per-iter nudge step in pixels (dx,dy).",
        )
        parser.add_argument(
            "--tray_cc_close_px",
            type=int,
            default=2,
            help="Morph close px for tray mask cleanup (fills small holes). 0 disables.",
        )

        # synthetic mask-only branch
        parser.add_argument(
            "--synthetic_prob",
            type=float,
            default=0.0,
            help="Probability of returning a synthetic mask-only sample during training.",
        )
        parser.add_argument(
            "--synthetic_mode",
            type=str,
            default="random_mask",
            choices=["random_mask", "paste"],
            help="How to generate synthetic mask-only A samples.",
        )
        parser.add_argument(
            "--synthetic_min_items",
            type=int,
            default=1,
            help="Minimum number of objects in synthetic mask generation.",
        )
        parser.add_argument(
            "--synthetic_max_items",
            type=int,
            default=3,
            help="Maximum number of objects in synthetic mask generation.",
        )
        parser.add_argument(
            "--synthetic_scale_min",
            type=float,
            default=0.6,
            help="Min scale for synthetic placed objects.",
        )
        parser.add_argument(
            "--synthetic_scale_max",
            type=float,
            default=1.4,
            help="Max scale for synthetic placed objects.",
        )
        parser.add_argument(
            "--synthetic_rot_min",
            type=float,
            default=0.0,
            help="Min rotation for synthetic placed objects.",
        )
        parser.add_argument(
            "--synthetic_rot_max",
            type=float,
            default=360.0,
            help="Max rotation for synthetic placed objects.",
        )
        parser.add_argument(
            "--synthetic_no_overlap",
            action="store_true",
            help="Disallow overlap in synthetic mask generation.",
        )
        parser.add_argument(
            "--cutout_dir",
            type=str,
            default="",
            help="Folder containing colored object cutouts for synthetic generation.",
        )
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size

        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

        self.force_gray_rgb = (self.output_nc == 3)
        self.match_empty_to_B = True
        self.debug_every = 50

        self.shift_reduce_count = 0
        self.shift_reduce_by_iters = []

        # synthetic branch
        self.synthetic_prob = float(getattr(self.opt, "synthetic_prob", 0.0))
        self.synthetic_mode = str(getattr(self.opt, "synthetic_mode", "random_mask"))
        self.synthetic_enabled = self.synthetic_prob > 0.0 and getattr(self.opt, "phase", "") == "train"

        self.cutout_items = []
        if self.synthetic_enabled:
            cutout_dir = str(getattr(self.opt, "cutout_dir", "")).strip()
            if not cutout_dir:
                raise ValueError("synthetic_prob > 0 but --cutout_dir is empty.")
            self.cutout_items = self._load_cutouts(Path(cutout_dir))

        self.use_tray_mask = bool(getattr(self.opt, "use_tray_mask", False))
        self.tray_mask_img = None
        if self.use_tray_mask:
            tray_mask_path = getattr(self.opt, "tray_mask_path", "")
            if not tray_mask_path:
                raise ValueError("--use_tray_mask is set but --tray_mask_path is empty.")
            p = Path(tray_mask_path)
            if not p.exists():
                raise FileNotFoundError(f"Tray mask not found: {p}")
            self.tray_mask_img = Image.open(str(p)).convert("L")

            T_arr = np.array(self.tray_mask_img)
            thr255 = int(np.clip(float(getattr(self.opt, "tray_mask_thr", 0.5)) * 255.0, 0, 255))
            T_bin = (T_arr > thr255).astype(np.uint8)
            if bool(getattr(self.opt, "tray_mask_invert", False)):
                T_bin = (1 - T_bin).astype(np.uint8)
            tray_area = T_bin.sum()
            total_pixels = T_bin.size
            pct = 100.0 * tray_area / total_pixels if total_pixels > 0 else 0.0
            print(
                f"[tray_info] mask size {T_arr.shape} | tray area: {tray_area} pixels ({pct:.1f}%) | "
                f"tray_obj_dilate_px={int(getattr(self.opt, 'tray_obj_dilate_px', 0))} | "
                f"tray_bbox_margin={int(getattr(self.opt, 'tray_bbox_margin', 2))}"
            )

    # -------------------------
    # Synthetic helpers
    # -------------------------
    def _infer_train_id_from_cutout_bgr(self, cut_bgr: np.ndarray) -> int:
        # Adjust this palette if you change dataset classes/colors
        palette = {
            0: (0, 0, 0),
            1: (0, 255, 0),   # Shampoo
            2: (255, 0, 0),   # Blade
        }
        m = np.any(cut_bgr > 0, axis=2)
        if not np.any(m):
            return 0
        pix = cut_bgr[m].reshape(-1, 3)
        uniq, counts = np.unique(pix, axis=0, return_counts=True)
        bgr = tuple(uniq[np.argmax(counts)].tolist())
        for tid, col in palette.items():
            if tuple(col) == bgr:
                return int(tid)
        return 0

    def _load_cutouts(self, cutout_root: Path):
        items = []
        if not cutout_root.exists():
            raise FileNotFoundError(f"cutout_dir not found: {cutout_root}")

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

                tid = self._infer_train_id_from_cutout_bgr(bgr)
                if tid == 0:
                    continue

                m = np.any(bgr > 0, axis=2)
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                y1, y2 = ys.min(), ys.max() + 1
                x1, x2 = xs.min(), xs.max() + 1
                bgr = bgr[y1:y2, x1:x2].copy()

                items.append({"bgr": bgr, "train_id": tid, "path": str(p)})

        if not items:
            raise RuntimeError(f"No valid cutouts loaded from {cutout_root}")
        print(f"[synthetic] loaded {len(items)} cutouts from {cutout_root}")
        return items

    def _rotate_preserve_bgr(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            return img
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        return cv2.warpAffine(
            img,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _transform_cutout(self, bgr: np.ndarray):
        s = np.random.uniform(
            float(getattr(self.opt, "synthetic_scale_min", 0.6)),
            float(getattr(self.opt, "synthetic_scale_max", 1.4)),
        )
        out = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

        ang = np.random.uniform(
            float(getattr(self.opt, "synthetic_rot_min", 0.0)),
            float(getattr(self.opt, "synthetic_rot_max", 360.0)),
        )
        out = self._rotate_preserve_bgr(out, ang)

        m = (np.any(out > 0, axis=2)).astype(np.uint8) * 255
        m = cv2.GaussianBlur(m, (5, 5), 0.8)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return None

        orig_mask = np.any(out > 0, axis=2)
        pix = out[orig_mask].reshape(-1, 3)
        uniq, counts = np.unique(pix, axis=0, return_counts=True)
        dom_color = uniq[np.argmax(counts)].astype(np.uint8)

        clean = np.zeros_like(out)
        clean[m > 0] = dom_color

        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1
        return clean[y1:y2, x1:x2].copy()

    def _build_synthetic_A_img(self, size_hw, T_img: Image.Image):
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool)

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)

        n_min = int(getattr(self.opt, "synthetic_min_items", 1))
        n_max = int(getattr(self.opt, "synthetic_max_items", 3))
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))
        n_obj = np.random.randint(n_min, n_max + 1)

        tries_per_obj = 300
        for _ in range(n_obj):
            item = self.cutout_items[np.random.randint(len(self.cutout_items))]
            cut = self._transform_cutout(item["bgr"])
            if cut is None:
                continue

            h, w = cut.shape[:2]
            if h >= H or w >= W or h < 2 or w < 2:
                continue

            obj_mask = np.any(cut > 0, axis=2)
            placed = False

            for _ in range(tries_per_obj):
                x = np.random.randint(0, W - w + 1)
                y = np.random.randint(0, H - h + 1)

                tray_region = T[y:y + h, x:x + w]
                if not np.all(tray_region[obj_mask]):
                    continue

                if no_overlap and np.any(occ[y:y + h, x:x + w] & obj_mask):
                    continue

                region = canvas[y:y + h, x:x + w]
                region[obj_mask] = cut[obj_mask]
                canvas[y:y + h, x:x + w] = region
                occ[y:y + h, x:x + w][obj_mask] = True
                placed = True
                break

            if not placed:
                continue

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    # -------------------------
    # Basic helpers
    # -------------------------
    def _to_gray_rgb(self, img: Image.Image) -> Image.Image:
        return img.convert("L").convert("RGB")

    def _binary_from_pil(self, img: Image.Image, thr255: int) -> np.ndarray:
        arr = np.array(img.convert("L"))
        return (arr > thr255).astype(np.uint8)

    def _mask_from_Aimg(self, A_img: Image.Image) -> np.ndarray:
        A = np.array(A_img)
        if A.ndim == 2:
            return (A > 0)
        return np.any(A > 0, axis=2)

    def _shift_np(self, arr: np.ndarray, dx: int, dy: int, fill=0) -> np.ndarray:
        H, W = arr.shape[:2]
        out = np.full_like(arr, fill)

        x0_src = max(0, -dx)
        x1_src = min(W, W - dx) if dx >= 0 else W
        y0_src = max(0, -dy)
        y1_src = min(H, H - dy) if dy >= 0 else H

        x0_dst = max(0, dx)
        y0_dst = max(0, dy)
        x1_dst = x0_dst + (x1_src - x0_src)
        y1_dst = y0_dst + (y1_src - y0_src)

        if (x1_src > x0_src) and (y1_src > y0_src):
            out[y0_dst:y1_dst, x0_dst:x1_dst] = arr[y0_src:y1_src, x0_src:x1_src]
        return out

    def _shift_pil_rgb(self, img: Image.Image, dx: int, dy: int) -> Image.Image:
        arr = np.array(img)
        out = self._shift_np(arr, dx, dy, fill=0)
        return Image.fromarray(out)

    def _bbox_from_binary(self, m: np.ndarray):
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _dilate_bin(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01
        k = 2 * px + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(m01.astype(np.uint8), kernel, iterations=1)

    def _clamp_shift_to_image(self, M: np.ndarray, dx: int, dy: int):
        H, W = M.shape[:2]
        bb = self._bbox_from_binary(M)
        if bb is None:
            return 0, 0
        x0, y0, x1, y1 = bb
        dx = int(np.clip(dx, -x0, (W - 1) - x1))
        dy = int(np.clip(dy, -y0, (H - 1) - y1))
        return dx, dy

    # -------------------------
    # Tray mask loader
    # -------------------------
    def _load_tray_T(self, target_size):
        assert self.tray_mask_img is not None
        T_img = self.tray_mask_img
        if T_img.size != target_size:
            T_img = T_img.resize(target_size, resample=Image.NEAREST)
        return T_img

    def _largest_cc(self, m01: np.ndarray) -> np.ndarray:
        m = (m01 > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return m
        areas = stats[1:, cv2.CC_STAT_AREA]
        k = 1 + int(np.argmax(areas))
        return (labels == k).astype(np.uint8)

    def _close(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01
        k = 2 * px + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.morphologyEx(m01.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    def _get_tray_bin(self, T_img: Image.Image) -> np.ndarray:
        thr01 = float(getattr(self.opt, "tray_mask_thr", 0.5))
        thr255 = int(np.clip(thr01 * 255.0, 0, 255))

        T = self._binary_from_pil(T_img, thr255)
        if bool(getattr(self.opt, "tray_mask_invert", False)):
            T = (1 - T).astype(np.uint8)

        T = self._largest_cc(T)
        close_px = int(getattr(self.opt, "tray_cc_close_px", 2))
        T = self._close(T, px=close_px)

        dil = int(getattr(self.opt, "tray_mask_dilate_px", 0))
        if dil > 0:
            T = self._dilate_bin(T, dil)

        return T

    # -------------------------
    # Shift logic: bbox shift + pixel-level nudge
    # -------------------------
    def _bbox_shift_into_tray(self, M: np.ndarray, T: np.ndarray, margin: int):
        obj_bb = self._bbox_from_binary(M)
        tray_bb = self._bbox_from_binary(T)
        if obj_bb is None or tray_bb is None:
            return 0, 0

        ox0, oy0, ox1, oy1 = obj_bb
        tx0, ty0, tx1, ty1 = tray_bb

        tx0 += margin
        ty0 += margin
        tx1 -= margin
        ty1 -= margin

        obj_w = ox1 - ox0 + 1
        obj_h = oy1 - oy0 + 1
        tray_w = tx1 - tx0 + 1
        tray_h = ty1 - ty0 + 1
        if obj_w > tray_w or obj_h > tray_h:
            return 0, 0

        dx = 0
        dy = 0
        if ox0 < tx0:
            dx = tx0 - ox0
        elif ox1 > tx1:
            dx = tx1 - ox1

        if oy0 < ty0:
            dy = ty0 - oy0
        elif oy1 > ty1:
            dy = ty1 - oy1

        return int(dx), int(dy)

    def _nudge_pixels_inside_tray(self, M: np.ndarray, T: np.ndarray, dx0: int, dy0: int):
        max_total = int(getattr(self.opt, "tray_shift_max_px", 400))
        n_iters = int(getattr(self.opt, "tray_nudge_iters", 8))
        max_step = int(getattr(self.opt, "tray_nudge_max_step", 20))

        dx_total, dy_total = self._clamp_shift_to_image(M, int(dx0), int(dy0))

        inv = (T == 1).astype(np.uint8)
        inv = 1 - inv
        dist, labels = cv2.distanceTransformWithLabels(
            inv, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL
        )
        H, W = T.shape

        def score_shift(dx, dy):
            M_shift = self._shift_np(M, dx, dy, fill=0)
            outside = ((M_shift == 1) & (T == 0)).sum()
            mag = dx * dx + dy * dy
            return int(outside), int(mag), M_shift

        best_dx, best_dy = dx_total, dy_total
        best_outside, best_mag, M_work = score_shift(dx_total, dy_total)

        if best_outside == 0:
            return best_dx, best_dy, True

        for _ in range(n_iters):
            outside_mask = (M_work == 1) & (T == 0)
            ys, xs = np.where(outside_mask)
            if len(xs) == 0:
                return dx_total, dy_total, True

            idx = labels[ys, xs].astype(np.int64) - 1
            nx = (idx % W).astype(np.int64)
            ny = (idx // W).astype(np.int64)

            vx = nx - xs
            vy = ny - ys

            step_dx = int(np.median(vx))
            step_dy = int(np.median(vy))

            step_dx = int(np.clip(step_dx, -max_step, max_step))
            step_dy = int(np.clip(step_dy, -max_step, max_step))

            if step_dx == 0 and step_dy == 0:
                j = int(np.argmax(dist[ys, xs]))
                step_dx = int(np.clip(vx[j], -1, 1))
                step_dy = int(np.clip(vy[j], -1, 1))
                if step_dx == 0 and step_dy == 0:
                    break

            cand_dx = int(np.clip(dx_total + step_dx, -max_total, max_total))
            cand_dy = int(np.clip(dy_total + step_dy, -max_total, max_total))
            cand_dx, cand_dy = self._clamp_shift_to_image(M, cand_dx, cand_dy)

            cand_outside, cand_mag, cand_M = score_shift(cand_dx, cand_dy)

            if (cand_outside < best_outside) or (cand_outside == best_outside and cand_mag < best_mag):
                best_dx, best_dy = cand_dx, cand_dy
                best_outside, best_mag = cand_outside, cand_mag

            dx_total, dy_total = cand_dx, cand_dy
            M_work = cand_M

            if cand_outside == 0:
                return cand_dx, cand_dy, True

        return best_dx, best_dy, (best_outside == 0)

    def _compute_autoshift(self, A_img: Image.Image, T_img: Image.Image):
        if not bool(getattr(self.opt, "tray_mask_autoshift", False)):
            return 0, 0

        T = self._get_tray_bin(T_img)

        A_arr = np.array(A_img)
        if A_arr.ndim == 2:
            M = (A_arr > 0).astype(np.uint8)
        else:
            M = (np.any(A_arr > 0, axis=2)).astype(np.uint8)

        if ((M == 1) & (T == 0)).sum() == 0:
            return 0, 0

        margin = int(getattr(self.opt, "tray_bbox_margin", 2))
        dx0, dy0 = self._bbox_shift_into_tray(M, T, margin=margin)

        dx, dy, success = self._nudge_pixels_inside_tray(M, T, dx0, dy0)
        if not success:
            print(
                f"[warning] object could not be fully fit inside tray; "
                f"using best minimal shift found (dx,dy)=({dx},{dy})"
            )
            return int(dx), int(dy)

        return int(dx), int(dy)

    # -------------------------
    # Move object in B using E as background
    # -------------------------
    def _move_object_in_B_with_shift(
        self,
        B_img: Image.Image,
        E_img: Image.Image,
        M_old: np.ndarray,
        dx: int,
        dy: int,
        T: np.ndarray = None,
    ):
        B = np.array(B_img).astype(np.uint8)
        E = np.array(E_img).astype(np.uint8)

        M_old = (M_old > 0).astype(np.uint8)
        if M_old.ndim != 2:
            raise ValueError("M_old must be HxW")

        dil = int(getattr(self.opt, "tray_obj_dilate_px", 0))
        if dil > 0:
            M_paste = self._dilate_bin(M_old, dil)
        else:
            M_paste = M_old.copy()

        dx_final, dy_final = self._clamp_shift_to_image(M_paste, int(dx), int(dy))

        if T is not None:
            M_shifted = self._shift_np(M_paste, dx_final, dy_final, fill=0)
            outside_tray = (M_shifted == 1) & (T == 0)

            if outside_tray.sum() > 0:
                print("[warning] shifted object extends outside tray; reducing shift minimally to fit")

                orig_dx, orig_dy = dx_final, dy_final
                best_dx, best_dy = dx_final, dy_final
                best_outside = int(outside_tray.sum())
                best_mag = int(dx_final * dx_final + dy_final * dy_final)

                max_iters = 200
                iters_used = 0

                for _ in range(max_iters):
                    cur_mag = abs(dx_final) + abs(dy_final)
                    if cur_mag == 0:
                        break

                    candidates = []
                    if dx_final != 0:
                        candidates.append((dx_final - (1 if dx_final > 0 else -1), dy_final))
                    if dy_final != 0:
                        candidates.append((dx_final, dy_final - (1 if dy_final > 0 else -1)))

                    scored = []
                    for cand_dx, cand_dy in candidates:
                        cand_dx, cand_dy = self._clamp_shift_to_image(M_paste, cand_dx, cand_dy)
                        M_cand = self._shift_np(M_paste, cand_dx, cand_dy, fill=0)
                        cand_outside = int(((M_cand == 1) & (T == 0)).sum())
                        cand_mag = int(cand_dx * cand_dx + cand_dy * cand_dy)
                        scored.append((cand_outside, -cand_mag, cand_dx, cand_dy))

                    if not scored:
                        break

                    scored.sort(key=lambda x: (x[0], x[1]))
                    cand_outside, neg_cand_mag, cand_dx, cand_dy = scored[0]
                    cand_mag = -neg_cand_mag

                    dx_final, dy_final = cand_dx, cand_dy
                    iters_used += 1

                    if (cand_outside < best_outside) or (cand_outside == best_outside and cand_mag > best_mag):
                        best_dx, best_dy = cand_dx, cand_dy
                        best_outside = cand_outside
                        best_mag = cand_mag

                    if cand_outside == 0:
                        best_dx, best_dy = cand_dx, cand_dy
                        best_outside = 0
                        break

                self.shift_reduce_count += 1
                self.shift_reduce_by_iters.append(iters_used)

                dx_final, dy_final = best_dx, best_dy

                reduction = int(
                    np.sqrt(float(orig_dx) ** 2 + float(orig_dy) ** 2)
                    - np.sqrt(float(dx_final) ** 2 + float(dy_final) ** 2)
                )

                if best_outside > 0:
                    print(
                        f"[info] shift reduced minimally by {reduction}px in {iters_used} iters to "
                        f"({dx_final},{dy_final}); still {best_outside} pixels outside tray"
                    )
                else:
                    print(
                        f"[info] shift reduced minimally by {reduction}px in {iters_used} iters to "
                        f"({dx_final},{dy_final}); object now fits inside tray"
                    )

        dx, dy = dx_final, dy_final

        out = B.copy()
        out[M_paste == 1] = E[M_paste == 1]

        obj = np.zeros_like(B)
        obj[M_paste == 1] = B[M_paste == 1]

        obj_shift = self._shift_np(obj, dx, dy, fill=0)
        M_obj_shift = self._shift_np(M_paste, dx, dy, fill=0).astype(np.uint8)

        feather_px = 2
        if feather_px > 0:
            k = 2 * feather_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

            M_erode = cv2.erode(M_obj_shift, kernel, iterations=1)
            M_blur = cv2.GaussianBlur(M_obj_shift.astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=1.2)
            M_blur = np.clip(M_blur, 0.0, 1.0)

            alpha = np.zeros_like(M_blur, dtype=np.float32)
            alpha[M_obj_shift > 0] = M_blur[M_obj_shift > 0]
            alpha[M_erode > 0] = 1.0
        else:
            alpha = (M_obj_shift > 0).astype(np.float32)

        alpha3 = alpha[..., None]
        out = obj_shift.astype(np.float32) * alpha3 + out.astype(np.float32) * (1.0 - alpha3)
        out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out)

    # -------------------------
    # Empty tray loader
    # -------------------------
    def _load_empty_E(self, AB_path: str):
        empty_dir = getattr(self.opt, "empty_dir", "")
        if empty_dir:
            bname = Path(AB_path).name
            e_path = Path(empty_dir) / bname

            if not e_path.exists():
                parts = bname.split("-", 1)
                timestamp = None
                if len(parts) == 2:
                    timestamp = parts[1].split("_tr")[0]

                if timestamp:
                    candidates = list(Path(empty_dir).glob(f"*{timestamp}*"))
                    if len(candidates) == 1:
                        e_path = candidates[0]
                    elif len(candidates) > 1:
                        raise FileNotFoundError(f"Multiple empty tray candidates for {bname}: {candidates}")
                    else:
                        empties = list(Path(empty_dir).iterdir())

                        def parse_ts(name: str) -> str:
                            try:
                                return name.split(".")[0]
                            except Exception:
                                return name

                        def ts_to_seconds(ts_str: str):
                            import datetime
                            fmt = "%Y-%m-%d_%H-%M-%S-%f"
                            try:
                                dt = datetime.datetime.strptime(ts_str, fmt)
                                return dt.timestamp()
                            except ValueError:
                                return None

                        target_sec = ts_to_seconds(timestamp)
                        best = None
                        best_diff = None
                        for emp in empties:
                            emp_ts = parse_ts(emp.name)
                            emp_sec = ts_to_seconds(emp_ts)
                            if emp_sec is None or target_sec is None:
                                continue
                            diff = abs(emp_sec - target_sec)
                            if best_diff is None or diff < best_diff:
                                best_diff = diff
                                best = emp

                        if best is not None:
                            print(f"[info] using nearest empty {best.name} for {bname}")
                            e_path = best
                        else:
                            raise FileNotFoundError(f"Empty tray not found for {bname}: {e_path}")
                else:
                    raise FileNotFoundError(f"Empty tray not found for {bname}: {e_path}")

            img = Image.open(str(e_path)).convert("RGB")
            if self.force_gray_rgb:
                img = self._to_gray_rgb(img)
            return img, True

        empty_path = getattr(self.opt, "empty_path", "")
        if empty_path:
            e_path = Path(empty_path)
            if not e_path.exists():
                raise FileNotFoundError(f"Empty tray not found: {e_path}")
            img = Image.open(str(e_path)).convert("RGB")
            if self.force_gray_rgb:
                img = self._to_gray_rgb(img)
            return img, False

        raise ValueError(
            "use_delta_comp is enabled but no empty tray provided. "
            "Set --empty_dir (folder) OR --empty_path (single image)."
        )

    def _match_empty_to_B(self, E_img: Image.Image, B_img: Image.Image, obj_mask: np.ndarray) -> Image.Image:
        E = np.array(E_img).astype(np.float32)
        B = np.array(B_img).astype(np.float32)

        if E.shape[:2] != B.shape[:2] or obj_mask.shape[:2] != B.shape[:2]:
            return E_img

        bg = ~obj_mask
        if bg.sum() < 2000:
            return E_img

        out = E.copy()
        for c in range(3):
            e = E[..., c][bg]
            b = B[..., c][bg]
            e_mean = float(e.mean())
            e_std = float(e.std() + 1e-6)
            b_mean = float(b.mean())
            b_std = float(b.std() + 1e-6)
            a = b_std / e_std
            b0 = b_mean - a * e_mean
            out[..., c] = a * E[..., c] + b0

        out = np.clip(out, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    # -------------------------
    # Main
    # -------------------------
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")

        w, h = AB.size
        w2 = w // 2
        A_img = AB.crop((0, 0, w2, h))
        B_img = AB.crop((w2, 0, w, h))

        use_synth = False
        if self.synthetic_enabled and (np.random.rand() < self.synthetic_prob):
            use_synth = True

        E_img = None
        loaded_from_dir = False
        if getattr(self.opt, "use_delta_comp", False):
            try:
                E_img, loaded_from_dir = self._load_empty_E(AB_path)
            except FileNotFoundError as exc:
                empty_path = getattr(self.opt, "empty_path", "")
                if empty_path and Path(empty_path).exists():
                    E_img = Image.open(empty_path).convert("RGB")
                    loaded_from_dir = False
                    print(f"[warning] {exc}; using --empty_path fallback for E")
                else:
                    E_img = B_img.copy()
                    loaded_from_dir = False
                    print(f"[warning] {exc}; using B as E placeholder (safer than black).")

            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BICUBIC)
            if self.force_gray_rgb:
                E_img = self._to_gray_rgb(E_img)

        # prepare tray image once
        T_img = None
        T_bin = None
        if self.use_tray_mask:
            T_img = self._load_tray_T(A_img.size)
            T_bin = self._get_tray_bin(T_img)

        # synthetic branch replaces A only; no real B supervision
        if use_synth:
            if not self.use_tray_mask:
                raise RuntimeError("Synthetic mode currently requires --use_tray_mask.")
            A_img = self._build_synthetic_A_img((A_img.size[1], A_img.size[0]), T_img)

        # autoshift / paired B shift only for real samples
        dx = dy = 0
        if self.use_tray_mask and (not use_synth):
            A_old = A_img
            dx, dy = self._compute_autoshift(A_img, T_img)

            if dx != 0 or dy != 0:
                A_img = self._shift_pil_rgb(A_img, dx, dy)
                if E_img is not None:
                    M_old = self._mask_from_Aimg(A_old).astype(np.uint8)
                    B_img = self._move_object_in_B_with_shift(B_img, E_img, M_old, dx, dy, T=T_bin)

        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)

        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = None if use_synth else B_transform(B_img)

        # Tray tensor T (1,H,W) in {0,1}
        T = None
        if self.use_tray_mask:
            T_transform = get_transform(self.opt, transform_params, grayscale=True)
            T_tensor = T_transform(T_img)
            T01 = torch.clamp((T_tensor + 1.0) * 0.5, 0.0, 1.0)

            thr = float(getattr(self.opt, "tray_mask_thr", 0.5))
            T01 = (T01 > thr).float()
            if bool(getattr(self.opt, "tray_mask_invert", False)):
                T01 = 1.0 - T01

            if T01.shape[0] != 1:
                T01 = T01[:1]
            T = T01

        if getattr(self.opt, "use_delta_comp", False):
            using_global_empty = (not loaded_from_dir) and bool(getattr(self.opt, "empty_path", ""))
            if self.match_empty_to_B and using_global_empty and (E_img is not None) and (not use_synth):
                obj_mask = self._mask_from_Aimg(A_img)
                E_img = self._match_empty_to_B(E_img, B_img, obj_mask)

            E = B_transform(E_img)

            if (getattr(self.opt, "phase", "") == "train") and (index % int(self.debug_every) == 0):
                msg = (
                    f"[debug] {Path(AB_path).name} | synth={use_synth} | shift(dx,dy)=({dx},{dy}) | "
                    f"A(min,max)=({A.min().item():.3f},{A.max().item():.3f}) "
                    f"E(min,max)=({E.min().item():.3f},{E.max().item():.3f})"
                )
                if not use_synth:
                    msg += f" B(min,max)=({B.min().item():.3f},{B.max().item():.3f})"
                if T is not None:
                    msg += f" T(min,max)=({T.min().item():.3f},{T.max().item():.3f})"
                print(msg)

                if self.shift_reduce_count > 0:
                    avg_iters = float(np.mean(self.shift_reduce_by_iters))
                    pct = 100.0 * self.shift_reduce_count / (index + 1)
                    print(
                        f"[stats] shift reductions: {self.shift_reduce_count}/{index+1} samples ({pct:.1f}%) "
                        f"| avg {avg_iters:.1f} iters to reduce"
                    )

            out = {
                "A": A,
                "E": E,
                "A_paths": AB_path,
                "is_synthetic": use_synth,
            }
            if not use_synth:
                out["B"] = B
                out["B_paths"] = AB_path
            if T is not None:
                out["T"] = T
            return out

        out = {
            "A": A,
            "A_paths": AB_path,
            "is_synthetic": use_synth,
        }
        if not use_synth:
            out["B"] = B
            out["B_paths"] = AB_path
        if T is not None:
            out["T"] = T
        return out

    def __len__(self):
        return len(self.AB_paths)