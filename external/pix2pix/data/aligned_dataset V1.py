import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


def round_up_to_multiple(x: int, base: int) -> int:
    return ((int(x) + base - 1) // base) * base


def pad_to_canvas(img: Image.Image, target_w: int, target_h: int, fill=(0, 0, 0), auto_expand=False, round_base=256):
    w, h = img.size

    if auto_expand:
        target_w = max(target_w, w)
        target_h = max(target_h, h)
        if round_base is not None and round_base > 1:
            target_w = round_up_to_multiple(target_w, round_base)
            target_h = round_up_to_multiple(target_h, round_base)

    if w == target_w and h == target_h:
        return img

    if w > target_w or h > target_h:
        raise ValueError(
            f"Image size {w}x{h} is larger than canvas {target_w}x{target_h}. "
            f"Increase canvas size."
        )

    canvas = Image.new(img.mode, (target_w, target_h), fill)
    x = (target_w - w) // 2
    y = (target_h - h) // 2
    canvas.paste(img, (x, y))
    return canvas


class AlignedDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--tray_mask_path", type=str, default="")
        parser.add_argument("--tray_mask_dir", type=str, default="")
        parser.add_argument("--tray_mask_invert", action="store_true")
        parser.add_argument("--tray_mask_thr", type=float, default=0.5)
        parser.add_argument("--tray_mask_autoshift", action="store_true")
        parser.add_argument("--tray_shift_max_px", type=int, default=400)
        parser.add_argument("--tray_bbox_margin", type=int, default=2)
        parser.add_argument("--tray_mask_dilate_px", type=int, default=0)
        parser.add_argument("--tray_obj_dilate_px", type=int, default=0)
        parser.add_argument("--tray_nudge_iters", type=int, default=6)
        parser.add_argument("--tray_nudge_max_step", type=int, default=25)
        parser.add_argument("--tray_cc_close_px", type=int, default=2)

        parser.add_argument("--return_instance_masks", action="store_true")

        parser.add_argument("--synthetic_prob", type=float, default=0.0)
        parser.add_argument(
            "--synthetic_mode",
            type=str,
            default="random_mask",
            choices=["random_mask", "paste"],
        )
        parser.add_argument("--synthetic_min_items", type=int, default=1)
        parser.add_argument("--synthetic_max_items", type=int, default=3)
        parser.add_argument("--synthetic_scale_min", type=float, default=0.6)
        parser.add_argument("--synthetic_scale_max", type=float, default=1.4)
        parser.add_argument("--synthetic_rot_min", type=float, default=0.0)
        parser.add_argument("--synthetic_rot_max", type=float, default=360.0)
        parser.add_argument("--synthetic_no_overlap", action="store_true")
        parser.add_argument("--synthetic_same_class_prob", type=float, default=0.0)
        parser.add_argument("--cutout_dir", type=str, default="")

        parser.add_argument("--disable_test_appearance", action="store_true")
        parser.add_argument("--appearance_dropout", type=float, default=0.5)
        parser.add_argument("--appearance_zero_prob", type=float, default=0.35)
        parser.add_argument("--appearance_weak_prob", type=float, default=0.35)
        parser.add_argument("--appearance_proto_prob", type=float, default=0.15)
        parser.add_argument("--appearance_blur_ksize", type=int, default=31)
        parser.add_argument("--appearance_blur_sigma", type=float, default=8.0)
        parser.add_argument("--build_appearance_prototypes", action="store_true")
        parser.add_argument("--max_appearance_prototypes", type=int, default=200)

        parser.add_argument("--use_edge_channel", action="store_true")
        parser.add_argument("--edge_dilate_px", type=int, default=1)
        parser.add_argument("--use_coord_channels", action="store_true")

        parser.add_argument("--mask_aug_px", type=int, default=2)

        parser.add_argument("--pad_to_canvas", action="store_true")
        parser.add_argument("--canvas_w", type=int, default=1024)
        parser.add_argument("--canvas_h", type=int, default=1536)
        parser.add_argument("--canvas_fill", type=int, default=0)

        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))

        if getattr(self.opt, "crop_size", 0) > 0 and getattr(self.opt, "load_size", 0) > 0:
            assert self.opt.load_size >= self.opt.crop_size

        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc
        self.force_gray_rgb = self.output_nc == 3
        self.debug_every = 50

        self.pad_to_canvas_enabled = bool(getattr(opt, "pad_to_canvas", False))
        fill_val = int(getattr(opt, "canvas_fill", 0))
        self.canvas_fill_rgb = (fill_val, fill_val, fill_val)

        raw_canvas_w = int(getattr(opt, "canvas_w", 1024))
        raw_canvas_h = int(getattr(opt, "canvas_h", 1536))
        self.canvas_w = round_up_to_multiple(raw_canvas_w, 256)
        self.canvas_h = round_up_to_multiple(raw_canvas_h, 256)

        if self.pad_to_canvas_enabled:
            print(f"[canvas] requested {raw_canvas_w}x{raw_canvas_h} -> base UNet-safe {self.canvas_w}x{self.canvas_h}")

        self.class_nc = int(getattr(opt, "class_nc", 2))
        self.thickness_nc = int(getattr(opt, "thickness_nc", 1))
        self.use_thickness_channel = bool(getattr(opt, "use_thickness_channel", False))
        self.use_appearance_channel = bool(getattr(opt, "use_appearance_channel", False))
        self.appearance_nc = int(getattr(opt, "appearance_nc", 1))
        self.return_instance_masks = bool(getattr(opt, "return_instance_masks", False))
        self.use_edge_channel = bool(getattr(opt, "use_edge_channel", False))
        self.use_coord_channels = bool(getattr(opt, "use_coord_channels", False))

        if self.use_appearance_channel and self.appearance_nc != 1:
            raise ValueError("Only appearance_nc=1 is supported.")

        self.synthetic_prob = float(getattr(opt, "synthetic_prob", 0.0))
        self.synthetic_enabled = self.synthetic_prob > 0.0 and getattr(opt, "phase", "") == "train"
        self.cutout_items = []
        self.appearance_prototypes = []

        if self.synthetic_enabled:
            cutout_dir = str(getattr(opt, "cutout_dir", "")).strip()
            if not cutout_dir:
                raise ValueError("synthetic_prob > 0 requires --cutout_dir.")
            self.cutout_items = self._load_cutouts(Path(cutout_dir))

        self.use_tray_mask = bool(getattr(opt, "use_tray_mask", False))
        self.tray_mask_img = None
        self.tray_mask_paths = []

        if self.use_tray_mask:
            tray_dir = str(getattr(opt, "tray_mask_dir", "")).strip()
            tray_path = str(getattr(opt, "tray_mask_path", "")).strip()

            if tray_dir:
                tray_dir_p = Path(tray_dir)
                if not tray_dir_p.exists():
                    raise FileNotFoundError(f"Tray mask dir not found: {tray_dir_p}")

                self.tray_mask_paths = sorted(tray_dir_p.glob("*.png"))
                if not self.tray_mask_paths:
                    raise FileNotFoundError(f"No tray mask PNGs found in: {tray_dir_p}")

                print(f"[tray] loaded {len(self.tray_mask_paths)} tray masks from {tray_dir_p}")

            elif tray_path:
                p = Path(tray_path)
                if not p.exists():
                    raise FileNotFoundError(f"Tray mask not found: {p}")
                self.tray_mask_img = Image.open(str(p)).convert("L")
                T_arr = np.array(self.tray_mask_img)
                thr255 = int(np.clip(float(getattr(opt, "tray_mask_thr", 0.5)) * 255, 0, 255))
                T_bin = (T_arr > thr255).astype(np.uint8)
                if bool(getattr(opt, "tray_mask_invert", False)):
                    T_bin = 1 - T_bin
                pct = 100.0 * T_bin.sum() / max(T_bin.size, 1)
                print(f"[tray] single mask {T_arr.shape} | area {T_bin.sum()} ({pct:.1f}%)")

            else:
                raise ValueError("--use_tray_mask requires --tray_mask_path or --tray_mask_dir.")

        if self.use_appearance_channel and bool(getattr(opt, "build_appearance_prototypes", False)):
            self.appearance_prototypes = self._build_appearance_prototype_bank(
                max_items=int(getattr(opt, "max_appearance_prototypes", 200))
            )
            print(f"[appearance] built {len(self.appearance_prototypes)} prototypes")

    def _is_tray_sample(self, ab_path) -> bool:
        parts = [p.lower() for p in Path(ab_path).parts]
        return "tray" in parts

    def _maybe_pad_rgb(self, img: Image.Image) -> Image.Image:
        if not self.pad_to_canvas_enabled:
            return img
        return pad_to_canvas(
            img,
            self.canvas_w,
            self.canvas_h,
            self.canvas_fill_rgb,
            auto_expand=True,
            round_base=256,
        )

    def _maybe_pad_gray(self, img: Image.Image) -> Image.Image:
        if not self.pad_to_canvas_enabled:
            return img
        fill_val = self.canvas_fill_rgb[0]
        return pad_to_canvas(
            img,
            self.canvas_w,
            self.canvas_h,
            fill_val,
            auto_expand=True,
            round_base=256,
        )

    def _rgb_to_train_masks(self, A_rgb: np.ndarray):
        shampoo = np.all(A_rgb == [0, 255, 0], axis=2).astype(np.uint8)
        blade = np.all(A_rgb == [0, 0, 255], axis=2).astype(np.uint8)
        return shampoo, blade

    def _mask_from_Aimg(self, A_img) -> np.ndarray:
        if isinstance(A_img, Image.Image):
            A = np.array(A_img)
        elif torch.is_tensor(A_img):
            A = A_img.detach().cpu().numpy()
            if A.ndim == 3:
                A = np.transpose(A, (1, 2, 0))
        else:
            A = np.array(A_img)
        return np.any(A > 0, axis=2) if A.ndim == 3 else (A > 0)

    def _mask_B_with_A(self, B_img: Image.Image, A_img: Image.Image, fill_value=0) -> Image.Image:
        B = np.array(B_img).copy()
        obj = self._mask_from_Aimg(A_img).astype(np.uint8)
        out = np.full_like(B, fill_value)
        out[obj > 0] = B[obj > 0]
        return Image.fromarray(out)

    def _extract_appearance_from_B(self, B_img: Image.Image, A_img: Image.Image) -> Image.Image:
        B_gray = np.array(B_img.convert("L")).astype(np.float32) / 255.0
        obj = self._mask_from_Aimg(A_img).astype(np.float32)
        app = B_gray * obj
        return Image.fromarray(np.clip(app * 255, 0, 255).astype(np.uint8), mode="L")

    _extract_object_grayscale_from_B = _extract_appearance_from_B

    def _zero_appearance_img(self, A_img: Image.Image) -> Image.Image:
        A = np.array(A_img)
        H, W = A.shape[:2]
        return Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L")

    def _weak_blur_appearance_img(self, app_img: Image.Image, A_img: Image.Image) -> Image.Image:
        app = np.array(app_img)
        if app.ndim == 3:
            app = cv2.cvtColor(app, cv2.COLOR_RGB2GRAY)
        obj = self._mask_from_Aimg(A_img).astype(np.uint8)
        if obj.sum() == 0:
            return self._zero_appearance_img(A_img)
        ksize = int(getattr(self.opt, "appearance_blur_ksize", 31))
        sigma = float(getattr(self.opt, "appearance_blur_sigma", 8.0))
        ksize = max(3, ksize + (0 if ksize % 2 else 1))
        blur = cv2.GaussianBlur(app, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        blur = (blur.astype(np.float32) * obj).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blur, mode="L")

    def _build_appearance_prototype_bank(self, max_items=200):
        bank = []
        for AB_path in self.AB_paths[:max_items]:
            if self._is_tray_sample(AB_path):
                continue
            try:
                AB = Image.open(AB_path).convert("RGB")
            except Exception:
                continue

            w, h = AB.size
            A_img = AB.crop((0, 0, w // 2, h))
            B_img = AB.crop((w // 2, 0, w, h))
            A_img = self._maybe_pad_rgb(A_img)
            B_img = self._maybe_pad_rgb(B_img)
            B_img = self._mask_B_with_A(B_img, A_img, fill_value=0)

            obj_mask = self._mask_from_Aimg(A_img).astype(np.uint8)
            ys, xs = np.where(obj_mask > 0)
            if not len(xs):
                continue

            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            B_gray = np.array(B_img.convert("L")).astype(np.uint8)
            obj_crop = B_gray[y0:y1, x0:x1].copy()
            mask_crop = obj_mask[y0:y1, x0:x1].copy()
            if mask_crop.sum() < 20:
                continue
            obj_crop[mask_crop == 0] = 0
            bank.append({"gray": obj_crop, "mask": mask_crop})
        return bank

    def _normalize_object_crop_to_canvas(self, obj_gray: np.ndarray, obj_mask: np.ndarray, out_hw):
        H, W = out_hw
        canvas = np.zeros((H, W), dtype=np.uint8)
        ys, xs = np.where(obj_mask > 0)
        if not len(xs):
            return canvas
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        bh, bw = max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)
        crop_r = cv2.resize(obj_gray, (bw, bh), interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(obj_mask * 255, (bw, bh), interpolation=cv2.INTER_NEAREST)
        patch = np.zeros((bh, bw), dtype=np.uint8)
        m = mask_r > 127
        patch[m] = crop_r[m]
        canvas[y0:y0 + bh, x0:x0 + bw] = patch
        return canvas

    def _sample_prototype_appearance_img(self, A_img: Image.Image) -> Image.Image:
        if not self.appearance_prototypes:
            return self._zero_appearance_img(A_img)
        proto = self.appearance_prototypes[np.random.randint(len(self.appearance_prototypes))]
        obj_mask = self._mask_from_Aimg(A_img).astype(np.uint8)
        canvas = self._normalize_object_crop_to_canvas(proto["gray"], obj_mask, obj_mask.shape[:2])
        canvas = (canvas.astype(np.float32) * obj_mask).clip(0, 255).astype(np.uint8)
        return Image.fromarray(canvas, mode="L")

    def _rgbmask_to_condition_tensor(self, A_img: Image.Image, app_img: Image.Image = None):
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)
        chs = [obj.astype(np.float32)]

        if self.use_edge_channel:
            chs.append(self._make_edge_map(obj))

        if self.use_thickness_channel:
            if obj.sum() > 0:
                dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5).astype(np.float32)
                dist = dist / (dist.max() + 1e-6)
            else:
                dist = np.zeros_like(obj, dtype=np.float32)
            chs.append(dist)

        if self.use_coord_channels:
            chs.extend(self._make_coord_maps(obj))

        if self.use_appearance_channel:
            if app_img is not None:
                app = np.array(app_img).astype(np.float32) / 255.0
            else:
                app = np.zeros_like(obj, dtype=np.float32)
            chs.append(app)

        cond = np.stack(chs, axis=0) * 2.0 - 1.0
        return torch.from_numpy(cond).float(), chs

    def _make_edge_map(self, obj: np.ndarray) -> np.ndarray:
        if not obj.sum():
            return np.zeros_like(obj, dtype=np.float32)
        px = max(1, int(getattr(self.opt, "edge_dilate_px", 1)))
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        edge = (cv2.dilate(obj, k) - cv2.erode(obj, k)) > 0
        return edge.astype(np.float32) * obj

    def _make_coord_maps(self, obj: np.ndarray):
        H, W = obj.shape[:2]
        coord_x = np.zeros((H, W), dtype=np.float32)
        coord_y = np.zeros((H, W), dtype=np.float32)
        ys, xs = np.where(obj > 0)
        if not len(xs):
            return coord_x, coord_y
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        x_norm = np.clip((np.arange(W, dtype=np.float32) - x0) / bw, 0, 1)
        y_norm = np.clip((np.arange(H, dtype=np.float32) - y0) / bh, 0, 1)
        objf = obj.astype(np.float32)
        coord_x = np.tile(x_norm[None, :], (H, 1)) * objf
        coord_y = np.tile(y_norm[:, None], (1, W)) * objf
        return coord_x, coord_y

    def _extract_instance_masks_tensor(self, A_img: Image.Image):
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
        insts = [
            torch.from_numpy((labels == k).astype(np.float32))
            for k in range(1, num)
            if int(stats[k, cv2.CC_STAT_AREA]) >= 20
        ]
        if not insts:
            return torch.zeros((0, obj.shape[0], obj.shape[1]), dtype=torch.float32)
        return torch.stack(insts, dim=0)

    def _build_condition_vis_from_channels(self, cond_chs):
        H, W = cond_chs[0].shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(min(3, len(cond_chs))):
            vis[..., c] = (np.clip(cond_chs[c], 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(vis, mode="RGB")

    def _infer_train_id_from_cutout_bgr(self, cut_bgr: np.ndarray) -> int:
        m = np.any(cut_bgr > 0, axis=2)
        if not np.any(m):
            return 0
        pix = cut_bgr[m].reshape(-1, 3)
        uniq, counts = np.unique(pix, axis=0, return_counts=True)
        bgr = tuple(uniq[np.argmax(counts)].tolist())
        if bgr in [(0, 255, 0), (255, 0, 0)]:
            return 1
        if bgr == (0, 0, 255):
            return 2
        return 0

    def _iter_semantic_cutout_paths(self, cutout_root: Path):
        sem_root = cutout_root / "semantic_rgba"
        if sem_root.exists() and sem_root.is_dir():
            for cls_dir in sorted(sem_root.iterdir()):
                if not cls_dir.is_dir():
                    continue
                for p in sorted(cls_dir.glob("*.png")):
                    yield p
            return

        for cls_dir in sorted(cutout_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            if cls_dir.name.lower() in {"gray", "mask", "preview", "semantic_rgba"}:
                continue
            for p in sorted(cls_dir.glob("*.png")):
                yield p

    def _find_gray_companion(self, p: Path):
        parent = p.parent
        cls_name = parent.name
        root = parent.parent.parent if parent.parent.name == "semantic_rgba" else parent.parent

        candidates = [
            root / "gray" / cls_name / p.name,
            root / "gray" / cls_name / f"{p.stem}_gray{p.suffix}",
            root / "gray" / cls_name / f"{p.stem}_real{p.suffix}",
            root / "gray" / cls_name / f"{p.stem}_b{p.suffix}",
            p.with_name(f"{p.stem}_gray{p.suffix}"),
            p.with_name(f"{p.stem}_real{p.suffix}"),
            p.with_name(f"{p.stem}_b{p.suffix}"),
            parent / "gray" / p.name,
            parent / "Gray" / p.name,
            parent / "grayscale" / p.name,
            parent / "real_B" / p.name,
            parent / "RealB" / p.name,
        ]

        for c in candidates:
            if c.exists():
                return c
        return None

    def _read_png_keep_alpha_mask(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        if img.ndim == 2:
            gray = img.copy()
            mask = gray > 0
            return gray, mask
        if img.shape[2] == 4:
            alpha = img[:, :, 3] > 0
            rgb = img[:, :, :3].copy()
            rgb[~alpha] = 0
            return rgb, alpha
        rgb = img[:, :, :3].copy()
        mask = np.any(rgb > 0, axis=2)
        return rgb, mask

    def _load_cutouts(self, cutout_root: Path):
        items = []
        if not cutout_root.exists():
            raise FileNotFoundError(f"cutout_dir not found: {cutout_root}")

        sem_paths = list(self._iter_semantic_cutout_paths(cutout_root))
        if not sem_paths:
            raise RuntimeError(
                f"No semantic cutouts found in {cutout_root}. "
                f"Expected semantic_rgba/<class>/*.png or <class>/*.png"
            )

        for p in sem_paths:
            sem_img, sem_mask = self._read_png_keep_alpha_mask(p)
            if sem_img is None or sem_mask is None or sem_img.ndim != 3:
                continue

            tid = self._infer_train_id_from_cutout_bgr(sem_img)
            if tid == 0:
                continue

            gray_img = None
            gray_path = self._find_gray_companion(p)
            if gray_path is not None:
                gray_raw, gray_mask = self._read_png_keep_alpha_mask(gray_path)
                if gray_raw is not None:
                    if gray_raw.ndim == 3:
                        gray_img = cv2.cvtColor(gray_raw, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_img = gray_raw.copy()
                    if gray_mask is not None:
                        gray_img[~gray_mask] = 0

            ys, xs = np.where(sem_mask)
            if not len(xs):
                continue

            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            sem_crop = sem_img[y0:y1, x0:x1].copy()
            mask_crop = sem_mask[y0:y1, x0:x1].copy()

            if gray_img is not None:
                if gray_img.shape[:2] != sem_img.shape[:2]:
                    gray_img = cv2.resize(
                        gray_img,
                        (sem_img.shape[1], sem_img.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                gray_crop = gray_img[y0:y1, x0:x1].copy()
                gray_crop[~mask_crop] = 0
            else:
                gray_crop = None

            items.append(
                {
                    "bgr": sem_crop,
                    "gray": gray_crop,
                    "mask": mask_crop.astype(np.uint8),
                    "train_id": tid,
                    "src_path": str(p),
                    "gray_path": str(gray_path) if gray_path is not None else "",
                }
            )

        if not items:
            raise RuntimeError(f"No valid cutouts in {cutout_root}")

        n_gray = sum(1 for it in items if it["gray"] is not None)
        print(f"[synthetic] loaded {len(items)} cutouts | grayscale companions: {n_gray}")
        if n_gray == 0:
            print("[warning] no grayscale companions found; synthetic B will fallback to flat grayscale semantic cutouts")
        return items

    def _rotate_preserve_gray_or_bgr(self, img: np.ndarray, angle_deg: float, is_mask=False):
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - cx
        M[1, 2] += nh / 2 - cy
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        border_value = 0 if img.ndim == 2 else (0, 0, 0)
        return cv2.warpAffine(
            img, M, (nw, nh), flags=interp,
            borderMode=cv2.BORDER_CONSTANT, borderValue=border_value,
        )

    def _transform_cutout(self, bgr: np.ndarray, gray: np.ndarray = None, mask: np.ndarray = None):
        out_bgr = bgr.copy()
        out_gray = None if gray is None else gray.copy()

        if mask is None:
            m = (np.any(out_bgr > 0, axis=2).astype(np.uint8) * 255)
        else:
            m = (mask.astype(np.uint8) * 255)

        rot = 0.0
        if getattr(self.opt, "phase", "") == "train":
            rmin = float(getattr(self.opt, "synthetic_rot_min", 0.0))
            rmax = float(getattr(self.opt, "synthetic_rot_max", 0.0))
            rot = np.random.uniform(rmin, rmax) if abs(rmax - rmin) > 1e-6 else rmin

        if abs(rot) > 1e-6:
            out_bgr = self._rotate_preserve_gray_or_bgr(out_bgr, rot, is_mask=False)
            m = self._rotate_preserve_gray_or_bgr(m, rot, is_mask=True)
            if out_gray is not None:
                out_gray = self._rotate_preserve_gray_or_bgr(out_gray, rot, is_mask=False)

        m = cv2.GaussianBlur(m, (5, 5), 0.8)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

        ys, xs = np.where(m > 0)
        if not len(xs):
            return None

        clean_bgr = out_bgr.copy()
        clean_bgr[m == 0] = 0

        if out_gray is not None:
            if out_gray.ndim == 3:
                out_gray = cv2.cvtColor(out_gray, cv2.COLOR_BGR2GRAY)
            clean_gray = out_gray.copy()
            clean_gray[m == 0] = 0
        else:
            clean_gray = None

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        mask_bool = (m[y0:y1, x0:x1] > 0)

        return {
            "bgr": clean_bgr[y0:y1, x0:x1].copy(),
            "gray": None if clean_gray is None else clean_gray[y0:y1, x0:x1].copy(),
            "mask": mask_bool,
        }

    def _binary_from_pil(self, img: Image.Image, thr255: int) -> np.ndarray:
        return (np.array(img.convert("L")) > thr255).astype(np.uint8)

    def _dilate_bin(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.dilate(m01.astype(np.uint8), k, iterations=1)

    def _largest_cc(self, m01: np.ndarray) -> np.ndarray:
        m = (m01 > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num <= 1:
            return m
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labels == k).astype(np.uint8)

    def _close(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.morphologyEx(m01.astype(np.uint8), cv2.MORPH_CLOSE, k)

    def _get_tray_bin(self, T_img: Image.Image) -> np.ndarray:
        thr255 = int(np.clip(float(getattr(self.opt, "tray_mask_thr", 0.5)) * 255, 0, 255))
        T = self._binary_from_pil(T_img, thr255)
        if bool(getattr(self.opt, "tray_mask_invert", False)):
            T = 1 - T
        T = self._largest_cc(T)
        T = self._close(T, int(getattr(self.opt, "tray_cc_close_px", 2)))
        dil = int(getattr(self.opt, "tray_mask_dilate_px", 0))
        return self._dilate_bin(T, dil) if dil > 0 else T

    def _resolve_tray_mask_for_ab(self, ab_path: str) -> Image.Image:
        if self.tray_mask_img is not None:
            return self.tray_mask_img.copy()

        if not self.tray_mask_paths:
            raise RuntimeError("Tray mask folder mode enabled, but no tray masks were loaded.")

        ab_stem = Path(ab_path).stem

        # Exact stem match
        for p in self.tray_mask_paths:
            if p.stem == ab_stem:
                return Image.open(str(p)).convert("L")

        # Loose match
        for p in self.tray_mask_paths:
            if ab_stem in p.stem or p.stem in ab_stem:
                return Image.open(str(p)).convert("L")

        # Fallback: random
        p = self.tray_mask_paths[np.random.randint(len(self.tray_mask_paths))]
        return Image.open(str(p)).convert("L")

    def _load_tray_T(self, target_size, ab_path=None):
        if ab_path is not None:
            T_img = self._resolve_tray_mask_for_ab(ab_path)
        else:
            if self.tray_mask_img is None:
                raise RuntimeError("No tray mask available.")
            T_img = self.tray_mask_img.copy()

        if T_img.size != target_size:
            T_img = T_img.resize(target_size, resample=Image.NEAREST)
        return T_img

    def _build_synthetic_pair_simple(self, size_hw, T_img: Image.Image):
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool) if T_img is not None else np.ones((H, W), dtype=bool)

        canvas_A = np.zeros((H, W, 3), dtype=np.uint8)
        canvas_B = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)

        n_obj = np.random.randint(
            int(getattr(self.opt, "synthetic_min_items", 1)),
            int(getattr(self.opt, "synthetic_max_items", 3)) + 1,
        )
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))
        same_class_prob = float(getattr(self.opt, "synthetic_same_class_prob", 0.0))

        chosen_tid = None
        if np.random.rand() < same_class_prob:
            tids = sorted({int(it["train_id"]) for it in self.cutout_items})
            if tids:
                chosen_tid = int(np.random.choice(tids))

        placed_any = False

        for _ in range(n_obj):
            cands = [it for it in self.cutout_items if chosen_tid is None or it["train_id"] == chosen_tid]
            item = cands[np.random.randint(len(cands))]
            cut_pack = self._transform_cutout(
                item["bgr"], gray=item.get("gray", None), mask=item.get("mask", None)
            )
            if cut_pack is None:
                continue

            cut_A = cut_pack["bgr"]
            cut_gray = cut_pack["gray"]
            obj_mask = cut_pack["mask"]

            h, w = cut_A.shape[:2]
            if h >= H or w >= W or h < 2 or w < 2:
                continue

            for _ in range(300):
                x = np.random.randint(0, W - w + 1)
                y = np.random.randint(0, H - h + 1)

                if not np.all(T[y:y + h, x:x + w][obj_mask]):
                    continue
                if no_overlap and np.any(occ[y:y + h, x:x + w] & obj_mask):
                    continue

                canvas_A[y:y + h, x:x + w][obj_mask] = cut_A[obj_mask]

                if cut_gray is not None:
                    gray3 = cv2.cvtColor(cut_gray, cv2.COLOR_GRAY2BGR)
                else:
                    fallback_gray = cv2.cvtColor(cut_A, cv2.COLOR_BGR2GRAY)
                    gray3 = cv2.cvtColor(fallback_gray, cv2.COLOR_GRAY2BGR)

                canvas_B[y:y + h, x:x + w][obj_mask] = gray3[obj_mask]
                occ[y:y + h, x:x + w][obj_mask] = True
                placed_any = True
                break

        if not placed_any:
            print("[warning] synthetic sample placed 0 objects; returning blank canvas")

        return (
            Image.fromarray(cv2.cvtColor(canvas_A, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(canvas_B, cv2.COLOR_BGR2RGB)),
        )

    def _to_gray_rgb(self, img: Image.Image) -> Image.Image:
        return img.convert("L").convert("RGB")

    def _apply_shared_geom_to_mask_rgb(self, img: Image.Image, params) -> Image.Image:
        if not self.opt.no_flip and params["flip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        is_tray = self._is_tray_sample(AB_path)

        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size

        A_img = AB.crop((0, 0, w // 2, h))
        B_img = AB.crop((w // 2, 0, w, h))

        A_img = self._maybe_pad_rgb(A_img)
        B_img = self._maybe_pad_rgb(B_img)

        use_synth = self.synthetic_enabled and (np.random.rand() < self.synthetic_prob) and (not is_tray)
        is_train = getattr(self.opt, "phase", "") == "train"

        T_img = None
        if self.use_tray_mask:
            T_img = self._load_tray_T(A_img.size, ab_path=AB_path)

        if use_synth:
            synth_h, synth_w = A_img.size[1], A_img.size[0]

            # Synthetic is for shampoo-only combinations, not tray composition.
            A_img, B_img = self._build_synthetic_pair_simple((synth_h, synth_w), None)

            A_img = self._maybe_pad_rgb(A_img)
            B_img = self._maybe_pad_rgb(B_img)
        else:
            if not is_tray:
                B_img = self._mask_B_with_A(B_img, A_img, fill_value=0)

        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)

        transform_params = get_params(self.opt, A_img.size)
        A_img_t = self._apply_shared_geom_to_mask_rgb(A_img, transform_params)

        T_img_t = None
        if self.use_tray_mask and T_img is not None:
            T_img_t = self._apply_shared_geom_to_mask_rgb(
                T_img.convert("RGB"), transform_params
            ).convert("L")

        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B_img)

        app_img_t = None
        if self.use_appearance_channel:
            app_raw = self._extract_appearance_from_B(B_img, A_img)
            app_raw = self._maybe_pad_gray(app_raw) if not use_synth else app_raw
            app_img_t = self._apply_shared_geom_to_mask_rgb(
                app_raw.convert("RGB"), transform_params
            ).convert("L")

            if is_train:
                p_zero = float(getattr(self.opt, "appearance_zero_prob", 0.35))
                p_weak = float(getattr(self.opt, "appearance_weak_prob", 0.35))
                p_proto = float(getattr(self.opt, "appearance_proto_prob", 0.15))
                r = np.random.rand()
                if r < p_zero:
                    app_img_t = self._zero_appearance_img(A_img_t)
                elif r < p_zero + p_weak:
                    app_img_t = self._weak_blur_appearance_img(app_img_t, A_img_t)
                elif r < p_zero + p_weak + p_proto:
                    app_img_t = self._sample_prototype_appearance_img(A_img_t)

            if not is_train and bool(getattr(self.opt, "disable_test_appearance", False)):
                app_img_t = self._zero_appearance_img(A_img_t)

        A, cond_chs = self._rgbmask_to_condition_tensor(A_img_t, app_img=app_img_t)
        A_vis = B_transform(self._build_condition_vis_from_channels(cond_chs))

        instance_masks = None
        if self.return_instance_masks:
            instance_masks = self._extract_instance_masks_tensor(A_img_t)

        T = None
        if self.use_tray_mask and T_img_t is not None:
            T_np = self._get_tray_bin(T_img_t).astype(np.float32)
            T = torch.from_numpy(T_np[None]).float()

        if is_train and (index % self.debug_every == 0):
            print(
                f"[debug] {Path(AB_path).name} type={'tray' if is_tray else 'shampoo'} "
                f"synth={use_synth} "
                f"A_size={A_img_t.size if isinstance(A_img_t, Image.Image) else 'na'} "
                f"B_size={B_img.size} "
                f"A({A.min():.3f},{A.max():.3f}) B({B.min():.3f},{B.max():.3f})"
            )

        out = {
            "A": A,
            "A_vis": A_vis,
            "B": B,
            "A_paths": AB_path,
            "B_paths": AB_path,
            "is_synthetic": use_synth,
            "is_tray": is_tray,
        }
        if T is not None:
            out["T"] = T
        if instance_masks is not None:
            out["instance_masks"] = instance_masks

        return out

    def __len__(self):
        return len(self.AB_paths)