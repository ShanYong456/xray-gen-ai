import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


def pad_to_canvas(img: Image.Image, target_w: int, target_h: int, fill=(0, 0, 0)):
    w, h = img.size

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
    """
    Paired {A, B} dataset for structured-condition pix2pix.

    A = conditioning tensor built from mask-style annotation image:
        class_nc=2:
            ch0   = shampoo mask
            ch1   = tray mask
            ch2   = edge map                    (if use_edge_channel)
            ch3   = distance transform          (if use_thickness_channel)
            ch4,5 = local coord maps x, y      (if use_coord_channels)
            chN   = masked appearance from B   (if use_appearance_channel)

        class_nc=1 fallback:
            ch0   = merged object mask
            ...

    B = target image

    Optional:
        T = tray mask to restrict synthetic object placement
        instance_masks = connected components from A
    """

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
        parser.add_argument("--tray_scale", type=float, default=1.0)

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
        parser.add_argument("--shampoo_horizontal_shift_only", action="store_true")
        parser.add_argument("--shampoo_max_horizontal_shift", type=int, default=0)
        parser.add_argument("--shampoo_max_vertical_shift", type=int, default=0)

        parser.add_argument("--disable_test_appearance", action="store_true")
        parser.add_argument("--appearance_dropout", type=float, default=0.5)
        parser.add_argument(
            "--appearance_zero_prob",
            type=float,
            default=0.35,
            help="Prob of zero appearance (fully unguided).",
        )
        parser.add_argument(
            "--appearance_weak_prob",
            type=float,
            default=0.35,
            help="Prob of blurred appearance (weakly guided).",
        )
        parser.add_argument(
            "--appearance_proto_prob",
            type=float,
            default=0.15,
            help="Prob of prototype appearance (class-guided).",
        )
        parser.add_argument("--appearance_blur_ksize", type=int, default=31)
        parser.add_argument("--appearance_blur_sigma", type=float, default=8.0)
        parser.add_argument("--build_appearance_prototypes", action="store_true")
        parser.add_argument("--max_appearance_prototypes", type=int, default=200)

        parser.add_argument("--use_edge_channel", action="store_true")
        parser.add_argument("--edge_dilate_px", type=int, default=1)
        parser.add_argument("--use_coord_channels", action="store_true")

        parser.add_argument("--synthetic_place_tries", type=int, default=120)
        parser.add_argument("--synthetic_item_retries", type=int, default=12)
        parser.add_argument("--synthetic_erode_px", type=int, default=6)
        parser.add_argument("--synthetic_fallback_shrink", type=float, default=0.85)
        parser.add_argument("--synthetic_sort_large_first", action="store_true")

        parser.add_argument("--mask_aug_px", type=int, default=2)

        parser.add_argument(
            "--pad_to_canvas",
            action="store_true",
            help="Pad A/B to a fixed canvas instead of resizing.",
        )
        parser.add_argument("--canvas_w", type=int, default=1024)
        parser.add_argument("--canvas_h", type=int, default=1536)
        parser.add_argument(
            "--canvas_fill",
            type=int,
            default=0,
            help="Fill value for padded RGB canvas. 0=black, 235=light gray.",
        )

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
        self.canvas_w = int(getattr(opt, "canvas_w", 1024))
        self.canvas_h = int(getattr(opt, "canvas_h", 1536))
        fill_val = int(getattr(opt, "canvas_fill", 0))
        self.canvas_fill_rgb = (fill_val, fill_val, fill_val)

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

        # FAST lookup map (replace O(N) scan per sample)
        self.tray_mask_map = {}
        if self.tray_mask_paths:
            for p in self.tray_mask_paths:
                self.tray_mask_map[p.stem] = str(p)

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
    

    def normalize_xray_intensity(self, img_np):
        img = img_np.astype(np.float32)

        p1, p99 = np.percentile(img, (1, 99))
        img = np.clip(img, p1, p99)

        img = (img - p1) / (p99 - p1 + 1e-6)

        return (img * 255).astype(np.uint8)

    def _resolve_tray_mask_for_ab(self, ab_path: str) -> Image.Image:
        if self.tray_mask_img is not None:
            return self.tray_mask_img.copy()

        ab_stem = Path(ab_path).stem

        # O(1) lookup
        hit = self.tray_mask_map.get(ab_stem, None)
        if hit is not None:
            return Image.open(hit).convert("L")

        # fallback (rare)
        return Image.open(str(self.tray_mask_paths[0])).convert("L")
        
    def _load_tray_T(self, target_size, ab_path=None):
        if ab_path is not None:
            T_img = self._resolve_tray_mask_for_ab(ab_path)
        else:
            if self.tray_mask_img is not None:
                T_img = self.tray_mask_img.copy()
            elif self.tray_mask_paths:
                p = self.tray_mask_paths[np.random.randint(len(self.tray_mask_paths))]
                T_img = Image.open(str(p)).convert("L")
            else:
                raise RuntimeError("No tray mask available.")

        tray_scale = float(getattr(self.opt, "tray_scale", 1.0))
        T_arr = np.array(T_img)

        if abs(tray_scale - 1.0) > 1e-6:
            new_w = max(1, int(round(T_arr.shape[1] * tray_scale)))
            new_h = max(1, int(round(T_arr.shape[0] * tray_scale)))
            T_arr = cv2.resize(T_arr, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            T_img = Image.fromarray(T_arr, mode="L")

        if self.pad_to_canvas_enabled:
            tw, th = T_img.size
            scale = min(self.canvas_w / tw, self.canvas_h / th)

            if scale < 1.0:
                new_w = max(1, int(round(tw * scale)))
                new_h = max(1, int(round(th * scale)))
                T_img = T_img.resize((new_w, new_h), resample=Image.NEAREST)

            T_img = pad_to_canvas(
                T_img,
                self.canvas_w,
                self.canvas_h,
                self.canvas_fill_rgb[0]
            )

        return T_img
    def _maybe_pad_rgb(self, img: Image.Image) -> Image.Image:
        if not self.pad_to_canvas_enabled:
            return img
        return pad_to_canvas(img, self.canvas_w, self.canvas_h, self.canvas_fill_rgb)

    def _maybe_pad_gray(self, img: Image.Image) -> Image.Image:
        if not self.pad_to_canvas_enabled:
            return img
        fill_val = self.canvas_fill_rgb[0]
        return pad_to_canvas(img, self.canvas_w, self.canvas_h, fill_val)

    def _train_id_to_rgb(self, train_id: int):
        return {
            1: np.array([0, 255, 0], dtype=np.uint8),
            2: np.array([0, 0, 255], dtype=np.uint8),
        }.get(int(train_id), np.array([0, 0, 0], dtype=np.uint8))

    def _rgb_to_train_masks(self, A_rgb: np.ndarray):
        """
        Semantic colors in A annotation image:
        shampoo only = green  (0,255,0)
        tray only    = blue   (0,0,255)
        overlap      = cyan   (0,255,255)

        Returns:
        shampoo_mask, tray_mask
        """
        shampoo_only = np.all(A_rgb == [0, 255, 0], axis=2)
        tray_only = np.all(A_rgb == [0, 0, 255], axis=2)
        overlap = np.all(A_rgb == [0, 255, 255], axis=2)

        shampoo = (shampoo_only | overlap).astype(np.uint8)
        tray = (tray_only | overlap).astype(np.uint8)

        return shampoo, tray

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

        if B.ndim == 3:
            out = np.full_like(B, fill_value)
            out[obj > 0] = B[obj > 0]
        else:
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
            try:
                AB = Image.open(AB_path).convert("RGB")
            except Exception:
                continue

            w, h = AB.size
            A_img = AB.crop((0, 0, w // 2, h))
            B_img = AB.crop((w // 2, 0, w, h))

            A_img = self._maybe_pad_rgb(A_img)
            B_img = self._maybe_pad_rgb(B_img)

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
        """
        Stage 10 class_nc=2 conditioning layout:

        ch0 = shampoo mask
        ch1 = tray mask
        ch2 = edge map              (if use_edge_channel)
        ch3 = thickness map         (if use_thickness_channel)
        ch4 = coord_x               (if use_coord_channels)
        ch5 = coord_y               (if use_coord_channels)
        chN = appearance            (if use_appearance_channel)

        Old fallback for class_nc=1 is kept for compatibility.
        """
        A = np.array(A_img).astype(np.uint8)
        shampoo, tray = self._rgb_to_train_masks(A)

        if self.class_nc == 2:
            obj_any = ((shampoo > 0) | (tray > 0)).astype(np.uint8)
            chs = [
                shampoo.astype(np.float32),
                tray.astype(np.float32),
            ]

            if self.use_edge_channel:
                chs.append(self._make_edge_map(obj_any))

            if self.use_thickness_channel:
                if obj_any.sum() > 0:
                    dist = cv2.distanceTransform(obj_any, cv2.DIST_L2, 5).astype(np.float32)
                    dist = dist / (dist.max() + 1e-6)
                else:
                    dist = np.zeros_like(obj_any, dtype=np.float32)
                chs.append(dist)

            if self.use_coord_channels:
                chs.extend(self._make_coord_maps(obj_any))

            if self.use_appearance_channel:
                if app_img is not None:
                    app = np.array(app_img).astype(np.float32) / 255.0
                else:
                    app = np.zeros_like(obj_any, dtype=np.float32)
                chs.append(app)

        else:
            obj = ((shampoo > 0) | (tray > 0)).astype(np.uint8)
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
        shampoo, tray = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (tray > 0)).astype(np.uint8)

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
        """
        For class_nc=2:
        R = edge
        G = shampoo
        B = tray
        """
        H, W = cond_chs[0].shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)

        if self.class_nc == 2 and len(cond_chs) >= 2:
            shampoo = (np.clip(cond_chs[0], 0, 1) * 255).astype(np.uint8)
            tray = (np.clip(cond_chs[1], 0, 1) * 255).astype(np.uint8)
            edge = (np.clip(cond_chs[2], 0, 1) * 255).astype(np.uint8) if len(cond_chs) >= 3 else np.zeros((H, W), dtype=np.uint8)

            vis[..., 1] = shampoo
            vis[..., 2] = tray
            vis[..., 0] = edge
            return Image.fromarray(vis, mode="RGB")

        for c in range(min(3, len(cond_chs))):
            vis[..., c] = (np.clip(cond_chs[c], 0, 1) * 255).astype(np.uint8)

        return Image.fromarray(vis, mode="RGB")

    # ---------------- Synthetic generation ----------------

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

        candidates = []

        gray_same_name = root / "gray" / cls_name / p.name
        candidates.append(gray_same_name)

        candidates.extend([
            root / "gray" / cls_name / f"{p.stem}_gray{p.suffix}",
            root / "gray" / cls_name / f"{p.stem}_real{p.suffix}",
            root / "gray" / cls_name / f"{p.stem}_b{p.suffix}",
        ])

        candidates.extend([
            p.with_name(f"{p.stem}_gray{p.suffix}"),
            p.with_name(f"{p.stem}_real{p.suffix}"),
            p.with_name(f"{p.stem}_b{p.suffix}"),
            parent / "gray" / p.name,
            parent / "Gray" / p.name,
            parent / "grayscale" / p.name,
            parent / "real_B" / p.name,
            parent / "RealB" / p.name,
        ])

        for c in candidates:
            if c.exists():
                return c

        return None
    
    def _erode_bin(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01.astype(np.uint8)
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.erode(m01.astype(np.uint8), k, iterations=1)

    def _valid_anchor_positions(self, T, obj_mask, occ=None, no_overlap=False):
        H, W = T.shape
        h, w = obj_mask.shape

        if h > H or w > W:
            return []

        # Fast bbox-based candidate generation instead of huge-mask erosion
        ys, xs = np.where(T > 0)
        if len(xs) == 0:
            return []

        tx0, tx1 = int(xs.min()), int(xs.max())
        ty0, ty1 = int(ys.min()), int(ys.max())

        # Anchor search range restricted to tray bbox
        x_start = max(0, tx0)
        y_start = max(0, ty0)
        x_end = min(W - w, tx1 - w + 1)
        y_end = min(H - h, ty1 - h + 1)

        if x_end < x_start or y_end < y_start:
            return []

        # Subsample candidates for speed
        step = max(4, int(getattr(self.opt, "x_search_step", 8))) if hasattr(self.opt, "x_search_step") else 8

        coords = []
        max_candidates = int(getattr(self.opt, "synthetic_place_tries", 10))

        for y in range(y_start, y_end + 1, step):
            for x in range(x_start, x_end + 1, step):
                tray_patch = T[y:y+h, x:x+w]
                if tray_patch.shape[0] != h or tray_patch.shape[1] != w:
                    continue

                # Must fit fully inside tray
                if not np.all(tray_patch[obj_mask]):
                    continue

                if no_overlap and occ is not None:
                    if np.any(occ[y:y+h, x:x+w] & obj_mask):
                        continue

                coords.append((x, y))
                if len(coords) >= max_candidates:
                    return coords

        return coords
    

    def _read_png_keep_alpha_mask(self, p: Path):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None

        # grayscale image
        if img.ndim == 2:
            gray = img.copy()
            mask = gray > 0
            return gray, mask

        # RGBA image
        if img.shape[2] == 4:
            alpha = img[:, :, 3] > 0
            rgb = img[:, :, :3].copy()
            rgb[~alpha] = 0
            return rgb, alpha

        # normal RGB/BGR image
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

            # HARD REQUIREMENT:
            # shampoo (train_id == 1) must have real grayscale companion
            if tid == 1 and gray_path is None:
                print(f"[skip] shampoo missing grayscale companion: {p}")
                continue

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
                gray_crop = self.normalize_xray_intensity(gray_crop)
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

        n_total = len(items)
        n_gray = sum(1 for it in items if it["gray"] is not None)
        n_shampoo = sum(1 for it in items if it["train_id"] == 1)
        n_shampoo_gray = sum(1 for it in items if it["train_id"] == 1 and it["gray"] is not None)
        n_tray = sum(1 for it in items if it["train_id"] == 2)
        n_tray_gray = sum(1 for it in items if it["train_id"] == 2 and it["gray"] is not None)

        print(f"[synthetic] total_cutouts={n_total} | with_gray={n_gray}")
        print(f"[synthetic] shampoo={n_shampoo} | shampoo_with_gray={n_shampoo_gray}")
        print(f"[synthetic] tray={n_tray} | tray_with_gray={n_tray_gray}")

        if n_shampoo == 0:
            raise RuntimeError("No shampoo cutouts loaded.")
        if n_shampoo_gray < n_shampoo:
            raise RuntimeError("Some shampoo cutouts are missing grayscale companions. Fix cutout_dir first.")

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
            img,
            M,
            (nw, nh),
            flags=interp,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
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

        # keep both hard mask and soft mask
        m_blur = cv2.GaussianBlur(m.astype(np.float32), (7, 7), 1.5)
        m_soft = np.clip(m_blur / 255.0, 0.0, 1.0)
        m_hard = (m_soft > 0.5).astype(np.uint8)

        ys, xs = np.where(m_hard > 0)
        if not len(xs):
            return None

        # hard requirement: shampoo synthetic object must carry grayscale
        if out_gray is None:
            return None

        clean_bgr = out_bgr.copy()
        clean_bgr[m_hard == 0] = 0

        if out_gray.ndim == 3:
            out_gray = cv2.cvtColor(out_gray, cv2.COLOR_BGR2GRAY)

        clean_gray = out_gray.copy()
        clean_gray[m_hard == 0] = 0
        clean_gray = self.normalize_xray_intensity(clean_gray)

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        return {
            "bgr": clean_bgr[y0:y1, x0:x1].copy(),
            "gray": clean_gray[y0:y1, x0:x1].copy(),
            "mask": m_hard[y0:y1, x0:x1].astype(bool),
            "soft_mask": m_soft[y0:y1, x0:x1].astype(np.float32),
        }

    def _random_mask_augment(self, mask: np.ndarray) -> np.ndarray:
        m = (mask > 0).astype(np.uint8)
        if m.sum() < 20:
            return m
        max_px = int(getattr(self.opt, "mask_aug_px", 2))
        if np.random.rand() < 0.7:
            px = np.random.randint(1, max_px + 1)
            k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
            m = cv2.dilate(m, k) if np.random.rand() < 0.5 else cv2.erode(m, k)
        if np.random.rand() < 0.5:
            _, m = cv2.threshold(
                cv2.GaussianBlur((m * 255).astype(np.uint8), (5, 5), 0.8),
                127,
                1,
                cv2.THRESH_BINARY,
            )
        return m.astype(np.uint8)

    def _build_synthetic_pair_simple(self, size_hw, T_img: Image.Image):
        H, W = size_hw

        # binary tray mask
        T = self._get_tray_bin(T_img).astype(np.uint8)
        T_place = self._erode_bin(T, 2)
        if T_place.sum() == 0:
            T_place = T

        # -----------------------------
        # A: semantic conditioning image
        # -----------------------------
        canvas_A = np.zeros((H, W, 3), dtype=np.uint8)
        tray_only_bgr = np.array([255, 0, 0], dtype=np.uint8)   # tray
        overlap_bgr   = np.array([255, 255, 0], dtype=np.uint8) # shampoo-in-tray
        canvas_A[T > 0] = tray_only_bgr

        # -----------------------------
        # B: REAL tray grayscale base
        # -----------------------------
        T_gray = np.array(T_img.convert("L")).astype(np.float32)
        if T_gray.shape != (H, W):
            T_gray = cv2.resize(T_gray, (W, H), interpolation=cv2.INTER_LINEAR)

        canvas_B = T_gray.copy()

        occ = np.zeros((H, W), dtype=bool)
        max_item_trials = max(4, int(getattr(self.opt, "synthetic_item_retries", 4)))
        placed = False

        for _ in range(max_item_trials):
            item = self.cutout_items[np.random.randint(len(self.cutout_items))]

            # only shampoo for this simple stage
            if int(item["train_id"]) != 1:
                continue

            cut = self._transform_cutout(
                item["bgr"],
                gray=item.get("gray", None),
                mask=item.get("mask", None),
            )
            if cut is None:
                continue

            cut_gray = cut["gray"]
            obj_mask = cut["mask"]
            obj_soft = cut.get("soft_mask", obj_mask.astype(np.float32))

            if cut_gray is None:
                continue

            smin = float(getattr(self.opt, "synthetic_scale_min", 0.85))
            smax = float(getattr(self.opt, "synthetic_scale_max", 0.85))
            scale = smin if abs(smax - smin) < 1e-6 else float(np.random.uniform(smin, smax))

            h0, w0 = obj_mask.shape
            new_w = max(1, int(round(w0 * scale)))
            new_h = max(1, int(round(h0 * scale)))

            if new_w > W or new_h > H:
                shrink = min(W / max(new_w, 1), H / max(new_h, 1), 1.0)
                new_w = max(1, int(round(new_w * shrink)))
                new_h = max(1, int(round(new_h * shrink)))

            obj_mask = cv2.resize(
                (obj_mask.astype(np.uint8) * 255),
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            ) > 127

            obj_soft = cv2.resize(obj_soft.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            obj_soft = np.clip(obj_soft, 0.0, 1.0)

            cut_gray = cv2.resize(cut_gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            cut_gray = self.normalize_xray_intensity(cut_gray)

            h, w = obj_mask.shape
            if h <= 0 or w <= 0 or h > H or w > W:
                continue

            anchors = self._valid_anchor_positions(
                T_place.astype(bool),
                obj_mask,
                occ,
                bool(getattr(self.opt, "synthetic_no_overlap", False)),
            )
            if not anchors:
                continue

            x, y = anchors[np.random.randint(len(anchors))]

            # confirm object really sits in tray
            region_A = canvas_A[y:y+h, x:x+w]
            tray_here = np.all(region_A == tray_only_bgr, axis=2)
            if not np.all(tray_here[obj_mask]):
                continue

            # write semantic overlap into A
            region_A[obj_mask] = overlap_bgr

            # -----------------------------
            # B compositing: attenuation-style blending
            # -----------------------------
            region_B = canvas_B[y:y+h, x:x+w].astype(np.float32)

            tray01 = np.clip(region_B / 255.0, 1e-4, 1.0)
            obj01  = np.clip(cut_gray / 255.0, 1e-4, 1.0)

            tray_abs = -np.log(tray01)
            obj_abs  = -np.log(obj01)

            # thickness-aware scaling
            atten_scale = float(scale ** 0.7)

            # add only where object exists
            target_abs = tray_abs + obj_abs * atten_scale

            # soft blend edge
            blended_abs = tray_abs * (1.0 - obj_soft) + target_abs * obj_soft

            # back to intensity
            blended = np.exp(-blended_abs)

            # small interior reinforcement
            obj_norm = (cut_gray - cut_gray.min()) / (cut_gray.max() - cut_gray.min() + 1e-6)
            blended = blended - (obj_norm * 0.03 * obj_soft)

            blended = np.clip(blended, 0.0, 1.0)
            canvas_B[y:y+h, x:x+w] = blended * 255.0

            occ[y:y+h, x:x+w][obj_mask] = True
            placed = True
            break

        if not placed:
            print("[warning] synthetic sample placed 0 objects; returning tray only")

        # final cleanup
        canvas_B = np.clip(canvas_B, 0, 255)

        # small sensor-like noise
        canvas_B = canvas_B + np.random.randn(H, W).astype(np.float32) * 1.2
        canvas_B = np.clip(canvas_B, 0, 255)

        canvas_B = self.normalize_xray_intensity(canvas_B)
        canvas_B = cv2.cvtColor(canvas_B.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return (
            Image.fromarray(cv2.cvtColor(canvas_A, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(canvas_B, cv2.COLOR_BGR2RGB)),
        )

    def _to_gray_rgb(self, img: Image.Image) -> Image.Image:
        return img.convert("L").convert("RGB")

    def _binary_from_pil(self, img: Image.Image, thr255: int) -> np.ndarray:
        return (np.array(img.convert("L")) > thr255).astype(np.uint8)

    def _shift_np(self, arr: np.ndarray, dx: int, dy: int, fill=0) -> np.ndarray:
        H, W = arr.shape[:2]
        out = np.full_like(arr, fill)
        x0s, x1s = max(0, -dx), min(W, W - dx) if dx >= 0 else W
        y0s, y1s = max(0, -dy), min(H, H - dy) if dy >= 0 else H
        x0d, y0d = max(0, dx), max(0, dy)
        if x1s > x0s and y1s > y0s:
            out[y0d:y0d + (y1s - y0s), x0d:x0d + (x1s - x0s)] = arr[y0s:y1s, x0s:x1s]
        return out

    def _bbox_from_binary(self, m: np.ndarray):
        ys, xs = np.where(m > 0)
        if not len(xs):
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _dilate_bin(self, m01: np.ndarray, px: int) -> np.ndarray:
        if px <= 0:
            return m01
        k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
        return cv2.dilate(m01.astype(np.uint8), k, iterations=1)

    def _clamp_shift_to_image(self, M: np.ndarray, dx: int, dy: int):
        H, W = M.shape[:2]
        bb = self._bbox_from_binary(M)
        if bb is None:
            return 0, 0
        x0, y0, x1, y1 = bb
        return int(np.clip(dx, -x0, (W - 1) - x1)), int(np.clip(dy, -y0, (H - 1) - y1))

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

    def _bbox_shift_into_tray(self, M, T, margin):
        ob, tb = self._bbox_from_binary(M), self._bbox_from_binary(T)
        if ob is None or tb is None:
            return 0, 0
        ox0, oy0, ox1, oy1 = ob
        tx0, ty0, tx1, ty1 = tb[0] + margin, tb[1] + margin, tb[2] - margin, tb[3] - margin
        if ox1 - ox0 > tx1 - tx0 or oy1 - oy0 > ty1 - ty0:
            return 0, 0
        dx = max(0, tx0 - ox0) - max(0, ox1 - tx1)
        dy = max(0, ty0 - oy0) - max(0, oy1 - ty1)
        return int(dx), int(dy)

    def _nudge_pixels_inside_tray(self, M, T, dx0, dy0):
        max_total = int(getattr(self.opt, "tray_shift_max_px", 400))
        n_iters = int(getattr(self.opt, "tray_nudge_iters", 8))
        max_step = int(getattr(self.opt, "tray_nudge_max_step", 20))
        dx_total, dy_total = self._clamp_shift_to_image(M, int(dx0), int(dy0))
        inv = 1 - (T == 1).astype(np.uint8)
        dist, labels = cv2.distanceTransformWithLabels(
            inv, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL
        )
        H, W = T.shape

        def score(dx, dy):
            Ms = self._shift_np(M, dx, dy, fill=0)
            return int(((Ms == 1) & (T == 0)).sum()), int(dx * dx + dy * dy), Ms

        best_dx, best_dy = dx_total, dy_total
        best_out, best_mag, M_work = score(dx_total, dy_total)
        if best_out == 0:
            return best_dx, best_dy, True

        for _ in range(n_iters):
            ys, xs = np.where((M_work == 1) & (T == 0))
            if not len(xs):
                return dx_total, dy_total, True
            idx = labels[ys, xs].astype(np.int64) - 1
            vx, vy = (idx % W).astype(np.int64) - xs, (idx // W).astype(np.int64) - ys
            sdx = int(np.clip(np.median(vx), -max_step, max_step))
            sdy = int(np.clip(np.median(vy), -max_step, max_step))
            if sdx == 0 and sdy == 0:
                break
            cand_dx = int(np.clip(dx_total + sdx, -max_total, max_total))
            cand_dy = int(np.clip(dy_total + sdy, -max_total, max_total))
            cand_dx, cand_dy = self._clamp_shift_to_image(M, cand_dx, cand_dy)
            co, cm, cM = score(cand_dx, cand_dy)
            if co < best_out or (co == best_out and cm < best_mag):
                best_dx, best_dy, best_out, best_mag = cand_dx, cand_dy, co, cm
            dx_total, dy_total, M_work = cand_dx, cand_dy, cM
            if co == 0:
                return cand_dx, cand_dy, True

        return best_dx, best_dy, best_out == 0

    def _compute_autoshift(self, A_img, T_img):
        if not bool(getattr(self.opt, "tray_mask_autoshift", False)):
            return 0, 0
        T = self._get_tray_bin(T_img)
        A_arr = np.array(A_img)
        M = (np.any(A_arr > 0, axis=2) if A_arr.ndim == 3 else A_arr > 0).astype(np.uint8)
        if not ((M == 1) & (T == 0)).sum():
            return 0, 0
        dx0, dy0 = self._bbox_shift_into_tray(M, T, int(getattr(self.opt, "tray_bbox_margin", 2)))
        dx, dy, success = self._nudge_pixels_inside_tray(M, T, dx0, dy0)
        if not success:
            print(f"[warning] object could not fully fit inside tray; best shift ({dx},{dy})")
        return int(dx), int(dy)

    def _apply_shared_geom_to_mask_rgb(self, img: Image.Image, params) -> Image.Image:
        if not self.opt.no_flip and params["flip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
    def _scale_pair(self, A_img: Image.Image, B_img: Image.Image, scale=0.7):
        W, H = A_img.size

        new_w = int(W * scale)
        new_h = int(H * scale)

        A_resized = A_img.resize((new_w, new_h), Image.NEAREST)
        B_resized = B_img.resize((new_w, new_h), Image.BILINEAR)

        # paste back to canvas center
        canvas_A = Image.new("RGB", (W, H), (0, 0, 0))
        canvas_B = Image.new("RGB", (W, H), (0, 0, 0))

        x = (W - new_w) // 2
        y = (H - new_h) // 2

        canvas_A.paste(A_resized, (x, y))
        canvas_B.paste(B_resized, (x, y))

        return canvas_A, canvas_B

    def _fit_pair_to_canvas(self, A_img: Image.Image, B_img: Image.Image, target_w: int, target_h: int):
        w, h = A_img.size
        scale = min(target_w / w, target_h / h, 1.0)

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        if (new_w, new_h) != (w, h):
            A_img = A_img.resize((new_w, new_h), resample=Image.NEAREST)
            B_img = B_img.resize((new_w, new_h), resample=Image.BILINEAR)

        return A_img, B_img
    
    def _scale_shampoo_only(self, A_img: Image.Image, B_img: Image.Image, scale: float):
        if abs(scale - 1.0) < 1e-6 and \
        int(getattr(self.opt, "shampoo_max_horizontal_shift", 0)) == 0 and \
        int(getattr(self.opt, "shampoo_max_vertical_shift", 0)) == 0:
            return A_img, B_img

        A = np.array(A_img).copy()
        B = np.array(B_img).copy()

        # semantic masks
        shampoo_mask = np.all(A == [0, 255, 0], axis=2).astype(np.uint8)
        tray_mask = np.all(A == [0, 0, 255], axis=2).astype(np.uint8)

        ys, xs = np.where(shampoo_mask > 0)
        if len(xs) == 0:
            return A_img, B_img

        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1

        # crop shampoo from A and B
        A_crop = A[y0:y1, x0:x1].copy()
        B_crop = B[y0:y1, x0:x1].copy()
        M_crop = shampoo_mask[y0:y1, x0:x1].copy()

        # keep only shampoo pixels
        A_obj = np.zeros_like(A_crop)
        A_obj[M_crop > 0] = A_crop[M_crop > 0]

        if B.ndim == 3:
            B_obj = np.zeros_like(B_crop)
            B_obj[M_crop > 0] = B_crop[M_crop > 0]
        else:
            B_obj = np.zeros_like(B_crop)
            B_obj[M_crop > 0] = B_crop[M_crop > 0]

        old_h, old_w = M_crop.shape
        new_w = max(1, int(round(old_w * scale)))
        new_h = max(1, int(round(old_h * scale)))

        # resize shampoo only
        A_obj_r = cv2.resize(A_obj, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        B_obj_r = cv2.resize(B_obj, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        M_r = cv2.resize((M_crop * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        M_r = (M_r > 127).astype(np.uint8)

        A_obj_r[M_r == 0] = 0
        if B_obj_r.ndim == 3:
            B_obj_r[M_r == 0] = 0
        else:
            B_obj_r[M_r == 0] = 0

        # remove original shampoo, keep tray untouched
        A_new = A.copy()
        A_new[shampoo_mask > 0] = 0

        B_new = B.copy()
        B_new[shampoo_mask > 0] = 0

        # original center
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2

        # random shift
        max_dx = int(getattr(self.opt, "shampoo_max_horizontal_shift", 0))
        max_dy = int(getattr(self.opt, "shampoo_max_vertical_shift", 0))
        horizontal_only = bool(getattr(self.opt, "shampoo_horizontal_shift_only", False))

        dx = np.random.randint(-max_dx, max_dx + 1) if max_dx > 0 else 0
        if horizontal_only:
            dy = 0
        else:
            dy = np.random.randint(-max_dy, max_dy + 1) if max_dy > 0 else 0

        # new center after shift
        cx = cx + dx
        cy = cy + dy

        # paste resized shampoo back
        px0 = max(0, cx - new_w // 2)
        py0 = max(0, cy - new_h // 2)
        px1 = min(A.shape[1], px0 + new_w)
        py1 = min(A.shape[0], py0 + new_h)

        src_w = px1 - px0
        src_h = py1 - py0

        A_patch = A_obj_r[:src_h, :src_w]
        B_patch = B_obj_r[:src_h, :src_w]
        M_patch = M_r[:src_h, :src_w] > 0

        roiA = A_new[py0:py1, px0:px1]
        roiA[M_patch] = A_patch[M_patch]
        A_new[py0:py1, px0:px1] = roiA

        roiB = B_new[py0:py1, px0:px1]
        roiB[M_patch] = B_patch[M_patch]
        B_new[py0:py1, px0:px1] = roiB

        # restore tray after paste
        A_new[tray_mask > 0] = np.array([0, 0, 255], dtype=np.uint8)

        return Image.fromarray(A_new), Image.fromarray(B_new)
    
    def pad_to_128_np(self, img):
        h, w = img.shape[:2]
        pad_h = (128 - h % 128) % 128
        pad_w = (128 - w % 128) % 128

        if img.ndim == 3:
            return np.pad(img, ((0,pad_h),(0,pad_w),(0,0)), mode='constant')
        else:
            return np.pad(img, ((0,pad_h),(0,pad_w)), mode='constant')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size

        A_img = AB.crop((0, 0, w // 2, h))
        B_img = AB.crop((w // 2, 0, w, h))

        # Fit + pad
        A_img, B_img = self._fit_pair_to_canvas(A_img, B_img, self.canvas_w, self.canvas_h)
        A_img = pad_to_canvas(A_img, self.canvas_w, self.canvas_h, self.canvas_fill_rgb)
        B_img = pad_to_canvas(B_img, self.canvas_w, self.canvas_h, self.canvas_fill_rgb)

        use_synth = self.synthetic_enabled and (np.random.rand() < self.synthetic_prob)
        is_train = getattr(self.opt, "phase", "") == "train"
        synth_preview_img = None

        T_img = None
        if self.use_tray_mask:
            T_img = self._load_tray_T(A_img.size, ab_path=AB_path)

        if use_synth:
            synth_h, synth_w = A_img.size[1], A_img.size[0]
            A_img, B_img = self._build_synthetic_pair_simple((synth_h, synth_w), T_img)
            synth_preview_img = B_img.copy()
            A_img = self._maybe_pad_rgb(A_img)
            B_img = self._maybe_pad_rgb(B_img)
            synth_preview_img = self._maybe_pad_rgb(synth_preview_img)
        else:
            shampoo_scale = float(getattr(self.opt, "synthetic_scale_min", 1.0))
            A_img, B_img = self._scale_shampoo_only(A_img, B_img, shampoo_scale)
            A_img = self._maybe_pad_rgb(A_img)
            B_img = self._maybe_pad_rgb(B_img)

        A_np = np.array(A_img)
        B_np = np.array(B_img)
        B_np = self.normalize_xray_intensity(B_np)
        B_img = Image.fromarray(B_np.astype(np.uint8))

        A_np = self.pad_to_128_np(A_np)
        B_np = self.pad_to_128_np(B_np)

        A_img = Image.fromarray(A_np.astype(np.uint8))
        B_img = Image.fromarray(B_np.astype(np.uint8))

        if self.use_tray_mask and T_img is not None:
            T_np = np.array(T_img)
            T_np = self.pad_to_128_np(T_np)
            T_img = Image.fromarray(T_np.astype(np.uint8))

        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)
            if synth_preview_img is not None:
                synth_preview_img = self._to_gray_rgb(synth_preview_img)

        transform_params = get_params(self.opt, A_img.size)
        A_img_t = self._apply_shared_geom_to_mask_rgb(A_img, transform_params)

        T_img_t = None
        if self.use_tray_mask and T_img is not None:
            T_img_t = self._apply_shared_geom_to_mask_rgb(
                T_img.convert("RGB"), transform_params
            ).convert("L")

        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B_img)

        synth_preview_t = None
        if use_synth and synth_preview_img is not None:
            synth_preview_t = B_transform(synth_preview_img)

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

        # FINAL HARD FIX
        A_np = np.array(A_img_t)
        A_np = self.pad_to_128_np(A_np)
        A_img_t = Image.fromarray(A_np.astype(np.uint8))

        if self.use_tray_mask and T_img_t is not None:
            T_np = np.array(T_img_t)
            T_np = self.pad_to_128_np(T_np)
            T_img_t = Image.fromarray(T_np.astype(np.uint8))

        B_np = B.numpy().transpose(1, 2, 0) if torch.is_tensor(B) else np.array(B_img)
        B_np = self.pad_to_128_np(B_np)

        if torch.is_tensor(B):
            if B_np.ndim == 2:
                B = torch.from_numpy(B_np[None, ...]).float()
            else:
                B = torch.from_numpy(B_np.transpose(2, 0, 1)).float()

        if synth_preview_t is not None:
            synth_preview_np = synth_preview_t.detach().cpu().numpy().transpose(1, 2, 0)
            synth_preview_np = self.pad_to_128_np(synth_preview_np)
            if synth_preview_np.ndim == 2:
                synth_preview_t = torch.from_numpy(synth_preview_np[None, ...]).float()
            else:
                synth_preview_t = torch.from_numpy(synth_preview_np.transpose(2, 0, 1)).float()

        A, cond_chs = self._rgbmask_to_condition_tensor(A_img_t, app_img_t)

        out = {
            "A": A,
            "B": B,
            "A_paths": AB_path,
            "B_paths": AB_path,
            "is_synthetic": torch.tensor([1 if use_synth else 0], dtype=torch.uint8),
        }

        if self.use_tray_mask and T_img_t is not None:
            T_arr = (np.array(T_img_t).astype(np.float32) / 255.0)
            T_t = torch.from_numpy(T_arr[None, ...]).float()
            out["T"] = T_t

        if self.return_instance_masks:
            out["instance_masks"] = self._extract_instance_masks_tensor(A_img_t)

        if synth_preview_t is not None:
            out["synthetic_preview_B"] = synth_preview_t

        return out

    def __len__(self):
        return len(self.AB_paths)