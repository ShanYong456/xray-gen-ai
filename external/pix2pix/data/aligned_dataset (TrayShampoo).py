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
    Paired {A, B} dataset with optional E (empty tray) for delta compositing.
    Optional T (tray mask) to restrict generation inside tray.

    Conditioning layout (A tensor channels):
        ch0   = binary object mask
        ch1   = edge map                    (if use_edge_channel)
        ch2   = distance transform          (if use_thickness_channel)
        ch3,4 = local coord maps x, y      (if use_coord_channels)
        chN   = masked appearance from B   (if use_appearance_channel)

    With E concatenated later in model:
        total input_nc = cond_nc + 3
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # Tray mask
        parser.add_argument("--tray_mask_path", type=str, default="")
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

        # Instance masks
        parser.add_argument("--return_instance_masks", action="store_true")

        # Synthetic generation
        parser.add_argument("--synthetic_prob", type=float, default=0.0)
        parser.add_argument("--synthetic_mode", type=str, default="random_mask",
                            choices=["random_mask", "paste"])
        parser.add_argument("--synthetic_min_items", type=int, default=1)
        parser.add_argument("--synthetic_max_items", type=int, default=3)
        parser.add_argument("--synthetic_scale_min", type=float, default=0.6)
        parser.add_argument("--synthetic_scale_max", type=float, default=1.4)
        parser.add_argument("--synthetic_rot_min", type=float, default=0.0)
        parser.add_argument("--synthetic_rot_max", type=float, default=360.0)
        parser.add_argument("--synthetic_no_overlap", action="store_true")
        parser.add_argument("--synthetic_same_class_prob", type=float, default=0.0)
        parser.add_argument("--cutout_dir", type=str, default="")

        # Appearance
        parser.add_argument("--disable_test_appearance", action="store_true")
        parser.add_argument("--appearance_dropout", type=float, default=0.5)
        parser.add_argument("--appearance_zero_prob", type=float, default=0.35,
                            help="Prob of zero appearance (fully unguided).")
        parser.add_argument("--appearance_weak_prob", type=float, default=0.35,
                            help="Prob of blurred appearance (weakly guided).")
        parser.add_argument("--appearance_proto_prob", type=float, default=0.15,
                            help="Prob of prototype appearance (class-guided).")
        parser.add_argument("--appearance_blur_ksize", type=int, default=31)
        parser.add_argument("--appearance_blur_sigma", type=float, default=8.0)
        parser.add_argument("--build_appearance_prototypes", action="store_true")
        parser.add_argument("--max_appearance_prototypes", type=int, default=200)

        # Conditioning channels
        parser.add_argument("--use_edge_channel", action="store_true")
        parser.add_argument("--edge_dilate_px", type=int, default=1)
        parser.add_argument("--use_coord_channels", action="store_true")

        # Delta augmentation
        parser.add_argument("--delta_aug_scale_min", type=float, default=0.90)
        parser.add_argument("--delta_aug_scale_max", type=float, default=1.12)
        parser.add_argument("--delta_aug_noise_std", type=float, default=0.01)
        parser.add_argument("--delta_aug_edge_gain", type=float, default=0.10)
        parser.add_argument("--mask_aug_px", type=int, default=2)
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

        # Feature flags
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

        # Synthetic setup
        self.synthetic_prob = float(getattr(opt, "synthetic_prob", 0.0))
        self.synthetic_enabled = self.synthetic_prob > 0.0 and getattr(opt, "phase", "") == "train"
        self.cutout_items = []
        self.pseudo_object_lib = []
        self.appearance_prototypes = []

        # Tray mask
        self.use_tray_mask = bool(getattr(opt, "use_tray_mask", False))
        self.tray_mask_img = None
        if self.use_tray_mask:
            tray_path = getattr(opt, "tray_mask_path", "")
            if not tray_path:
                raise ValueError("--use_tray_mask requires --tray_mask_path.")
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
            print(f"[tray] mask {T_arr.shape} | area {T_bin.sum()} ({pct:.1f}%)")

        if self.synthetic_enabled:
            cutout_dir = str(getattr(opt, "cutout_dir", "")).strip()
            if not cutout_dir:
                raise ValueError("synthetic_prob > 0 requires --cutout_dir.")
            self.cutout_items = self._load_cutouts(Path(cutout_dir))
            if bool(getattr(opt, "use_delta_comp", False)):
                self.pseudo_object_lib = self._build_pseudo_object_library(max_items=500)
                print(f"[synthetic] built {len(self.pseudo_object_lib)} pseudo objects")

        if self.use_appearance_channel and bool(getattr(opt, "build_appearance_prototypes", False)):
            self.appearance_prototypes = self._build_appearance_prototype_bank(
                max_items=int(getattr(opt, "max_appearance_prototypes", 200))
            )
            print(f"[appearance] built {len(self.appearance_prototypes)} prototypes")

    # ─────────────────────────────────────────────────────────────────────────
    # Palette helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _train_id_to_rgb(self, train_id: int):
        return {1: np.array([0, 255, 0], dtype=np.uint8),
                2: np.array([0, 0, 255], dtype=np.uint8)}.get(int(train_id),
                                                               np.array([0, 0, 0], dtype=np.uint8))

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

    # ─────────────────────────────────────────────────────────────────────────
    # Appearance extraction — REALISM FIXES
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_appearance_from_B(self, B_img: Image.Image, A_img: Image.Image) -> Image.Image:
        """Masked grayscale from real B — the ground truth appearance cue."""
        B_gray = np.array(B_img.convert("L")).astype(np.float32) / 255.0
        obj = self._mask_from_Aimg(A_img).astype(np.float32)
        app = B_gray * obj
        return Image.fromarray(np.clip(app * 255, 0, 255).astype(np.uint8), mode="L")

    # Alias kept for synthetic path
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

    # ─────────────────────────────────────────────────────────────────────────
    # Conditioning tensor
    # ─────────────────────────────────────────────────────────────────────────

    def _rgbmask_to_condition_tensor(self, A_img: Image.Image, app_img: Image.Image = None):
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)
        chs = []

        # ch0: binary mask
        chs.append(obj.astype(np.float32))

        # ch1: edge map
        if self.use_edge_channel:
            chs.append(self._make_edge_map(obj))

        # ch2: distance transform (thickness proxy)
        if self.use_thickness_channel:
            if obj.sum() > 0:
                dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5).astype(np.float32)
                dist = dist / (dist.max() + 1e-6)
            else:
                dist = np.zeros_like(obj, dtype=np.float32)
            chs.append(dist)

        # ch3,4: local coord maps
        if self.use_coord_channels:
            chs.extend(self._make_coord_maps(obj))

        # chN: appearance
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
        return (edge.astype(np.float32) * obj)

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
        insts = [torch.from_numpy((labels == k).astype(np.float32))
                 for k in range(1, num)
                 if int(stats[k, cv2.CC_STAT_AREA]) >= 20]
        if not insts:
            return torch.zeros((0, obj.shape[0], obj.shape[1]), dtype=torch.float32)
        return torch.stack(insts, dim=0)

    def _build_condition_vis_from_channels(self, cond_chs):
        H, W = cond_chs[0].shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(min(3, len(cond_chs))):
            vis[..., c] = (np.clip(cond_chs[c], 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(vis, mode="RGB")

    # ─────────────────────────────────────────────────────────────────────────
    # Synthetic generation
    # ─────────────────────────────────────────────────────────────────────────

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
                bgr = img[:, :, :3].copy()
                if img.shape[2] == 4:
                    bgr[img[:, :, 3] == 0] = 0
                tid = self._infer_train_id_from_cutout_bgr(bgr)
                if tid == 0:
                    continue
                m = np.any(bgr > 0, axis=2)
                ys, xs = np.where(m)
                if not len(xs):
                    continue
                bgr = bgr[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
                items.append({"bgr": bgr, "train_id": tid})
        if not items:
            raise RuntimeError(f"No valid cutouts in {cutout_root}")
        print(f"[synthetic] loaded {len(items)} cutouts")
        return items

    def _rotate_preserve_bgr(self, img: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        M[0, 2] += nw / 2 - cx
        M[1, 2] += nh / 2 - cy
        return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _transform_cutout(self, bgr: np.ndarray):
        s = np.random.uniform(float(getattr(self.opt, "synthetic_scale_min", 0.6)),
                              float(getattr(self.opt, "synthetic_scale_max", 1.4)))
        out = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        ang = np.random.uniform(float(getattr(self.opt, "synthetic_rot_min", 0.0)),
                                float(getattr(self.opt, "synthetic_rot_max", 360.0)))
        out = self._rotate_preserve_bgr(out, ang)

        m = cv2.GaussianBlur((np.any(out > 0, axis=2).astype(np.uint8) * 255), (5, 5), 0.8)
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(m > 0)
        if not len(xs):
            return None

        pix = out[np.any(out > 0, axis=2)].reshape(-1, 3)
        dom = pix[np.unique(pix, axis=0, return_counts=True)[1].argmax()]
        clean = np.zeros_like(out)
        clean[m > 0] = dom
        return clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()

    def _build_synthetic_A_img(self, size_hw, T_img: Image.Image):
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)
        n_obj = np.random.randint(int(getattr(self.opt, "synthetic_min_items", 1)),
                                  int(getattr(self.opt, "synthetic_max_items", 3)) + 1)
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))
        same_class_prob = float(getattr(self.opt, "synthetic_same_class_prob", 0.0))
        chosen_tid = None
        if np.random.rand() < same_class_prob:
            tids = sorted({int(it["train_id"]) for it in self.cutout_items})
            if tids:
                chosen_tid = int(np.random.choice(tids))

        for _ in range(n_obj):
            cands = [it for it in self.cutout_items if chosen_tid is None or it["train_id"] == chosen_tid]
            item = cands[np.random.randint(len(cands))]
            cut = self._transform_cutout(item["bgr"])
            if cut is None:
                continue
            h, w = cut.shape[:2]
            if h >= H or w >= W or h < 2 or w < 2:
                continue
            obj_mask = np.any(cut > 0, axis=2)
            for _ in range(300):
                x, y = np.random.randint(0, W - w + 1), np.random.randint(0, H - h + 1)
                if not np.all(T[y:y + h, x:x + w][obj_mask]):
                    continue
                if no_overlap and np.any(occ[y:y + h, x:x + w] & obj_mask):
                    continue
                canvas[y:y + h, x:x + w][obj_mask] = cut[obj_mask]
                occ[y:y + h, x:x + w][obj_mask] = True
                break

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def _build_pseudo_object_library(self, max_items=500):
        lib = []
        for AB_path in self.AB_paths[:max_items]:
            try:
                AB = Image.open(AB_path).convert("RGB")
            except Exception:
                continue
            w, h = AB.size
            A_img = AB.crop((0, 0, w // 2, h))
            B_img = AB.crop((w // 2, 0, w, h))
            try:
                E_img, _ = self._load_empty_E(AB_path)
            except Exception:
                continue
            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BICUBIC)

            M = self._mask_from_Aimg(A_img).astype(np.uint8)
            ys, xs = np.where(M > 0)
            if not len(xs):
                continue
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            M_crop = M[y0:y1, x0:x1].copy()
            if M_crop.sum() < 50:
                continue

            B_np = np.array(B_img).astype(np.uint8)
            E_np = np.array(E_img).astype(np.uint8)
            delta = np.maximum(self._rgb_to_od(B_np) - self._rgb_to_od(E_np), 0.0)
            delta_crop = delta[y0:y1, x0:x1].copy()
            delta_crop[M_crop == 0] = 0.0

            A_crop = np.array(A_img)[y0:y1, x0:x1]
            train_id = self._infer_train_id_from_cutout_bgr(
                cv2.cvtColor(A_crop, cv2.COLOR_RGB2BGR)) or 1

            lib.append({"mask": M_crop, "delta": delta_crop.astype(np.float32),
                        "train_id": int(train_id)})

        lib = [x for x in lib if x["train_id"] > 0 and x["mask"].sum() >= 50]
        return lib

    def _transform_pseudo_object(self, obj):
        mask = self._random_mask_augment(obj["mask"].astype(np.uint8))
        delta = self._random_delta_augment(obj["delta"].astype(np.float32), mask)
        s = np.random.uniform(float(getattr(self.opt, "synthetic_scale_min", 0.6)),
                              float(getattr(self.opt, "synthetic_scale_max", 1.4)))
        angle = np.random.uniform(float(getattr(self.opt, "synthetic_rot_min", 0.0)),
                                  float(getattr(self.opt, "synthetic_rot_max", 360.0)))
        h, w = mask.shape[:2]
        nw, nh = max(2, int(w * s)), max(2, int(h * s))
        mask_r = cv2.resize((mask * 255).astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
        delta_r = cv2.resize(delta, (nw, nh), interpolation=cv2.INTER_LINEAR)

        M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        rw, rh = int(nh * sin + nw * cos), int(nh * cos + nw * sin)
        M[0, 2] += rw / 2 - nw / 2
        M[1, 2] += rh / 2 - nh / 2
        mask_t = cv2.warpAffine(mask_r, M, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
        delta_t = cv2.warpAffine(delta_r, M, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)
        mask_t = (mask_t > 127).astype(np.uint8)
        if mask_t.sum() < 20:
            return None
        ys, xs = np.where(mask_t > 0)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask_t = mask_t[y0:y1, x0:x1]
        delta_t = delta_t[y0:y1, x0:x1]
        delta_t[mask_t == 0] = 0.0
        return {"mask": mask_t, "delta": delta_t, "train_id": obj["train_id"]}

    def _build_synthetic_A_and_Bpseudo(self, size_hw, T_img: Image.Image, E_img: Image.Image):
        """
        Build synthetic conditioning mask A and physics-based pseudo target B.
        Realism fix: adds scanner noise + mild blur to pseudo-B to close the
        domain gap between training synthetic and real samples.
        """
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool)
        E_np = np.array(E_img).astype(np.uint8)
        OD_E = self._rgb_to_od(E_np)

        canvas_A = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)
        OD_total = OD_E.copy()

        n_obj = np.random.randint(int(getattr(self.opt, "synthetic_min_items", 1)),
                                  int(getattr(self.opt, "synthetic_max_items", 3)) + 1)
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))

        if not self.pseudo_object_lib:
            return self._build_synthetic_A_img(size_hw, T_img), E_img.copy()

        for _ in range(n_obj):
            obj = self.pseudo_object_lib[np.random.randint(len(self.pseudo_object_lib))]
            transformed = self._transform_pseudo_object(obj)
            if transformed is None:
                continue
            mask, delta, train_id = transformed["mask"], transformed["delta"], int(transformed["train_id"])
            if train_id <= 0:
                train_id = 1
            h, w = mask.shape[:2]
            if h < 2 or w < 2 or h >= H or w >= W:
                continue
            obj_region = mask > 0
            for _ in range(300):
                x, y = np.random.randint(0, W - w + 1), np.random.randint(0, H - h + 1)
                if not np.all(T[y:y + h, x:x + w][obj_region]):
                    continue
                if no_overlap and np.any(occ[y:y + h, x:x + w] & obj_region):
                    continue
                canvas_A[y:y + h, x:x + w][obj_region] = self._train_id_to_rgb(train_id)
                for c in range(3):
                    OD_total[y:y + h, x:x + w, c][obj_region] += delta[..., c][obj_region]
                occ[y:y + h, x:x + w][obj_region] = True
                break

        B_pseudo = self._od_to_rgb(OD_total)
        B_pseudo[~T] = E_np[~T]

        # ── Realism: match scanner statistics ─────────────────────────────
        # 1. Scanner noise (always apply, realistic σ range)
        noise_sigma = np.random.uniform(1.0, 3.5)
        noise = np.random.normal(0.0, noise_sigma, B_pseudo.shape).astype(np.float32)
        B_pseudo = np.clip(B_pseudo.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 2. Mild PSF blur (always apply)
        sigma = np.random.uniform(0.3, 0.8)
        B_pseudo = cv2.GaussianBlur(B_pseudo, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 3. Occasional slight sharpening (35% of samples)
        if np.random.rand() < 0.35:
            blur_for_sharp = cv2.GaussianBlur(B_pseudo, (0, 0), sigmaX=0.6, sigmaY=0.6)
            B_pseudo = np.clip(B_pseudo.astype(np.float32) + 0.15 * (
                B_pseudo.astype(np.float32) - blur_for_sharp.astype(np.float32)
            ), 0, 255).astype(np.uint8)
        # ──────────────────────────────────────────────────────────────────

        return Image.fromarray(canvas_A), Image.fromarray(B_pseudo)

    # ─────────────────────────────────────────────────────────────────────────
    # OD / image helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _rgb_to_od(self, img_rgb: np.ndarray, eps: float = 1e-6, gamma: float = 1.0) -> np.ndarray:
        x = np.clip(img_rgb.astype(np.float32) / 255.0, 0.0, 1.0)
        if gamma != 1.0:
            x = np.power(x, gamma)
        return -np.log(x + eps)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        return (np.clip(np.exp(-od), 0, 1) * 255).round().astype(np.uint8)

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

    def _shift_pil_rgb(self, img: Image.Image, dx: int, dy: int) -> Image.Image:
        return Image.fromarray(self._shift_np(np.array(img), dx, dy, fill=0))

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

    # ─────────────────────────────────────────────────────────────────────────
    # Tray mask helpers
    # ─────────────────────────────────────────────────────────────────────────

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

    def _load_tray_T(self, target_size):
        assert self.tray_mask_img is not None
        T_img = self.tray_mask_img
        if T_img.size != target_size:
            T_img = T_img.resize(target_size, resample=Image.NEAREST)
        return T_img

    # ─────────────────────────────────────────────────────────────────────────
    # Autoshift (tray fitting)
    # ─────────────────────────────────────────────────────────────────────────

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
            inv, distanceType=cv2.DIST_L2, maskSize=5, labelType=cv2.DIST_LABEL_PIXEL)
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

        return best_dx, best_dy, (best_out == 0)

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

    # ─────────────────────────────────────────────────────────────────────────
    # OD-based object move
    # ─────────────────────────────────────────────────────────────────────────

    def _move_object_in_B_with_shift(self, B_img, E_img, M_old, dx, dy, T=None):
        """Shift object in OD space (physics-correct for X-ray)."""
        B, E = np.array(B_img).astype(np.uint8), np.array(E_img).astype(np.uint8)
        M_obj = (M_old > 0).astype(np.uint8)
        dil = int(getattr(self.opt, "tray_obj_dilate_px", 0))
        if dil > 0:
            M_obj = self._dilate_bin(M_obj, dil)
        dx, dy = self._clamp_shift_to_image(M_obj, dx, dy)

        eps, gamma = 1e-6, 1.0
        OD_B = self._rgb_to_od(B, eps, gamma)
        OD_E = self._rgb_to_od(E, eps, gamma)
        delta_obj = np.maximum(OD_B - OD_E, 0.0)
        delta_obj[M_obj == 0] = 0.0

        delta_shift = self._shift_np(delta_obj, dx, dy, fill=0.0)
        M_shift = self._shift_np(M_obj, dx, dy, fill=0).astype(np.uint8)
        OD_new = OD_E.copy()
        inside = M_shift > 0
        for c in range(3):
            OD_new[..., c][inside] += delta_shift[..., c][inside]

        B_new = self._od_to_rgb(OD_new)
        if T is not None:
            B_new[T == 0] = E[T == 0]
        return Image.fromarray(B_new)

    # ─────────────────────────────────────────────────────────────────────────
    # Empty tray loader
    # ─────────────────────────────────────────────────────────────────────────

    def _load_empty_E(self, AB_path: str):
        import datetime

        def parse_ts(name: str):
            try:
                return datetime.datetime.strptime(name.split(".")[0], "%Y-%m-%d_%H-%M-%S-%f").timestamp()
            except Exception:
                return None

        empty_dir = getattr(self.opt, "empty_dir", "")
        if empty_dir:
            bname = Path(AB_path).name
            e_path = Path(empty_dir) / bname
            if not e_path.exists():
                timestamp = bname.split("-", 1)[1].split("_tr")[0] if "-" in bname else None
                if timestamp:
                    cands = list(Path(empty_dir).glob(f"*{timestamp}*"))
                    if len(cands) >= 1:
                        e_path = cands[0]
                    else:
                        target_sec = parse_ts(timestamp)
                        best, best_diff = None, None
                        for emp in Path(empty_dir).iterdir():
                            emp_sec = parse_ts(emp.stem)
                            if emp_sec is None or target_sec is None:
                                continue
                            diff = abs(emp_sec - target_sec)
                            if best_diff is None or diff < best_diff:
                                best_diff, best = diff, emp
                        if best is not None:
                            e_path = best
                        else:
                            raise FileNotFoundError(f"Empty not found for {bname}")
            img = Image.open(str(e_path)).convert("RGB")
            if self.force_gray_rgb:
                img = self._to_gray_rgb(img)
            return img, True

        empty_path = getattr(self.opt, "empty_path", "")
        if empty_path:
            p = Path(empty_path)
            if not p.exists():
                raise FileNotFoundError(f"Empty not found: {p}")
            img = Image.open(str(p)).convert("RGB")
            if self.force_gray_rgb:
                img = self._to_gray_rgb(img)
            return img, False

        raise ValueError("use_delta_comp requires --empty_dir or --empty_path.")

    def _match_empty_to_B(self, E_img, B_img, obj_mask) -> Image.Image:
        """Global affine colour match of E to B background — improves E/B consistency."""
        E = np.array(E_img).astype(np.float32)
        B = np.array(B_img).astype(np.float32)
        if E.shape[:2] != B.shape[:2]:
            return E_img
        bg = ~obj_mask
        if bg.sum() < 2000:
            return E_img
        out = E.copy()
        for c in range(3):
            e, b = E[..., c][bg], B[..., c][bg]
            e_mean, e_std = float(e.mean()), float(e.std() + 1e-6)
            b_mean, b_std = float(b.mean()), float(b.std() + 1e-6)
            a = b_std / e_std
            out[..., c] = np.clip(a * E[..., c] + (b_mean - a * e_mean), 0, 255)
        return Image.fromarray(out.astype(np.uint8))

    # ─────────────────────────────────────────────────────────────────────────
    # Delta augmentation
    # ─────────────────────────────────────────────────────────────────────────

    def _random_delta_augment(self, delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = delta.astype(np.float32).copy()
        m = mask > 0
        if not np.any(m):
            return out
        scale = np.random.uniform(float(getattr(self.opt, "delta_aug_scale_min", 0.90)),
                                  float(getattr(self.opt, "delta_aug_scale_max", 1.12)))
        out[m] *= scale

        h, w = mask.shape[:2]
        noise_small = np.random.normal(0, 0.08,
                                       (max(8, h // 8), max(8, w // 8))).astype(np.float32)
        field = 1.0 + 0.10 * np.clip(
            cv2.GaussianBlur(cv2.resize(noise_small, (w, h), cv2.INTER_CUBIC), (0, 0), 5.0),
            -0.12, 0.15)
        for c in range(3):
            out[..., c][m] *= field[m]

        mask_u8 = (m.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)
        if dist.max() > 1e-6:
            dist /= dist.max() + 1e-6
            edge_gain = np.random.uniform(-0.05, float(getattr(self.opt, "delta_aug_edge_gain", 0.10)))
            ef = np.clip(1.0 + edge_gain * (1.0 - dist), 0.90, 1.15)
            for c in range(3):
                out[..., c][m] *= ef[m]

        out[m] += np.random.normal(0, float(getattr(self.opt, "delta_aug_noise_std", 0.01)),
                                   size=out.shape)[m]

        r = np.random.rand()
        if r < 0.33:
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=np.random.uniform(0.4, 1.0))
        elif r < 0.66:
            blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8)
            out = out + 0.20 * (out - blurred)

        out[~m] = 0.0
        return np.clip(out, 0.0, 4.0).astype(np.float32)

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
            _, m = cv2.threshold(cv2.GaussianBlur((m * 255).astype(np.uint8), (5, 5), 0.8),
                                 127, 1, cv2.THRESH_BINARY)
        return m.astype(np.uint8)

    # ─────────────────────────────────────────────────────────────────────────
    # Geometry transform (NEAREST for mask to preserve palette colours)
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_shared_geom_to_mask_rgb(self, img: Image.Image, params) -> Image.Image:
        ow, oh = img.size
        pre = self.opt.preprocess

        if "resize" in pre or pre in ["resize"]:
            img = img.resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)
        elif "scale_width" in pre:
            new_w = self.opt.load_size
            img = img.resize((new_w, int(round(new_w * oh / ow))), resample=Image.NEAREST)
        elif "scale_shortside" in pre:
            scale = self.opt.load_size / min(ow, oh)
            img = img.resize((int(round(ow * scale)), int(round(oh * scale))), resample=Image.NEAREST)

        if "crop" in pre:
            x, y = params["crop_pos"]
            img = img.crop((x, y, x + self.opt.crop_size, y + self.opt.crop_size))

        if not self.opt.no_flip and params["flip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    # ─────────────────────────────────────────────────────────────────────────
    # __getitem__
    # ─────────────────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")
        w, h = AB.size
        A_img = AB.crop((0, 0, w // 2, h))
        B_img = AB.crop((w // 2, 0, w, h))

        use_synth = self.synthetic_enabled and (np.random.rand() < self.synthetic_prob)
        use_delta = bool(getattr(self.opt, "use_delta_comp", False))
        is_train = getattr(self.opt, "phase", "") == "train"

        # ── Load empty tray E ──────────────────────────────────────────────
        E_img, loaded_from_dir = None, False
        if use_delta:
            try:
                E_img, loaded_from_dir = self._load_empty_E(AB_path)
            except FileNotFoundError as exc:
                ep = getattr(self.opt, "empty_path", "")
                if ep and Path(ep).exists():
                    E_img = Image.open(ep).convert("RGB")
                    print(f"[warning] {exc}; using --empty_path fallback")
                else:
                    E_img = B_img.copy()
                    print(f"[warning] {exc}; using B as E placeholder")
            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BICUBIC)
            if self.force_gray_rgb:
                E_img = self._to_gray_rgb(E_img)

        # ── Tray mask ──────────────────────────────────────────────────────
        T_img, T_bin = None, None
        if self.use_tray_mask:
            T_img = self._load_tray_T(A_img.size)
            T_bin = self._get_tray_bin(T_img)

        # ── Synthetic or real sample ───────────────────────────────────────
        if use_synth:
            if not self.use_tray_mask:
                raise RuntimeError("Synthetic mode requires --use_tray_mask.")
            A_img, B_img = self._build_synthetic_A_and_Bpseudo(
                (A_img.size[1], A_img.size[0]), T_img, E_img)

        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)

        # ── Match E to B background (real samples only, global empty) ─────
        if use_delta and self.match_empty_to_B and (not loaded_from_dir) and \
                bool(getattr(self.opt, "empty_path", "")) and (not use_synth):
            obj_mask = self._mask_from_Aimg(A_img)
            E_img = self._match_empty_to_B(E_img, B_img, obj_mask)

        # ── Shared geometric transform ─────────────────────────────────────
        transform_params = get_params(self.opt, A_img.size)
        A_img_t = self._apply_shared_geom_to_mask_rgb(A_img, transform_params)

        T_img_t = None
        if self.use_tray_mask and T_img is not None:
            T_img_t = self._apply_shared_geom_to_mask_rgb(
                T_img.convert("RGB"), transform_params).convert("L")

        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B_img)
        E = B_transform(E_img) if E_img is not None else None

        # ── Appearance channel ─────────────────────────────────────────────
        app_img_t = None
        if self.use_appearance_channel:
            app_raw = self._extract_appearance_from_B(B_img, A_img)
            app_img_t = self._apply_shared_geom_to_mask_rgb(
                app_raw.convert("RGB"), transform_params).convert("L")

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
                # else: keep real appearance (remaining ~15% by default)

            if not is_train and bool(getattr(self.opt, "disable_test_appearance", False)):
                app_img_t = self._zero_appearance_img(A_img_t)

        A, cond_chs = self._rgbmask_to_condition_tensor(A_img_t, app_img=app_img_t)
        A_vis = B_transform(self._build_condition_vis_from_channels(cond_chs))

        # ── Instance masks ─────────────────────────────────────────────────
        instance_masks = None
        if self.return_instance_masks:
            instance_masks = self._extract_instance_masks_tensor(A_img_t)

        # ── Tray tensor ────────────────────────────────────────────────────
        T = None
        if self.use_tray_mask and T_img_t is not None:
            T_np = self._get_tray_bin(T_img_t).astype(np.float32)
            T = torch.from_numpy(T_np[None]).float()

        # ── Debug logging ──────────────────────────────────────────────────
        if use_delta and is_train and (index % self.debug_every == 0):
            print(f"[debug] {Path(AB_path).name} synth={use_synth} "
                  f"A({A.min():.3f},{A.max():.3f}) B({B.min():.3f},{B.max():.3f})"
                  + (f" E({E.min():.3f},{E.max():.3f})" if E is not None else ""))

        out = {"A": A, "A_vis": A_vis, "B": B,
               "A_paths": AB_path, "B_paths": AB_path, "is_synthetic": use_synth}
        if use_delta:
            out["E"] = E
        if T is not None:
            out["T"] = T
        if instance_masks is not None:
            out["instance_masks"] = instance_masks
        return out

    def __len__(self):
        return len(self.AB_paths)