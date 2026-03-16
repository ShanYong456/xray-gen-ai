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

    CURRENT CONDITIONING LAYOUT:
      A is returned as conditioning tensor:
          ch0 = binary object mask
          ch1 = masked object grayscale appearance from B
          ch2 = thickness / distance transform (if enabled)

      With E concatenated later in model:
          total input_nc = cond_nc + 3

      Example:
          cond_nc = 3  -> [mask, appearance, thickness]
          total input_nc = 6 after concatenating E
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

        parser.add_argument(
            "--return_instance_masks",
            action="store_true",
            help="Return connected-component instance masks for instance-wise supervision.",
        )

        parser.add_argument(
            "--synthetic_prob",
            type=float,
            default=0.0,
            help="Probability of returning a synthetic sample during training.",
        )
        parser.add_argument(
            "--synthetic_mode",
            type=str,
            default="random_mask",
            choices=["random_mask", "paste"],
            help="How to generate synthetic A samples.",
        )
        parser.add_argument(
            "--synthetic_min_items",
            type=int,
            default=1,
            help="Minimum number of objects in synthetic generation.",
        )
        parser.add_argument(
            "--synthetic_max_items",
            type=int,
            default=3,
            help="Maximum number of objects in synthetic generation.",
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
            help="Disallow overlap in synthetic generation.",
        )
        parser.add_argument(
            "--synthetic_same_class_prob",
            type=float,
            default=0.0,
            help="Probability that all synthetic objects in a sample come from the same class.",
        )
        parser.add_argument(
            "--cutout_dir",
            type=str,
            default="",
            help="Folder containing colored object cutouts for synthetic generation.",
        )

        parser.add_argument(
            "--disable_test_appearance",
            action="store_true",
            help="During test, do not extract appearance from B; appearance channel becomes zero."
        )
        parser.add_argument(
            "--appearance_dropout",
            type=float,
            default=0.5,
            help="Probability of removing appearance channel during training."
        )
        parser.add_argument(
            "--use_edge_channel",
            action="store_true",
            help="Add 1-channel object edge map to conditioning.",
        )
        parser.add_argument(
            "--edge_dilate_px",
            type=int,
            default=1,
            help="Edge thickness control in pixels.",
        )
        parser.add_argument(
            "--use_coord_channels",
            action="store_true",
            help="Add 2-channel normalized object-local coord maps (x,y).",
        )
        parser.add_argument(
            "--appearance_zero_prob",
            type=float,
            default=0.35,
            help="Probability of replacing appearance with all-zero map during training.",
        )
        parser.add_argument(
            "--appearance_weak_prob",
            type=float,
            default=0.35,
            help="Probability of replacing appearance with weak blurred appearance during training.",
        )
        parser.add_argument(
            "--appearance_proto_prob",
            type=float,
            default=0.15,
            help="Probability of replacing appearance with class prototype appearance during training.",
        )
        parser.add_argument(
            "--appearance_blur_ksize",
            type=int,
            default=31,
            help="Gaussian blur kernel size for weak appearance mode.",
        )
        parser.add_argument(
            "--appearance_blur_sigma",
            type=float,
            default=8.0,
            help="Gaussian blur sigma for weak appearance mode.",
        )
        parser.add_argument(
            "--build_appearance_prototypes",
            action="store_true",
            help="Build appearance prototype bank for unguided robustness.",
        )
        parser.add_argument(
            "--max_appearance_prototypes",
            type=int,
            default=200,
            help="Maximum number of prototype appearance samples to build.",
        )
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

        self.shift_reduce_count = 0
        self.shift_reduce_by_iters = []

        self.class_nc = int(getattr(self.opt, "class_nc", 2))
        self.thickness_nc = int(getattr(self.opt, "thickness_nc", 1))
        self.use_thickness_channel = bool(getattr(self.opt, "use_thickness_channel", False))
        self.use_appearance_channel = bool(getattr(self.opt, "use_appearance_channel", False))
        self.appearance_nc = int(getattr(self.opt, "appearance_nc", 1))
        self.return_instance_masks = bool(getattr(self.opt, "return_instance_masks", False))
        self.use_edge_channel = bool(getattr(self.opt, "use_edge_channel", False))
        self.use_coord_channels = bool(getattr(self.opt, "use_coord_channels", False))

        if self.use_appearance_channel and self.appearance_nc != 1:
            raise ValueError("Current implementation supports --appearance_nc 1 only.")

        self.synthetic_prob = float(getattr(self.opt, "synthetic_prob", 0.0))
        self.synthetic_mode = str(getattr(self.opt, "synthetic_mode", "random_mask"))
        self.synthetic_enabled = self.synthetic_prob > 0.0 and getattr(self.opt, "phase", "") == "train"

        self.cutout_items = []
        self.pseudo_object_lib = []
        self.appearance_prototypes = []

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

        if self.synthetic_enabled:
            cutout_dir = str(getattr(self.opt, "cutout_dir", "")).strip()
            if not cutout_dir:
                raise ValueError("synthetic_prob > 0 but --cutout_dir is empty.")
            self.cutout_items = self._load_cutouts(Path(cutout_dir))

        if self.synthetic_enabled and bool(getattr(self.opt, "use_delta_comp", False)):
            self.pseudo_object_lib = self._build_pseudo_object_library(max_items=500)
            print(f"[synthetic-pseudo] built {len(self.pseudo_object_lib)} pseudo object entries")

        if self.use_appearance_channel and bool(getattr(self.opt, "build_appearance_prototypes", False)):
            self.appearance_prototypes = self._build_appearance_prototype_bank(
                max_items=int(getattr(self.opt, "max_appearance_prototypes", 200))
            )
            print(f"[appearance-proto] built {len(self.appearance_prototypes)} prototypes")

    # ------------------------  -
    # Palette helpers
    # -------------------------
    def _train_id_to_rgb(self, train_id: int):
        if int(train_id) == 1:
            return np.array([0, 255, 0], dtype=np.uint8)   # Shampoo
        if int(train_id) == 2:
            return np.array([0, 0, 255], dtype=np.uint8)   # Blade
        return np.array([0, 0, 0], dtype=np.uint8)

    def _rgb_to_train_masks(self, A_rgb: np.ndarray):
        shampoo_rgb = np.array([0, 255, 0], dtype=np.uint8)
        blade_rgb = np.array([0, 0, 255], dtype=np.uint8)

        shampoo = np.all(A_rgb == shampoo_rgb[None, None, :], axis=2).astype(np.uint8)
        blade = np.all(A_rgb == blade_rgb[None, None, :], axis=2).astype(np.uint8)
        return shampoo, blade

    def _build_condition_preview_rgb(self, B_img: Image.Image, A_img: Image.Image) -> Image.Image:
        """
        Build a 3-channel grayscale preview that preserves the real object tone
        from B inside the object region.
        """
        B_gray = np.array(B_img.convert("L")).astype(np.float32) / 255.0
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)

        vis = np.zeros_like(B_gray, dtype=np.float32)
        vis[obj > 0] = B_gray[obj > 0]

        vis_u8 = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
        vis_rgb = np.stack([vis_u8, vis_u8, vis_u8], axis=2)
        return Image.fromarray(vis_rgb, mode="RGB")

    def _extract_appearance_from_B(self, B_img: Image.Image, A_img: Image.Image) -> Image.Image:
        """
        Build a 1-channel appearance map from the real/pseudo target image B.
        Only keep values inside the object mask so the model gets item interior cues
        without leaking the full tray background.
        """
        B_gray = np.array(B_img.convert("L")).astype(np.float32) / 255.0
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.float32)

        app = B_gray * obj
        return Image.fromarray(np.clip(app * 255.0, 0, 255).astype(np.uint8), mode="L")

    def _extract_object_grayscale_from_B(self, B_img: Image.Image, A_img: Image.Image) -> Image.Image:
        B_gray = np.array(B_img.convert("L")).astype(np.float32) / 255.0
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.float32)

        obj_gray = B_gray * obj
        return Image.fromarray(np.clip(obj_gray * 255.0, 0, 255).astype(np.uint8), mode="L")

    def _rgbmask_to_condition_tensor(self, A_img: Image.Image, app_img: Image.Image = None):
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)

        chs = []

        # 1) binary object mask
        obj_mask = obj.astype(np.float32)
        chs.append(obj_mask)

        # 2) edge map
        if self.use_edge_channel:
            edge = self._make_edge_map(obj)
            chs.append(edge)

        # 3) thickness / distance transform
        if self.use_thickness_channel:
            if obj.sum() > 0:
                dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5).astype(np.float32)
                if dist.max() > 0:
                    dist = dist / (dist.max() + 1e-6)
            else:
                dist = np.zeros_like(obj_mask, dtype=np.float32)
            chs.append(dist)

        # 4) coord channels
        if self.use_coord_channels:
            coord_x, coord_y = self._make_coord_maps(obj)
            chs.append(coord_x)
            chs.append(coord_y)
        
        # 5) appearance channel
        if self.use_appearance_channel:
            if app_img is not None:
                app = np.array(app_img).astype(np.float32) / 255.0
            else:
                app = np.zeros_like(obj_mask, dtype=np.float32)
            chs.append(app)

        cond = np.stack(chs, axis=0)
        cond = cond * 2.0 - 1.0
        return torch.from_numpy(cond).float(), chs

    def _extract_instance_masks_tensor(self, A_img: Image.Image):
        A = np.array(A_img).astype(np.uint8)
        shampoo, blade = self._rgb_to_train_masks(A)
        obj = ((shampoo > 0) | (blade > 0)).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(obj, connectivity=8)
        insts = []
        for k in range(1, num):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < 20:
                continue
            mk = (labels == k).astype(np.float32)
            insts.append(torch.from_numpy(mk))

        if len(insts) == 0:
            return torch.zeros((0, obj.shape[0], obj.shape[1]), dtype=torch.float32)

        return torch.stack(insts, dim=0)

    # -------------------------
    # Synthetic helpers
    # -------------------------

    def _weak_blur_appearance_img(self, app_img: Image.Image, A_img: Image.Image) -> Image.Image:
        """
        Weak appearance cue:
        keep only low-frequency interior tone, remove fine details.
        """
        app = np.array(app_img).astype(np.uint8)
        A = np.array(A_img).astype(np.uint8)

        if app.ndim == 3:
            app = cv2.cvtColor(app, cv2.COLOR_RGB2GRAY)

        obj = self._mask_from_Aimg(A).astype(np.uint8)
        if obj.sum() == 0:
            return self._zero_appearance_img(A_img)

        ksize = int(getattr(self.opt, "appearance_blur_ksize", 31))
        sigma = float(getattr(self.opt, "appearance_blur_sigma", 8.0))

        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)

        blur = cv2.GaussianBlur(app, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        blur = (blur.astype(np.float32) * obj.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blur, mode="L")


    def _normalize_object_crop_to_canvas(self, obj_gray: np.ndarray, obj_mask: np.ndarray, out_hw):
        """
        Resize an object's grayscale interior crop back into full canvas bbox.
        """
        H, W = out_hw
        canvas = np.zeros((H, W), dtype=np.uint8)

        ys, xs = np.where(obj_mask > 0)
        if len(xs) == 0:
            return canvas

        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        bh = max(1, y1 - y0 + 1)
        bw = max(1, x1 - x0 + 1)

        crop_resized = cv2.resize(obj_gray, (bw, bh), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(obj_mask.astype(np.uint8) * 255, (bw, bh), interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 127).astype(np.uint8)

        patch = np.zeros((bh, bw), dtype=np.uint8)
        patch[mask_resized > 0] = crop_resized[mask_resized > 0]

        canvas[y0:y0+bh, x0:x0+bw] = patch
        return canvas


    def _build_appearance_prototype_bank(self, max_items=200):
        """
        Build class-level prototype interiors from training set.
        Each prototype stores cropped grayscale interior and cropped mask.
        """
        bank = []

        for AB_path in self.AB_paths[:max_items]:
            try:
                AB = Image.open(AB_path).convert("RGB")
            except Exception:
                continue

            w, h = AB.size
            w2 = w // 2
            A_img = AB.crop((0, 0, w2, h))
            B_img = AB.crop((w2, 0, w, h))

            obj_mask = self._mask_from_Aimg(A_img).astype(np.uint8)
            ys, xs = np.where(obj_mask > 0)
            if len(xs) == 0:
                continue

            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1

            B_gray = np.array(B_img.convert("L")).astype(np.uint8)
            obj_crop = B_gray[y0:y1, x0:x1].copy()
            mask_crop = obj_mask[y0:y1, x0:x1].copy()

            if mask_crop.sum() < 20:
                continue

            obj_crop[mask_crop == 0] = 0
            bank.append({
                "gray": obj_crop,
                "mask": mask_crop,
                "path": str(AB_path),
            })

        return bank


    def _sample_prototype_appearance_img(self, A_img: Image.Image) -> Image.Image:
        """
        Take a random prototype interior and warp it into the current object's bbox.
        Gives class-style interior cue without leaking the exact target.
        """
        if len(self.appearance_prototypes) == 0:
            return self._zero_appearance_img(A_img)

        proto = self.appearance_prototypes[np.random.randint(len(self.appearance_prototypes))]
        obj_mask = self._mask_from_Aimg(A_img).astype(np.uint8)

        canvas = self._normalize_object_crop_to_canvas(
            proto["gray"],
            obj_mask,
            out_hw=obj_mask.shape[:2]
        )

        canvas = (canvas.astype(np.float32) * obj_mask.astype(np.float32)).clip(0, 255).astype(np.uint8)
        return Image.fromarray(canvas, mode="L")


    def _apply_shared_geom_to_mask_rgb(self, img: Image.Image, params):
        """
        Apply the same resize/crop/flip geometry as B, but preserve label colors.
        Uses NEAREST interpolation so exact palette values remain unchanged.
        """
        ow, oh = img.size

        if self.opt.preprocess in ["resize", "resize_and_crop"]:
            img = img.resize((self.opt.load_size, self.opt.load_size), resample=Image.NEAREST)

        elif self.opt.preprocess in ["scale_width", "scale_width_and_crop"]:
            if ow != self.opt.load_size:
                new_w = self.opt.load_size
                new_h = int(round(self.opt.load_size * oh / ow))
                img = img.resize((new_w, new_h), resample=Image.NEAREST)

        elif self.opt.preprocess in ["scale_shortside", "scale_shortside_and_crop"]:
            short = min(ow, oh)
            if short != self.opt.load_size:
                scale = float(self.opt.load_size) / float(short)
                new_w = int(round(ow * scale))
                new_h = int(round(oh * scale))
                img = img.resize((new_w, new_h), resample=Image.NEAREST)

        if "crop" in self.opt.preprocess:
            x, y = params["crop_pos"]
            tw = self.opt.crop_size
            th = self.opt.crop_size
            img = img.crop((x, y, x + tw, y + th))

        if not self.opt.no_flip and params["flip"]:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        return img



    def _make_edge_map(self, obj: np.ndarray) -> np.ndarray:
        """
        1-channel binary-ish edge map from object mask.
        """
        obj = obj.astype(np.uint8)
        if obj.sum() == 0:
            return np.zeros_like(obj, dtype=np.float32)

        px = int(getattr(self.opt, "edge_dilate_px", 1))
        k = max(1, 2 * px + 1)
        kernel = np.ones((k, k), np.uint8)

        dil = cv2.dilate(obj, kernel, iterations=1)
        ero = cv2.erode(obj, kernel, iterations=1)
        edge = (dil - ero) > 0
        edge = edge.astype(np.float32) * obj.astype(np.float32)
        return edge

    def _make_coord_maps(self, obj: np.ndarray):
        """
        2 channels:
          coord_x in [0,1] inside object bbox
          coord_y in [0,1] inside object bbox
        Outside object = 0
        """
        obj = obj.astype(np.uint8)
        H, W = obj.shape[:2]

        coord_x = np.zeros((H, W), dtype=np.float32)
        coord_y = np.zeros((H, W), dtype=np.float32)

        ys, xs = np.where(obj > 0)
        if len(xs) == 0:
            return coord_x, coord_y

        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        bw = max(1, x1 - x0)
        bh = max(1, y1 - y0)

        xx = np.arange(W, dtype=np.float32)
        yy = np.arange(H, dtype=np.float32)

        x_norm = (xx - float(x0)) / float(bw)
        y_norm = (yy - float(y0)) / float(bh)

        x_norm = np.clip(x_norm, 0.0, 1.0)
        y_norm = np.clip(y_norm, 0.0, 1.0)

        coord_x_full = np.tile(x_norm[None, :], (H, 1))
        coord_y_full = np.tile(y_norm[:, None], (1, W))

        objf = obj.astype(np.float32)
        coord_x = coord_x_full * objf
        coord_y = coord_y_full * objf

        return coord_x, coord_y

    def _build_condition_vis_from_channels(self, cond_chs):
        """
        Build preview RGB from first 3 condition channels if available.
        """
        if len(cond_chs) == 0:
            raise ValueError("cond_chs is empty")

        H, W = cond_chs[0].shape[:2]
        vis = np.zeros((H, W, 3), dtype=np.uint8)

        for c in range(min(3, len(cond_chs))):
            ch = np.clip(cond_chs[c], 0.0, 1.0)
            vis[..., c] = (ch * 255.0).astype(np.uint8)

        return Image.fromarray(vis, mode="RGB")
    def _zero_appearance_img(self, A_img: Image.Image) -> Image.Image:
        """
        Honest unguided appearance:
        return an all-zero 1-channel image.
        This avoids teaching the network that a blurry fake blob = valid appearance cue.
        """
        A = np.array(A_img)
        H, W = A.shape[:2]
        return Image.fromarray(np.zeros((H, W), dtype=np.uint8), mode="L")
    
    def _infer_train_id_from_cutout_bgr(self, cut_bgr: np.ndarray) -> int:
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

    def _sample_cutout_item(self, chosen_train_id=None):
        if chosen_train_id is None:
            return self.cutout_items[np.random.randint(len(self.cutout_items))]
        cands = [it for it in self.cutout_items if int(it["train_id"]) == int(chosen_train_id)]
        if len(cands) == 0:
            return self.cutout_items[np.random.randint(len(self.cutout_items))]
        return cands[np.random.randint(len(cands))]

    def _sample_pseudo_object(self, chosen_train_id=None):
        if chosen_train_id is None:
            return self.pseudo_object_lib[np.random.randint(len(self.pseudo_object_lib))]
        cands = [it for it in self.pseudo_object_lib if int(it["train_id"]) == int(chosen_train_id)]
        if len(cands) == 0:
            return self.pseudo_object_lib[np.random.randint(len(self.pseudo_object_lib))]
        return cands[np.random.randint(len(cands))]

    def _build_synthetic_A_img(self, size_hw, T_img: Image.Image):
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool)

        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)

        n_min = int(getattr(self.opt, "synthetic_min_items", 1))
        n_max = int(getattr(self.opt, "synthetic_max_items", 3))
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))
        n_obj = np.random.randint(n_min, n_max + 1)

        same_class_prob = float(getattr(self.opt, "synthetic_same_class_prob", 0.0))
        chosen_train_id = None
        if np.random.rand() < same_class_prob:
            tids = sorted(list({int(it["train_id"]) for it in self.cutout_items}))
            if len(tids) > 0:
                chosen_train_id = int(np.random.choice(tids))

        tries_per_obj = 300

        for _ in range(n_obj):

            item = self._sample_cutout_item(chosen_train_id=chosen_train_id)
            if item is None:
                continue

            cut = self._transform_cutout(item["bgr"])
            if cut is None:
                continue

            h, w = cut.shape[:2]

            if h < 2 or w < 2:
                continue

            if h >= H or w >= W:
                continue

            obj_mask = np.any(cut > 0, axis=2)
            if obj_mask.sum() < 10:
                continue

            placed = False

            for _ in range(tries_per_obj):

                x = np.random.randint(0, W - w + 1)
                y = np.random.randint(0, H - h + 1)

                tray_region = T[y:y + h, x:x + w]

                # ensure object pixels lie inside tray
                if not np.all(tray_region[obj_mask]):
                    continue

                if no_overlap:
                    if np.any(occ[y:y + h, x:x + w] & obj_mask):
                        continue

                region = canvas[y:y + h, x:x + w]

                region[obj_mask] = cut[obj_mask]

                canvas[y:y + h, x:x + w] = region

                occ_region = occ[y:y + h, x:x + w]
                occ_region[obj_mask] = True
                occ[y:y + h, x:x + w] = occ_region

                placed = True
                break

            if not placed:
                continue

        return Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    def _build_pseudo_object_library(self, max_items=500):
        lib = []
        eps = 1e-6
        gamma = 1.0

        for AB_path in self.AB_paths[:max_items]:
            try:
                AB = Image.open(AB_path).convert("RGB")
            except Exception:
                continue

            w, h = AB.size
            w2 = w // 2
            A_img = AB.crop((0, 0, w2, h))
            B_img = AB.crop((w2, 0, w, h))

            try:
                E_img, _ = self._load_empty_E(AB_path)
            except Exception:
                continue

            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BICUBIC)

            A_np = np.array(A_img)
            B_np = np.array(B_img)
            E_np = np.array(E_img)

            M = self._mask_from_Aimg(A_img).astype(np.uint8)
            ys, xs = np.where(M > 0)
            if len(xs) == 0:
                continue

            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
            M_crop = M[y1:y2, x1:x2].copy()
            if M_crop.sum() < 50:
                continue

            OD_B = self._rgb_to_od(B_np, eps=eps, gamma=gamma)
            OD_E = self._rgb_to_od(E_np, eps=eps, gamma=gamma)
            delta = np.maximum(OD_B - OD_E, 0.0)

            delta_crop = delta[y1:y2, x1:x2, :].copy()
            delta_crop[M_crop == 0] = 0.0

            A_crop = A_np[y1:y2, x1:x2]
            train_id = self._infer_train_id_from_cutout_bgr(cv2.cvtColor(A_crop, cv2.COLOR_RGB2BGR))

            lib.append(
                {
                    "mask": M_crop,
                    "delta": delta_crop,
                    "train_id": train_id,
                    "path": str(AB_path),
                }
            )

        return lib

    def _transform_pseudo_object(self, obj):
        mask = obj["mask"].astype(np.uint8)
        delta = obj["delta"].astype(np.float32)

        # augment mask + delta BEFORE geometry
        mask = self._random_mask_augment(mask)
        delta = self._random_delta_augment(delta, mask)

        s = np.random.uniform(
            float(getattr(self.opt, "synthetic_scale_min", 0.6)),
            float(getattr(self.opt, "synthetic_scale_max", 1.4)),
        )
        angle = np.random.uniform(
            float(getattr(self.opt, "synthetic_rot_min", 0.0)),
            float(getattr(self.opt, "synthetic_rot_max", 360.0)),
        )

        h, w = mask.shape[:2]
        new_w = max(2, int(round(w * s)))
        new_h = max(2, int(round(h * s)))

        mask_r = cv2.resize((mask * 255).astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        delta_r = cv2.resize(delta, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        rot_w = int(new_h * sin + new_w * cos)
        rot_h = int(new_h * cos + new_w * sin)
        M[0, 2] += (rot_w / 2) - new_w / 2
        M[1, 2] += (rot_h / 2) - new_h / 2

        mask_t = cv2.warpAffine(mask_r, M, (rot_w, rot_h), flags=cv2.INTER_NEAREST, borderValue=0)
        delta_t = cv2.warpAffine(delta_r, M, (rot_w, rot_h), flags=cv2.INTER_LINEAR, borderValue=0)

        mask_t = (mask_t > 127).astype(np.uint8)
        if mask_t.sum() < 20:
            return None

        ys, xs = np.where(mask_t > 0)
        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        mask_t = mask_t[y1:y2, x1:x2]
        delta_t = delta_t[y1:y2, x1:x2, :]
        delta_t[mask_t == 0] = 0.0

        return {
            "mask": mask_t,
            "delta": delta_t,
            "train_id": obj["train_id"],
        }


    def _build_synthetic_A_and_Bpseudo(self, size_hw, T_img: Image.Image, E_img: Image.Image):
        H, W = size_hw
        T = self._get_tray_bin(T_img).astype(bool)

        E_np = np.array(E_img).astype(np.uint8)
        OD_E = self._rgb_to_od(E_np)

        canvas_A = np.zeros((H, W, 3), dtype=np.uint8)
        occ = np.zeros((H, W), dtype=bool)
        OD_total = OD_E.copy()

        n_min = int(getattr(self.opt, "synthetic_min_items", 1))
        n_max = int(getattr(self.opt, "synthetic_max_items", 3))
        no_overlap = bool(getattr(self.opt, "synthetic_no_overlap", False))
        n_obj = np.random.randint(n_min, n_max + 1)

        same_class_prob = float(getattr(self.opt, "synthetic_same_class_prob", 0.0))
        chosen_train_id = None
        if np.random.rand() < same_class_prob and len(self.pseudo_object_lib) > 0:
            tids = sorted(list({int(it["train_id"]) for it in self.pseudo_object_lib}))
            if len(tids) > 0:
                chosen_train_id = int(np.random.choice(tids))

        if len(self.pseudo_object_lib) == 0:
            A_img = self._build_synthetic_A_img(size_hw, T_img)
            return A_img, E_img.copy()

        tries_per_obj = 300
        for _ in range(n_obj):
            obj = self._sample_pseudo_object(chosen_train_id=chosen_train_id)
            transformed = self._transform_pseudo_object(obj)
            if transformed is None:
                continue

            mask = transformed["mask"]
            delta = transformed["delta"]
            train_id = transformed["train_id"]

            h, w = mask.shape[:2]
            if h >= H or w >= W:
                continue

            for _ in range(tries_per_obj):
                x = np.random.randint(0, W - w + 1)
                y = np.random.randint(0, H - h + 1)

                tray_region = T[y:y + h, x:x + w]
                if not np.all(tray_region[mask > 0]):
                    continue

                if no_overlap and np.any(occ[y:y + h, x:x + w] & (mask > 0)):
                    continue

                color_rgb = self._train_id_to_rgb(int(train_id))

                region_A = canvas_A[y:y + h, x:x + w]
                region_A[mask > 0] = color_rgb
                canvas_A[y:y + h, x:x + w] = region_A

                region_OD = OD_total[y:y + h, x:x + w]
                for c in range(3):
                    region_OD[..., c][mask > 0] += delta[..., c][mask > 0]
                OD_total[y:y + h, x:x + w] = region_OD

                occ_region = occ[y:y + h, x:x + w]
                occ_region[mask > 0] = True
                occ[y:y + h, x:x + w] = occ_region
                break

                

        # convert OD back to RGB
        B_pseudo = self._od_to_rgb(OD_total)

        # outside tray = exact empty tray
        outside = ~T
        B_pseudo[outside] = E_np[outside]

        # mild scanner-like intensity perturbation
        if np.random.rand() < 0.8:
            noise_sigma = np.random.uniform(0.5, 2.0)
            noise = np.random.normal(0.0, noise_sigma, size=B_pseudo.shape).astype(np.float32)
            B_pseudo = np.clip(B_pseudo.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # mild blur sometimes
        if np.random.rand() < 0.3:
            sigma = np.random.uniform(0.3, 0.8)
            B_pseudo = cv2.GaussianBlur(B_pseudo, (0, 0), sigmaX=sigma, sigmaY=sigma)

        A_img = Image.fromarray(canvas_A)
        B_img = Image.fromarray(B_pseudo)
        return A_img, B_img

    # -------------------------
    # Basic helpers
    # -------------------------
    def _to_gray_rgb(self, img: Image.Image) -> Image.Image:
        return img.convert("L").convert("RGB")

    def _binary_from_pil(self, img: Image.Image, thr255: int) -> np.ndarray:
        arr = np.array(img.convert("L"))
        return (arr > thr255).astype(np.uint8)

    def _mask_from_Aimg(self, A_img) -> np.ndarray:
        if isinstance(A_img, Image.Image):
            A = np.array(A_img)
        elif torch.is_tensor(A_img):
            A = A_img.detach().cpu().numpy()
            if A.ndim == 3:
                A = np.transpose(A, (1, 2, 0))
        else:
            A = np.array(A_img)

        if A.ndim == 2:
            return (A > 0)
        return np.any(A > 0, axis=2)

    def _rgb_to_od(self, img_rgb: np.ndarray, eps: float = 1e-6, gamma: float = 1.0) -> np.ndarray:
        x = img_rgb.astype(np.float32) / 255.0
        x = np.clip(x, 0.0, 1.0)
        if gamma != 1.0:
            x = np.power(x, gamma)
        return -np.log(x + eps)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        x = np.exp(-od)
        x = np.clip(x, 0.0, 1.0)
        return (x * 255.0).round().astype(np.uint8)

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
    # Shift logic
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
    def _move_object_in_B_with_shift(self, B_img: Image.Image, E_img: Image.Image,
                                 M_old: np.ndarray, dx: int, dy: int, T: np.ndarray = None):
        """
        Move object in B by shifting its OD delta relative to E.

        This is much safer than raw pixel cut-paste for X-ray because:
        B ~= exp(-(OD_E + delta_obj))
        so we should shift delta_obj, not RGB intensities.

        Inputs:
        B_img : real target image
        E_img : empty tray image aligned to B
        M_old : object mask at ORIGINAL position
        dx,dy : desired shift
        T     : tray mask (optional)
        """
        B = np.array(B_img).astype(np.uint8)
        E = np.array(E_img).astype(np.uint8)

        M_old = (M_old > 0).astype(np.uint8)
        if M_old.ndim != 2:
            raise ValueError("M_old must be HxW")

        # optional dilation so moved target matches training mask support better
        dil = int(getattr(self.opt, "tray_obj_dilate_px", 0))
        if dil > 0:
            M_obj = self._dilate_bin(M_old, dil)
        else:
            M_obj = M_old.copy()

        dx_final, dy_final = self._clamp_shift_to_image(M_obj, int(dx), int(dy))

        # If tray mask is provided, reduce shift if shifted mask spills outside tray
        if T is not None:
            M_shifted = self._shift_np(M_obj, dx_final, dy_final, fill=0)
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
                        cand_dx, cand_dy = self._clamp_shift_to_image(M_obj, cand_dx, cand_dy)
                        M_cand = self._shift_np(M_obj, cand_dx, cand_dy, fill=0)
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

        # -------- OD / delta based move --------
        eps = 1e-6
        gamma = 1.0

        OD_B = self._rgb_to_od(B, eps=eps, gamma=gamma)
        OD_E = self._rgb_to_od(E, eps=eps, gamma=gamma)

        # Object contribution in OD space
        delta = np.maximum(OD_B - OD_E, 0.0)

        # Keep only original object region
        delta_obj = delta.copy()
        delta_obj[M_obj == 0] = 0.0

        # Shift delta and shifted mask
        delta_shift = self._shift_np(delta_obj, dx, dy, fill=0.0)
        M_shift = self._shift_np(M_obj.astype(np.uint8), dx, dy, fill=0).astype(np.uint8)

        # Recompose on top of empty tray
        OD_new = OD_E.copy()
        inside = (M_shift > 0)

        for c in range(3):
            OD_new[..., c][inside] += delta_shift[..., c][inside]

        # Outside tray keep exact empty tray if tray mask provided
        B_new = self._od_to_rgb(OD_new)
        if T is not None:
            outside = (T == 0)
            B_new[outside] = E[outside]

        return Image.fromarray(B_new)

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

    def _random_delta_augment(self, delta: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Augment OD-space object residual:
        delta = OD(B) - OD(E)

        This is much better than plain RGB augmentation because it changes
        attenuation magnitude/texture while preserving X-ray composition logic.

        Args:
            delta: HxWx3 float32 OD residual crop
            mask:  HxW uint8/bool object mask crop

        Returns:
            augmented delta, same shape as input
        """
        out = delta.astype(np.float32).copy()
        m = (mask > 0)

        if not np.any(m):
            return out

        # -------------------------
        # 1) Global attenuation scaling
        # -------------------------
        scale = np.random.uniform(0.90, 1.12)
        out[m] *= scale

        # -------------------------
        # 2) Low-frequency thickness variation
        # simulates slightly different material density / projection thickness
        # -------------------------
        h, w = mask.shape[:2]
        noise_small = np.random.normal(loc=0.0, scale=0.08, size=(max(8, h // 8), max(8, w // 8))).astype(np.float32)
        noise_field = cv2.resize(noise_small, (w, h), interpolation=cv2.INTER_CUBIC)

        # smooth it
        noise_field = cv2.GaussianBlur(noise_field, (0, 0), sigmaX=5.0, sigmaY=5.0)

        # convert to multiplicative field
        # range roughly ~ [0.9, 1.1]
        field = 1.0 + 0.10 * noise_field
        field = np.clip(field, 0.88, 1.15)

        for c in range(3):
            ch = out[..., c]
            ch[m] *= field[m]
            out[..., c] = ch

        # -------------------------
        # 3) Edge emphasis / suppression
        # helps model not memorize one exact contour profile
        # -------------------------
        mask_u8 = (m.astype(np.uint8) * 255)
        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5).astype(np.float32)

        if dist.max() > 1e-6:
            dist = dist / (dist.max() + 1e-6)
            edge_band = 1.0 - dist   # stronger near boundary
            edge_gain = np.random.uniform(-0.05, 0.12)  # small only
            edge_factor = 1.0 + edge_gain * edge_band
            edge_factor = np.clip(edge_factor, 0.90, 1.15)

            for c in range(3):
                ch = out[..., c]
                ch[m] *= edge_factor[m]
                out[..., c] = ch

        # -------------------------
        # 4) Fine interior OD noise
        # -------------------------
        fine_sigma = np.random.uniform(0.003, 0.015)
        fine_noise = np.random.normal(0.0, fine_sigma, size=out.shape).astype(np.float32)
        out[m] += fine_noise[m]

        # -------------------------
        # 5) Mild blur or sharpen in OD space
        # -------------------------
        r = np.random.rand()
        if r < 0.33:
            blur_sigma = np.random.uniform(0.4, 1.0)
            out = cv2.GaussianBlur(out, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        elif r < 0.66:
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.8, sigmaY=0.8)
            out = out + 0.25 * (out - blur)

        # keep only masked area
        out[~m] = 0.0

        # physics-safe clamp
        out = np.clip(out, 0.0, 4.0).astype(np.float32)
        return out

    def _random_mask_augment(self, mask: np.ndarray) -> np.ndarray:
        """
        Small shape perturbation so the generator does not memorize one exact silhouette.
        """
        m = (mask > 0).astype(np.uint8)

        if m.sum() < 20:
            return m

        # random erode/dilate
        if np.random.rand() < 0.7:
            px = np.random.randint(1, 3)  # 1 or 2
            k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
            if np.random.rand() < 0.5:
                m = cv2.dilate(m, k, iterations=1)
            else:
                m = cv2.erode(m, k, iterations=1)

        # small blur + threshold
        if np.random.rand() < 0.5:
            m2 = cv2.GaussianBlur((m * 255).astype(np.uint8), (5, 5), 0.8)
            _, m = cv2.threshold(m2, 127, 1, cv2.THRESH_BINARY)

        return m.astype(np.uint8)

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

        use_synth = self.synthetic_enabled and (np.random.rand() < self.synthetic_prob)

        E_img = None
        loaded_from_dir = False
        if bool(getattr(self.opt, "use_delta_comp", False)):
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

        T_img = None
        T_bin = None
        if self.use_tray_mask:
            T_img = self._load_tray_T(A_img.size)
            T_bin = self._get_tray_bin(T_img)

        dx = dy = 0

        if use_synth:
            if not self.use_tray_mask:
                raise RuntimeError("Synthetic mode currently requires --use_tray_mask.")
            if E_img is None:
                raise RuntimeError("Synthetic pseudo-target mode requires E_img.")

            A_img, B_img = self._build_synthetic_A_and_Bpseudo(
                (A_img.size[1], A_img.size[0]), T_img, E_img
            )
        else:
            """
            if self.use_tray_mask:
                A_old = A_img
                dx, dy = self._compute_autoshift(A_img, T_img)

                if dx != 0 or dy != 0:
                    A_img = self._shift_pil_rgb(A_img, dx, dy)
                    if E_img is not None:
                        M_old = self._mask_from_Aimg(A_old).astype(np.uint8)
                        B_img = self._move_object_in_B_with_shift(B_img, E_img, M_old, dx, dy, T=T_bin)

                    if (index % int(self.debug_every) == 0):
                        M_new = self._mask_from_Aimg(A_img).astype(np.uint8)
                        bbA = self._bbox_from_binary(M_new)
                        print(f"[debug-shift] {Path(AB_path).name} shift=({dx},{dy}) A_bbox={bbA}")
            """
            pass

        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)

        # Match global empty to B BEFORE transform, using aligned original geometry
        if bool(getattr(self.opt, "use_delta_comp", False)):
            using_global_empty = (not loaded_from_dir) and bool(getattr(self.opt, "empty_path", ""))
            if self.match_empty_to_B and using_global_empty and (E_img is not None) and (not use_synth):
                obj_mask = self._mask_from_Aimg(A_img)
                E_img = self._match_empty_to_B(E_img, B_img, obj_mask)

        # Sample one shared transform and apply same geometry to everything
        transform_params = get_params(self.opt, A_img.size)

        A_img_t = self._apply_shared_geom_to_mask_rgb(A_img, transform_params)

        if self.use_tray_mask and T_img is not None:
            T_img_t = self._apply_shared_geom_to_mask_rgb(T_img.convert("RGB"), transform_params).convert("L")
        else:
            T_img_t = None

        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))
        B = B_transform(B_img)

        E = None
        if E_img is not None:
            E = B_transform(E_img)

        # Build conditioning from transformed mask, not original mask
        app_img_t = None
        if self.use_appearance_channel:
            if use_synth:
                app_img_raw = self._extract_object_grayscale_from_B(B_img, A_img)
            else:
                app_img_raw = self._extract_appearance_from_B(B_img, A_img)

            # apply same geometry first
            app_img_t = self._apply_shared_geom_to_mask_rgb(app_img_raw.convert("RGB"), transform_params).convert("L")

            if getattr(self.opt, "phase", "") == "train":
                p_zero = float(getattr(self.opt, "appearance_zero_prob", 0.35))
                p_weak = float(getattr(self.opt, "appearance_weak_prob", 0.35))
                p_proto = float(getattr(self.opt, "appearance_proto_prob", 0.15))

                r = np.random.rand()

                if r < p_zero:
                    app_img_t = self._zero_appearance_img(A_img_t)

                elif r < (p_zero + p_weak):
                    app_img_t = self._weak_blur_appearance_img(app_img_t, A_img_t)

                elif r < (p_zero + p_weak + p_proto):
                    app_img_t = self._sample_prototype_appearance_img(A_img_t)

                else:
                    pass  # keep real appearance

            if getattr(self.opt, "phase", "") == "test" and bool(getattr(self.opt, "disable_test_appearance", False)):
                app_img_t = self._zero_appearance_img(A_img_t)

        A, cond_chs = self._rgbmask_to_condition_tensor(A_img_t, app_img=app_img_t)
        A_vis_img = self._build_condition_vis_from_channels(cond_chs)
        A_vis = B_transform(A_vis_img)

        instance_masks = None
        if self.return_instance_masks:
            instance_masks = self._extract_instance_masks_tensor(A_img_t)

        T = None
        if self.use_tray_mask and T_img_t is not None:
            T_np = self._get_tray_bin(T_img_t).astype(np.float32)
            T = torch.from_numpy(T_np[None, :, :]).float()

        if bool(getattr(self.opt, "use_delta_comp", False)):
            if (getattr(self.opt, "phase", "") == "train") and (index % int(self.debug_every) == 0):
                msg = (
                    f"[debug] {Path(AB_path).name} | synth={use_synth} | shift(dx,dy)=({dx},{dy}) | "
                    f"A(min,max)=({A.min().item():.3f},{A.max().item():.3f}) "
                    f"B(min,max)=({B.min().item():.3f},{B.max().item():.3f}) "
                    f"E(min,max)=({E.min().item():.3f},{E.max().item():.3f})"
                )
                if instance_masks is not None:
                    msg += f" Ninst={int(instance_masks.shape[0])}"
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
                "A_vis": A_vis,
                "B": B,
                "E": E,
                "A_paths": AB_path,
                "B_paths": AB_path,
                "is_synthetic": use_synth,
            }
            if T is not None:
                out["T"] = T
            if instance_masks is not None:
                out["instance_masks"] = instance_masks
            return out

        out = {
            "A": A,
            "A_vis": A_vis,
            "B": B,
            "A_paths": AB_path,
            "B_paths": AB_path,
            "is_synthetic": use_synth,
        }
        if T is not None:
            out["T"] = T
        if instance_masks is not None:
            out["instance_masks"] = instance_masks
        return out

    def __len__(self):
        return len(self.AB_paths)