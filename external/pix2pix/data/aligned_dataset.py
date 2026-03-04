import os
from pathlib import Path
from PIL import Image
import numpy as np

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """Paired {A,B} dataset. Optional E (empty tray) for delta compositing.

    Notes:
    - If output_nc=3 but X-ray is grayscale-ish, we force B/E to gray->RGB to avoid color artifacts.
    - If using ONE global --empty_path, we can match E exposure to each sample's B (background only).
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size

        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

        # If training output_nc=3, keep RGB I/O but remove color degrees of freedom for X-ray.
        self.force_gray_rgb = (self.output_nc == 3)

        # Only really useful when using a single global empty_path (not per-sample empty_dir)
        self.match_empty_to_B = True

        # Debug throttling
        self.debug_every = 50

    # -------------------------
    # Helpers
    # -------------------------
    def _to_gray_rgb(self, img: Image.Image) -> Image.Image:
        """Convert to grayscale then replicate to RGB (3 identical channels)."""
        return img.convert("L").convert("RGB")

    def _mask_from_Aimg(self, A_img: Image.Image) -> np.ndarray:
        """Boolean mask True where object exists, computed from A_img (mask image)."""
        A = np.array(A_img)
        if A.ndim == 2:
            return (A > 0)
        # Use >0 on any channel
        return np.any(A > 0, axis=2)

    def _match_empty_to_B(self, E_img: Image.Image, B_img: Image.Image, obj_mask: np.ndarray) -> Image.Image:
        """
        Match E exposure to B using BACKGROUND pixels only (outside obj_mask).
        Per-channel affine: E' = a*E + b.

        All inputs MUST have identical H/W here.
        """
        E = np.array(E_img).astype(np.float32)
        B = np.array(B_img).astype(np.float32)

        # Safety: all must match spatially
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
    # Empty tray loader
    # -------------------------
    def _load_empty_E(self, AB_path: str):
        """
        Returns: (E_img: PIL.Image RGB, loaded_from_dir: bool)

        - If empty_dir is set, tries to match per-sample empty tray.
        - Else falls back to single empty_path.
        """
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

        # Force B to grayscale-replicated RGB if desired (prevents yellow/odd tints)
        if self.force_gray_rgb:
            B_img = self._to_gray_rgb(B_img)

        # Apply same geometric transform to both A and B
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        # OPTIONAL: also return E if delta comp is enabled
        if getattr(self.opt, "use_delta_comp", False):
            try:
                E_img, loaded_from_dir = self._load_empty_E(AB_path)
            except FileNotFoundError as exc:
                # fallback to empty_path if exists else safe B placeholder
                empty_path = getattr(self.opt, "empty_path", "")
                if empty_path and Path(empty_path).exists():
                    E_img = Image.open(empty_path).convert("RGB")
                    if self.force_gray_rgb:
                        E_img = self._to_gray_rgb(E_img)
                    loaded_from_dir = False
                    print(f"[warning] {exc}; using --empty_path fallback for E")
                else:
                    E_img = B_img.copy()
                    loaded_from_dir = False
                    print(f"[warning] {exc}; using B as E placeholder (safer than black).")

            # CRITICAL: make sure E matches A/B size BEFORE any matching
            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BICUBIC)

            # Build object mask from A_img (same size as E_img and B_img)
            obj_mask = self._mask_from_Aimg(A_img)

            # If using global empty_path, match E exposure to B background
            using_global_empty = (not loaded_from_dir) and bool(getattr(self.opt, "empty_path", ""))
            if self.match_empty_to_B and using_global_empty:
                # Ensure B_img size matches too (it should)
                if B_img.size != A_img.size:
                    B_img = B_img.resize(A_img.size, resample=Image.BICUBIC)
                E_img = self._match_empty_to_B(E_img, B_img, obj_mask)

            # Apply SAME transform params as B (keeps alignment)
            E = B_transform(E_img)

            # ---- DEBUG PRINT (throttled) ----
            if (getattr(self.opt, "phase", "") == "train") and (index % int(self.debug_every) == 0):
                print(
                    f"[debug] {Path(AB_path).name} | "
                    f"A(min,max)=({A.min().item():.3f},{A.max().item():.3f}) "
                    f"B(min,max)=({B.min().item():.3f},{B.max().item():.3f}) "
                    f"E(min,max)=({E.min().item():.3f},{E.max().item():.3f})"
                )

            return {"A": A, "B": B, "E": E, "A_paths": AB_path, "B_paths": AB_path}

        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)