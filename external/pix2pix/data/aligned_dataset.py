import os
from pathlib import Path
from PIL import Image

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.

    OPTIONAL: If --use_delta_comp is set, it will also return E (empty tray).
    """

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))

        assert self.opt.load_size >= self.opt.crop_size

        self.input_nc = self.opt.output_nc if self.opt.direction == "BtoA" else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == "BtoA" else self.opt.output_nc

    def _load_empty_E(self, AB_path: str) -> Image.Image:
        """Load empty tray image as PIL.Image (RGB).

        Two modes are supported:
        1) per-sample empty tray from dir (match filename exactly)
        2) fallback glob search based on timestamp substring if exact
           filename doesn't exist. This is handy when AB filenames contain
           UUID prefixes but the empty directory only has timestamp-based
           names.
        3) a single global empty image via ``--empty_path``.
        """
        # 1) per-sample empty tray from dir (match filename)
        empty_dir = getattr(self.opt, "empty_dir", "")
        if empty_dir:
            bname = Path(AB_path).name
            e_path = Path(empty_dir) / bname
            if not e_path.exists():
                # attempt to recover by matching timestamp portion
                # AB name format: <uuid>-<timestamp>_tr_<index>.png
                parts = bname.split("-", 1)
                timestamp = None
                if len(parts) == 2:
                    timestamp = parts[1].split("_tr")[0]
                if timestamp:
                    # look for exact timestamp match first
                    candidates = list(Path(empty_dir).glob(f"*{timestamp}*"))
                    if len(candidates) == 1:
                        e_path = candidates[0]
                    elif len(candidates) > 1:
                        raise FileNotFoundError(
                            f"Multiple empty tray candidates for {bname}: {candidates}"
                        )
                    else:
                        # no exact match; pick nearest timestamp numerically
                        empties = list(Path(empty_dir).iterdir())
                        def parse_ts(name):
                            # assume empty filenames are pure timestamps
                            try:
                                return name.split(".")[0]
                            except Exception:
                                return name
                        def ts_to_seconds(ts_str):
                            # convert YYYY-MM-DD_HH-MM-SS-ms to seconds
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
                            raise FileNotFoundError(
                                f"Empty tray not found for {bname}: {e_path}"
                            )
                else:
                    raise FileNotFoundError(f"Empty tray not found for {bname}: {e_path}")
            return Image.open(str(e_path)).convert("RGB")

        # 2) one global empty tray image
        empty_path = getattr(self.opt, "empty_path", "")
        if empty_path:
            e_path = Path(empty_path)
            if not e_path.exists():
                raise FileNotFoundError(f"Empty tray not found: {e_path}")
            return Image.open(str(e_path)).convert("RGB")

        raise ValueError(
            "use_delta_comp is enabled but no empty tray provided. "
            "Set --empty_dir (folder) OR --empty_path (single image)."
        )

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert("RGB")

        # split AB into A and B
        w, h = AB.size
        w2 = w // 2
        A_img = AB.crop((0, 0, w2, h))
        B_img = AB.crop((w2, 0, w, h))

        # apply same geometric transform to both A and B
        transform_params = get_params(self.opt, A_img.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A_img)
        B = B_transform(B_img)

        # OPTIONAL: also return E if delta comp is enabled
        if getattr(self.opt, "use_delta_comp", False):
            try:
                E_img = self._load_empty_E(AB_path)
            except FileNotFoundError as exc:
                # fall back to a single supplied empty_path if available
                empty_path = getattr(self.opt, "empty_path", "")
                if empty_path and Path(empty_path).exists():
                    E_img = Image.open(empty_path).convert("RGB")
                else:
                    # create a zero image matching the original B image size so training can continue
                    E_img = Image.new("RGB", B_img.size, (0, 0, 0))
                    print(f"[warning] {exc}; using zero-image placeholder for E")
            # ensure empty has same spatial dimensions as A/B before transform
            if E_img.size != A_img.size:
                E_img = E_img.resize(A_img.size, resample=Image.BILINEAR)
            # IMPORTANT: apply SAME transform params as B (keeps alignment)
            E = B_transform(E_img)
            return {"A": A, "B": B, "E": E, "A_paths": AB_path, "B_paths": AB_path}

        return {"A": A, "B": B, "A_paths": AB_path, "B_paths": AB_path}

    def __len__(self):
        return len(self.AB_paths)