#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps
import argparse

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(src: Path):
    return [p for p in sorted(src.rglob("*")) if p.suffix.lower() in EXTS]

def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))

def pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    pad_left = (side - w) // 2
    pad_top = (side - h) // 2
    pad_right = side - w - pad_left
    pad_bottom = side - h - pad_top
    return ImageOps.expand(img, border=(pad_left, pad_top, pad_right, pad_bottom), fill=fill)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder of RGB images (3-channel)")
    ap.add_argument("--dst", required=True, help="Output folder for square images")
    ap.add_argument("--size", type=int, default=256, help="Output size (256 or 512)")
    ap.add_argument(
        "--mode",
        choices=["resize", "center_crop", "pad"],
        default="center_crop",
        help="How to make square images before resize",
    )
    ap.add_argument("--pad_fill", type=int, default=0, help="Pad fill value if mode=pad (0-255)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    img_paths = list_images(src)
    if not img_paths:
        raise SystemExit(f"No images found in {src}")

    total = 0
    for p in img_paths:
        img = Image.open(p).convert("RGB")

        if args.mode == "resize":
            out_img = img.resize((args.size, args.size), Image.BICUBIC)

        elif args.mode == "center_crop":
            sq = center_crop_square(img)
            out_img = sq.resize((args.size, args.size), Image.BICUBIC)

        elif args.mode == "pad":
            fill = (args.pad_fill, args.pad_fill, args.pad_fill)
            sq = pad_to_square(img, fill=fill)
            out_img = sq.resize((args.size, args.size), Image.BICUBIC)

        out_path = dst / f"{p.stem}.png"
        out_img.save(out_path)
        total += 1

    print(f"Wrote {total} images -> {dst} (mode={args.mode}, size={args.size})")

if __name__ == "__main__":
    main()

"""
python Codes_Notebooks/StyleGan/make_patches.py \
  --src data/interim/GAN/Stage1/gray_clahe_STYLEGAN \
  --dst data/interim/GAN/Stage1/myset_full_256_pad \
  --size 256 \
  --mode pad \
  --pad_fill 0


"""
