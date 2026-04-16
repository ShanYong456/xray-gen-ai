from pathlib import Path
import cv2

# ======================
# SETTINGS
# ======================
IN_DIR  = Path("data/raw/Empty")          # <-- your empty tray folder
OUT_DIR = Path("data/interim/empty_trays_clahe") # <-- output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZE = 1024  # or 1024, must match what you want later

# CLAHE params (good defaults; tweak if needed)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)

def resize_img(img_bgr, size):
    return cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)

def apply_clahe_bgr(img_bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Apply CLAHE on luminance channel (LAB) so colors don't shift weirdly.
    Works for both grayscale-looking and pseudo-color X-ray images.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l2 = clahe.apply(l)

    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def main():
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [p for p in sorted(IN_DIR.iterdir()) if p.suffix.lower() in exts]

    print("Found:", len(paths), "images in", IN_DIR)

    ok = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print("WARN: failed reading:", p)
            continue

        img = resize_img(img, SIZE)
        img = apply_clahe_bgr(img, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID)

        out_path = OUT_DIR / (p.stem + f"_r{SIZE}_clahe.png")
        if cv2.imwrite(str(out_path), img):
            ok += 1

    print("Done. Wrote:", ok, "to", OUT_DIR.resolve())

if __name__ == "__main__":
    main()