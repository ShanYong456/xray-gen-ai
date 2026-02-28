import cv2
import numpy as np
import random
from pathlib import Path

SIZE = 1024
OUT_COUNT = 10

OBJECT_LIB = Path("data/raw/Contraband/Metal/Cropped")
OUT_DIR = Path("datasets/Contraband/Metal/gen_random_masks")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- load object cutouts ----------
def load_objects():
    objs = []
    for cls in OBJECT_LIB.iterdir():
        if not cls.is_dir():
            continue
        for img in cls.glob("*.png"):
            m = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue

            # convert RGBA → RGB mask
            if m.shape[2] == 4:
                alpha = m[:,:,3] > 0
                rgb = m[:,:,:3]
                rgb[~alpha] = 0
                m = rgb

            objs.append(m)
    print("Loaded objects:", len(objs))
    return objs


# ---------- place object ----------
def place(canvas, obj):
    h, w = obj.shape[:2]

    # random scale
    scale = random.uniform(0.3, 1.3)
    obj = cv2.resize(obj, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    # random rotation
    angle = random.uniform(0, 360)
    M = cv2.getRotationMatrix2D((obj.shape[1]/2, obj.shape[0]/2), angle, 1)
    obj = cv2.warpAffine(obj, M, (obj.shape[1], obj.shape[0]),
                         flags=cv2.INTER_NEAREST,
                         borderValue=(0,0,0))

    h, w = obj.shape[:2]

    # allow partial outside tray (realistic)
    x = random.randint(-w//3, SIZE - w//3)
    y = random.randint(-h//3, SIZE - h//3)

    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x+w, SIZE)
    y2 = min(y+h, SIZE)

    obj_x1 = max(0, -x)
    obj_y1 = max(0, -y)
    obj_x2 = obj_x1 + (x2-x1)
    obj_y2 = obj_y1 + (y2-y1)

    region = canvas[y1:y2, x1:x2]
    obj_crop = obj[obj_y1:obj_y2, obj_x1:obj_x2]

    mask = obj_crop.sum(axis=2) > 0
    region[mask] = obj_crop[mask]


# ---------- generate one tray ----------
def generate(objects, idx):
    canvas = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)

    # random number of items
    n = random.randint(1, 7)

    # clustering behavior
    if random.random() < 0.35:
        center = (random.randint(300,700), random.randint(300,700))
    else:
        center = None

    for _ in range(n):
        obj = random.choice(objects)

        if center:
            # clustered clutter
            dx = random.randint(-200,200)
            dy = random.randint(-200,200)
            tmp = np.zeros_like(canvas)
            place(tmp, obj)
            canvas = np.maximum(canvas, tmp)
        else:
            place(canvas, obj)

    cv2.imwrite(str(OUT_DIR / f"{idx:06d}.png"), canvas)


# ---------- main ----------
def main():
    objects = load_objects()

    for i in range(OUT_COUNT):
        generate(objects, i)
        if i % 100 == 0:
            print("generated", i)

    print("DONE")


if __name__ == "__main__":
    main()