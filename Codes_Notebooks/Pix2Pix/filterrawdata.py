import json
import shutil
from pathlib import Path


# =========================
# CONFIG
# =========================
INPUT_JSON = "/home/ssy/Downloads/SHAMPOOBLADEINTRAY_TGT/result.json"
OUTPUT_JSON = "/home/ssy/Downloads/SHAMPOOBLADEINTRAY_TGT/result_tray_blade_or_tray_shampoo.json"

OUTPUT_IMAGE_DIR = "/home/ssy/Downloads/SHAMPOOBLADEINTRAY_TGT/tray_blade_or_tray_shampoo_images"

COPY_IMAGES = True

TRAY_CATEGORY_NAME = "tray"
BLADE_CATEGORY_NAME = "blade"
SHAMPOO_CATEGORY_NAME = "shampoo"

IMAGE_SEARCH_ROOT = "/home/ssy/Downloads/SHAMPOOBLADEINTRAY_TGT/images"


# =========================
# HELPERS
# =========================
def normalize(name):
    return str(name).strip().lower()


def build_filename_index(root):
    index = {}
    for p in Path(root).rglob("*"):
        if p.is_file():
            index.setdefault(p.name, p)
    return index


def resolve_path(file_name, root, index):
    raw = str(file_name)

    # 1) direct
    p = Path(raw)
    if p.exists():
        return p

    # 2) fix ../../home/...
    if "/home/" in raw:
        fixed = Path(raw[raw.index("/home/"):])
        if fixed.exists():
            return fixed

    # 3) basename under root
    base = Path(raw).name
    p2 = Path(root) / base
    if p2.exists():
        return p2

    # 4) indexed search
    if base in index:
        return index[base]

    return None


# =========================
# MAIN
# =========================
def main():
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    # =========================
    # 1. Find category IDs
    # =========================
    tray_id = None
    blade_id = None
    shampoo_id = None

    for c in categories:
        cname = normalize(c["name"])
        if cname == normalize(TRAY_CATEGORY_NAME):
            tray_id = c["id"]
        elif cname == normalize(BLADE_CATEGORY_NAME):
            blade_id = c["id"]
        elif cname == normalize(SHAMPOO_CATEGORY_NAME):
            shampoo_id = c["id"]

    if tray_id is None:
        raise ValueError("Tray category not found")
    if blade_id is None:
        raise ValueError("Blade category not found")
    if shampoo_id is None:
        raise ValueError("Shampoo category not found")

    print(f"Tray ID: {tray_id}")
    print(f"Blade ID: {blade_id}")
    print(f"Shampoo ID: {shampoo_id}")

    # =========================
    # 2. Build per-image category presence
    # =========================
    image_to_cat_ids = {}

    for ann in annotations:
        image_to_cat_ids.setdefault(ann["image_id"], set()).add(ann["category_id"])

    # Keep only:
    #   (tray AND blade) OR (tray AND shampoo)
    selected_image_ids = set()

    for image_id, cat_ids in image_to_cat_ids.items():
        has_tray = tray_id in cat_ids
        has_blade = blade_id in cat_ids
        has_shampoo = shampoo_id in cat_ids

        if (has_tray and has_blade) or (has_tray and has_shampoo):
            selected_image_ids.add(image_id)

    # =========================
    # 3. Filter images
    # =========================
    filtered_images = [img for img in images if img["id"] in selected_image_ids]

    # =========================
    # 4. Filter annotations
    # Keep all annotations belonging to selected images
    # =========================
    filtered_annotations = [
        ann for ann in annotations
        if ann["image_id"] in selected_image_ids
    ]

    # =========================
    # 5. Keep original categories
    # No category removal
    # =========================
    filtered_categories = categories

    # Optional: preserve original category ids exactly
    output = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": filtered_categories
    }

    # =========================
    # 6. Save new JSON
    # =========================
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    # =========================
    # 7. Print summary
    # =========================
    num_tray_blade = 0
    num_tray_shampoo = 0
    num_all_three = 0

    for image_id in selected_image_ids:
        cat_ids = image_to_cat_ids.get(image_id, set())
        has_tray = tray_id in cat_ids
        has_blade = blade_id in cat_ids
        has_shampoo = shampoo_id in cat_ids

        if has_tray and has_blade and has_shampoo:
            num_all_three += 1
        elif has_tray and has_blade:
            num_tray_blade += 1
        elif has_tray and has_shampoo:
            num_tray_shampoo += 1

    print("\n===== SUMMARY =====")
    print(f"Images kept: {len(filtered_images)}")
    print(f"Annotations kept: {len(filtered_annotations)}")
    print(f"Categories kept: {[c['name'] for c in filtered_categories]}")
    print(f"Tray + Blade only: {num_tray_blade}")
    print(f"Tray + Shampoo only: {num_tray_shampoo}")
    print(f"Tray + Blade + Shampoo: {num_all_three}")

    # =========================
    # 8. Copy images
    # =========================
    if COPY_IMAGES:
        root = Path(IMAGE_SEARCH_ROOT)
        index = build_filename_index(root)

        out_dir = Path(OUTPUT_IMAGE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        missing = 0

        for img in filtered_images:
            src = resolve_path(img["file_name"], root, index)

            if src is None:
                print(f"[MISS] {img['file_name']}")
                missing += 1
                continue

            dst = out_dir / src.name
            shutil.copy2(src, dst)
            copied += 1

        print("\n===== COPY RESULT =====")
        print(f"Copied: {copied}")
        print(f"Missing: {missing}")


if __name__ == "__main__":
    main()