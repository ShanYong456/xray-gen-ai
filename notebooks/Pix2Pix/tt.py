import json
import shutil
from pathlib import Path


# =========================
# CONFIG
# =========================
INPUT_JSON = "/home/ssy/Downloads/SHAMPOOINTRAY/result.json"
OUTPUT_JSON = "/home/ssy/Downloads/SHAMPOOINTRAY/result_tray_only.json"

OUTPUT_IMAGE_DIR = "/home/ssy/Downloads/SHAMPOOINTRAY/tray_only_images"

COPY_IMAGES = True

TARGET_CATEGORY_NAME = "tray"
REMOVE_CATEGORY_NAME = "blade"

IMAGE_SEARCH_ROOT = "/home/ssy/Downloads/SHAMPOOINTRAY/images"

KEEP_ALL_ANNOTATIONS_FOR_SELECTED_IMAGES = True


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
    remove_id = None

    for c in categories:
        if normalize(c["name"]) == normalize(TARGET_CATEGORY_NAME):
            tray_id = c["id"]
        if normalize(c["name"]) == normalize(REMOVE_CATEGORY_NAME):
            remove_id = c["id"]

    if tray_id is None:
        raise ValueError("Tray category not found")

    print(f"Tray ID: {tray_id}")
    print(f"Blade ID (to remove): {remove_id}")

    # =========================
    # 2. Find images with tray
    # =========================
    tray_image_ids = set()

    for ann in annotations:
        if ann["category_id"] == tray_id:
            tray_image_ids.add(ann["image_id"])

    filtered_images = [img for img in images if img["id"] in tray_image_ids]

    # =========================
    # 3. Filter annotations
    # =========================
    filtered_annotations = []

    for ann in annotations:
        if ann["image_id"] not in tray_image_ids:
            continue

        # REMOVE blade
        if ann["category_id"] == remove_id:
            continue

        filtered_annotations.append(ann)

    # =========================
    # 4. Remove blade category
    # =========================
    filtered_categories = [
        c for c in categories
        if normalize(c["name"]) != normalize(REMOVE_CATEGORY_NAME)
    ]

    # =========================
    # 5. (IMPORTANT) Re-map category IDs
    # =========================
    old_to_new = {}
    new_categories = []

    for new_id, c in enumerate(filtered_categories):
        old_to_new[c["id"]] = new_id
        c["id"] = new_id
        new_categories.append(c)

    for ann in filtered_annotations:
        ann["category_id"] = old_to_new[ann["category_id"]]

    # =========================
    # 6. Save new JSON
    # =========================
    output = {
        "images": filtered_images,
        "annotations": filtered_annotations,
        "categories": new_categories
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)

    print("\n===== SUMMARY =====")
    print(f"Images kept: {len(filtered_images)}")
    print(f"Annotations kept: {len(filtered_annotations)}")
    print(f"Categories kept: {[c['name'] for c in new_categories]}")

    # =========================
    # 7. Copy images (FIXED)
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