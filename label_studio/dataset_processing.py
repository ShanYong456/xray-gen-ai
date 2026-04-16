import os
import json
from collections import defaultdict

import fiftyone as fo
import fiftyone.types as fot


def build_ls_cls_plus_coco_dataset(
    images_dir: str,
    cls_json: str,
    coco_json: str,
    dataset_name: str = "xray_ls_cls_plus_coco",
    overwrite: bool = True,
    det_field: str = "det_gt",   # where boxes will be stored
    cls_field: str = "cls_gt",   # where multilabels will be stored
    launch_app: bool = False,
    port: int = 5151,
) -> fo.Dataset:
    """
    Builds a FiftyOne dataset from:
      - Label Studio classification export JSON (multi-label) -> cls_field (fo.Classifications)
      - Label Studio COCO detection export JSON -> det_field (fo.Detections)

    Returns:
      fo.Dataset
    """

    # -----------------------
    # Helpers
    # -----------------------
    def clean_ls_filename(path: str) -> str:
        base = os.path.basename(path)
        # WARNING: only keep this if your LS export prefixes "<random>-"
        if "-" in base:
            base = base.split("-", 1)[1]
        return base
    

    def extract_class_labels(task: dict):
        labels = []
        confs = {}

        for ann in task.get("annotations", []):
            for res in ann.get("result", []):
                rtype = res.get("type")
                val = res.get("value", {})

                if rtype == "choices":
                    for c in val.get("choices", []):
                        if c not in labels:
                            labels.append(c)
                            if "confidence" in res:
                                confs[c] = res.get("confidence")

                elif rtype == "choice":
                    c = val.get("choice")
                    if c and c not in labels:
                        labels.append(c)
                        if "confidence" in res:
                            confs[c] = res.get("confidence")

                elif rtype == "taxonomy":
                    tax = val.get("taxonomy", [])
                    if tax:
                        if isinstance(tax[0], list) and tax[0]:
                            c = tax[0][-1]
                        elif isinstance(tax[0], str):
                            c = tax[-1]
                        else:
                            c = None
                        if c and c not in labels:
                            labels.append(c)
                            if "confidence" in res:
                                confs[c] = res.get("confidence")

        return labels, confs

    # -----------------------
    # 1) Create dataset from images
    # -----------------------
    dataset = fo.Dataset.from_dir(
        dataset_dir=images_dir,
        dataset_type=fot.ImageDirectory,
        name=dataset_name,
        overwrite=overwrite,
    )

    by_abs = {os.path.abspath(s.filepath): s for s in dataset}
    by_name = {os.path.basename(s.filepath): s for s in dataset}

    # -----------------------
    # 2) Import classification JSON -> cls_field
    # -----------------------
    with open(cls_json, "r") as f:
        tasks = json.load(f)

    missing_cls = 0
    labeled_cls = 0
    no_label = 0

    for task in tasks:
        img = task.get("data", {}).get("image")
        if not img:
            continue

        fname = clean_ls_filename(img)
        local_path = os.path.abspath(os.path.join(images_dir, fname))

        sample = by_abs.get(local_path) or by_name.get(os.path.basename(local_path))
        if sample is None:
            print("[CLS] Image not found:", local_path, "(from:", img, ")")
            missing_cls += 1
            continue

        labels, confs = extract_class_labels(task)
        if not labels:
            no_label += 1
            continue

        sample[cls_field] = fo.Classifications(
            classifications=[
                fo.Classification(label=l, confidence=confs.get(l))
                for l in labels
            ]
        )
        sample.save()
        labeled_cls += 1

    print("CLS labeled samples:", labeled_cls)
    print("CLS missing images:", missing_cls)
    print("CLS tasks with no label:", no_label)

    # -----------------------
    # 3) Import COCO detections -> det_field
    # -----------------------
    with open(coco_json, "r") as f:
        coco_data = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}
    annots_by_img = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        annots_by_img[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in coco_data.get("images", [])}

    missing_det = 0
    merged_det = 0
    empty_det = 0

    # Create a quick map from cleaned COCO filename -> image dict to avoid O(N^2)
    coco_by_fname = {}
    for img in images.values():
        coco_fname = clean_ls_filename(os.path.basename(img["file_name"]))
        coco_by_fname[coco_fname] = img

    for sample in dataset:
        fname = os.path.basename(sample.filepath)

        matching_img = coco_by_fname.get(fname)
        if matching_img is None:
            missing_det += 1
            continue

        img_id = matching_img["id"]
        annots = annots_by_img.get(img_id, [])
        if not annots:
            empty_det += 1
            continue

        detections = []
        width = matching_img["width"]
        height = matching_img["height"]

        for ann in annots:
            label = categories.get(ann["category_id"], str(ann["category_id"]))
            x, y, w, h = ann["bbox"]  # absolute pixels
            bbox_norm = [x / width, y / height, w / width, h / height]
            detections.append(fo.Detection(label=label, bounding_box=bbox_norm))

        sample[det_field] = fo.Detections(detections=detections)
        sample.save()
        merged_det += 1

    print("DET merged samples:", merged_det)
    print("DET missing matches:", missing_det)
    print("DET samples with no dets:", empty_det)

    if launch_app:
        session = fo.launch_app(dataset, port=port)
        session.wait()

    return dataset


# Example usage:
if __name__ == "__main__":
    ds = build_ls_cls_plus_coco_dataset(
        images_dir="/home/ssy/Desktop/dataset/Gray",
        cls_json="/home/ssy/Downloads/image_classification.json",
        coco_json="/home/ssy/Downloads/object_detection/result.json",
        dataset_name="xray_ls_cls_plus_coco",
        overwrite=True,
        det_field="det_gt",
        cls_field="cls_gt",
        launch_app=False,
        port=5151,
    )

    ds = fo.load_dataset("xray_ls_cls_plus_coco")

    print("Returned dataset:", ds.name, "num_samples:", len(ds))

    # Export to an ABSOLUTE path
    EXPORT_DIR = "/home/ssy/Desktop/data_preprocessing/exports/xray_ls_cls_plus_coco_full"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    print("Exporting dataset to:", EXPORT_DIR)

    ds.export(
        export_dir=EXPORT_DIR,
        dataset_type=fot.FiftyOneDataset,
        export_media=True,
    )

    print("✅ Dataset exported to:", EXPORT_DIR)
    print("Export contains:", os.listdir(EXPORT_DIR))

    # Optional: launch AFTER export
    session = fo.launch_app(ds, port=5151)
    session.wait()
