import os
import json
import fiftyone as fo
import fiftyone.types as fot

# -----------------------
# Paths (edit these)
# -----------------------
IMAGES_DIR   = "/home/ssy/Desktop/dataset/Gray"
CLS_JSON     = "/home/ssy/Downloads/image_classification.json"   # Label Studio export (classification)
COCO_JSON    = "/home/ssy/Downloads/object_detection/result.json"        # Label Studio export (COCO detection)

DATASET_NAME = "xray_ls_cls_plus_coco"

# -----------------------
# Helpers
# -----------------------
def clean_ls_filename(path: str) -> str:
    """
    Label Studio sometimes prefixes filenames like '<random>-realname.png'.
    This removes the prefix.
    """
    base = os.path.basename(path)
    if "-" in base:
        base = base.split("-", 1)[1]
    return base

def extract_class_labels(task: dict):
    """
    Returns (labels_list, confidences_dict)
    labels_list: ["label1", "label2", ...]
    confidences_dict: optional {label: confidence}
    """
    labels = []
    confs = {}

    for ann in task.get("annotations", []):
        for res in ann.get("result", []):
            rtype = res.get("type")
            val = res.get("value", {})

            # Multi-label (checkboxes)
            if rtype == "choices":
                for c in val.get("choices", []):
                    if c not in labels:
                        labels.append(c)
                        if "confidence" in res:
                            confs[c] = res.get("confidence")

            # Single-choice
            elif rtype == "choice":
                c = val.get("choice")
                if c and c not in labels:
                    labels.append(c)
                    if "confidence" in res:
                        confs[c] = res.get("confidence")

            # Taxonomy
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
# 1) Create / load dataset from image directory
# -----------------------
dataset = fo.Dataset.from_dir(
    dataset_dir=IMAGES_DIR,
    dataset_type=fot.ImageDirectory,
    name=DATASET_NAME,
    overwrite=True,
)

# Index samples by absolute path + filename for robust matching
by_abs = {os.path.abspath(s.filepath): s for s in dataset}
by_name = {os.path.basename(s.filepath): s for s in dataset}

# -----------------------
# 2) Import Label Studio classification JSON into field "cls_gt"
# -----------------------
with open(CLS_JSON, "r") as f:
    tasks = json.load(f)

missing_cls = 0
labeled_cls = 0
no_label = 0

for task in tasks:
    img = task.get("data", {}).get("image")
    if not img:
        continue

    fname = clean_ls_filename(img)
    local_path = os.path.abspath(os.path.join(IMAGES_DIR, fname))

    sample = by_abs.get(local_path) or by_name.get(os.path.basename(local_path))
    if sample is None:
        print("CLS image not found:", local_path, "(from:", img, ")")
        missing_cls += 1
        continue

    labels, confs = extract_class_labels(task)
    if not labels:
        no_label += 1
        continue

    sample["cls_gt"] = fo.Classifications(
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
# 3) Import COCO detections into the main dataset field "det_gt"
# -----------------------
from collections import defaultdict

with open(COCO_JSON, "r") as f:
    coco_data = json.load(f)

categories = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
annots_by_img = defaultdict(list)
for ann in coco_data.get('annotations', []):
    annots_by_img[ann['image_id']].append(ann)
images = {img['id']: img for img in coco_data.get('images', [])}

missing_det = 0
merged_det = 0
empty_det = 0

for sample in dataset:
    fname = os.path.basename(sample.filepath)
    fname_clean = fname  # Dataset filenames are already cleaned

    # Find matching image in coco
    matching_img = None
    for img_id, img in images.items():
        coco_fname = os.path.basename(img['file_name'])
        coco_fname_clean = clean_ls_filename(coco_fname)
        if fname_clean == coco_fname_clean:
            matching_img = img
            break

    if matching_img is None:
        missing_det += 1
        continue

    img_id = matching_img['id']
    annots = annots_by_img[img_id]
    if not annots:
        empty_det += 1
        continue

    detections = []
    for ann in annots:
        cat_id = ann['category_id']
        label = categories.get(cat_id, str(cat_id))
        bbox = ann['bbox']  # [x, y, w, h] in absolute pixels
        # Normalize to [0,1]
        x, y, w, h = bbox
        width = matching_img['width']
        height = matching_img['height']
        bbox_norm = [x / width, y / height, w / width, h / height]
        detections.append(fo.Detection(label=label, bounding_box=bbox_norm))

    sample['ground_truth'] = fo.Detections(detections=detections)
    sample.save()
    merged_det += 1

print("DET merged samples:", merged_det)
print("DET missing matches:", missing_det)
print("DET samples with no dets:", empty_det)

# -----------------------
# 4) Launch FiftyOne
# -----------------------
session = fo.launch_app(dataset, port=5151)
session.wait()
