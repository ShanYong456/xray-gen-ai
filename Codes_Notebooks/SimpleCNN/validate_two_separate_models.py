from pathlib import Path
import os, json, csv, random, logging
from typing import Dict, List
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# VALIDATE BOTH SEPARATE MODELS
# ============================================================
# This script validates:
# 1) Spatial model: isolated vs overlap
# 2) Threat model: non_contraband vs contraband
#
# It expects that you already ran:
#   freeze_two_model_datasets.py
#   train_single_task_classifier.py with TASK="spatial"
#   train_single_task_classifier.py with TASK="threat"
# ============================================================

TASKS_TO_VALIDATE = ["spatial", "threat"]   # use ["spatial"], ["threat"], or both

# ============================================================
# CONFIG
# ============================================================
class BaseConfig:
    PROCESSED_ROOT = Path("../../data/processed/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/gray")
    INDEX_CSV = PROCESSED_ROOT / "index.csv"

    IMAGE_MODE = "gray"
    IMAGE_SIZE = 1024
    BATCH_SIZE = 8
    NUM_WORKERS = 2

    GRAY_MEAN = (0.5,)
    GRAY_STD = (0.25,)

    SEED = 42

    OUTPUT_ROOT = Path("../../reports/validation/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models")

class TaskConfig:
    def __init__(self, task: str):
        if task not in ["spatial", "threat"]:
            raise ValueError("task must be 'spatial' or 'threat'")

        self.task = task

        if task == "spatial":
            self.label_dir = Path("../../data/labels/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/gray/spatial_overlap_isolated")
            self.class_names = ["isolated", "overlap"]  # 0, 1
            self.model_dir = Path("../../models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/spatial_overlap_isolated")
        else:
            self.label_dir = Path("../../data/labels/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/gray/threat_contraband_noncontraband")
            self.class_names = ["non_contraband", "contraband"]  # 0, 1
            self.model_dir = Path("../../models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/threat_contraband_noncontraband")

        self.test_labels_json = self.label_dir / "test.json"

        # Prefer best checkpoint from training script
        self.model_path_candidates = [
            self.model_dir / "checkpoints" / "best.pt",
            self.model_dir / "model.pt",
            self.model_dir / "checkpoints" / "best_checkpoint.pt",
        ]

        self.output_dir = BaseConfig.OUTPUT_ROOT / task
        self.output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# REPRODUCIBILITY
# ============================================================
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(BaseConfig.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# DATA HELPERS
# ============================================================
def read_index_csv(index_csv_path: Path) -> List[Dict]:
    if not index_csv_path.exists():
        raise FileNotFoundError(f"Index CSV not found: {index_csv_path}")

    rows = []
    with open(index_csv_path, "r", newline="") as f:
        rows.extend(csv.DictReader(f))
    return rows

def load_label_map(label_path: Path) -> Dict[str, int]:
    if not label_path.exists():
        raise FileNotFoundError(f"Label JSON not found: {label_path}")

    data = json.load(open(label_path, "r"))
    out = {}

    for item in data:
        filepath = item["image"].replace("\\", "/").strip()
        fname = Path(filepath).name
        y = int(item["class_id"])

        # Store both full relative filepath and filename for matching
        out[filepath] = y
        out[fname] = y

    return out

class SingleTaskDataset(Dataset):
    def __init__(self, index_rows, processed_root, split, label_map, transform=None):
        self.processed_root = processed_root
        self.split = split
        self.transform = transform
        self.filepaths = []
        self.labels = []

        missing = []

        for r in index_rows:
            fp = r["filepath"].replace("\\", "/").strip()

            if r["split"].lower() != split:
                continue

            fname = Path(fp).name
            label = label_map.get(fp, label_map.get(fname))

            # Skip samples not present in this task label JSON
            if label is None:
                continue

            img_path = processed_root / fp
            if not img_path.exists():
                missing.append(str(img_path))
                continue

            self.filepaths.append(fp)
            self.labels.append(int(label))

        if missing:
            raise FileNotFoundError(f"Missing image files, first 10: {missing[:10]}")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.processed_root / self.filepaths[idx]
        img = Image.open(img_path).convert("L" if BaseConfig.IMAGE_MODE == "gray" else "RGB")

        if self.transform:
            img = self.transform(img)

        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, y, self.filepaths[idx]

# Same validation transform as training script
test_transform = T.Compose([
    T.Resize(int(BaseConfig.IMAGE_SIZE * 1.10)),
    T.CenterCrop(BaseConfig.IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(BaseConfig.GRAY_MEAN, BaseConfig.GRAY_STD),
])

# ============================================================
# MODEL - must match train_single_task_classifier.py
# ============================================================
class SimpleCNN_Binary(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.classifier(x)

def find_model_path(task_cfg: TaskConfig) -> Path:
    for p in task_cfg.model_path_candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No model found. Checked:\n" + "\n".join(str(p) for p in task_cfg.model_path_candidates)
    )

def load_model(task_cfg: TaskConfig):
    in_channels = 1 if BaseConfig.IMAGE_MODE == "gray" else 3
    model = SimpleCNN_Binary(in_channels=in_channels, num_classes=2).to(device)

    model_path = find_model_path(task_cfg)
    ckpt = torch.load(model_path, map_location=device)

    # Support both raw state_dict and checkpoint dict
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print(f"[{task_cfg.task}] Loaded model:", model_path)
    return model, model_path

# ============================================================
# EVALUATION
# ============================================================
@torch.no_grad()
def evaluate_model(model, loader):
    y_true, y_pred, y_prob, filepaths = [], [], [], []

    for imgs, labels, fps in loader:
        imgs = imgs.to(device, non_blocking=True)

        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].cpu().numpy().tolist())
        filepaths.extend(list(fps))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    precision_bin, recall_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "filepaths": filepaths,
        "accuracy": acc,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_binary_class1": float(precision_bin),
        "recall_binary_class1": float(recall_bin),
        "f1_binary_class1": float(f1_bin),
        "auc": auc,
        "confusion_matrix": cm,
    }

def save_confusion_matrix_plot(cm, class_names, out_path: Path, title: str):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], class_names, rotation=25, ha="right")
    plt.yticks([0, 1], class_names)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_prediction_csv(task_cfg: TaskConfig, metrics):
    out_csv = task_cfg.output_dir / f"{task_cfg.task}_test_predictions.csv"

    with open(out_csv, "w", newline="") as f:
        fieldnames = [
            "filepath",
            "true_id",
            "true_label",
            "pred_id",
            "pred_label",
            "prob_class1",
            "correct",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for fp, yt, yp, prob in zip(
            metrics["filepaths"],
            metrics["y_true"],
            metrics["y_pred"],
            metrics["y_prob"],
        ):
            w.writerow({
                "filepath": fp,
                "true_id": int(yt),
                "true_label": task_cfg.class_names[int(yt)],
                "pred_id": int(yp),
                "pred_label": task_cfg.class_names[int(yp)],
                "prob_class1": float(prob),
                "correct": int(yt == yp),
            })

    return out_csv

def save_summary_json(task_cfg: TaskConfig, metrics, model_path: Path):
    summary = {
        "task": task_cfg.task,
        "model_path": str(model_path),
        "class_names": task_cfg.class_names,
        "class_id_mapping": {
            "0": task_cfg.class_names[0],
            "1": task_cfg.class_names[1],
        },
        "num_test_samples": int(len(metrics["y_true"])),
        "test_label_distribution": {
            task_cfg.class_names[0]: int((metrics["y_true"] == 0).sum()),
            task_cfg.class_names[1]: int((metrics["y_true"] == 1).sum()),
        },
        "accuracy": metrics["accuracy"],
        "auc": metrics["auc"],
        "precision_binary_class1": metrics["precision_binary_class1"],
        "recall_binary_class1": metrics["recall_binary_class1"],
        "f1_binary_class1": metrics["f1_binary_class1"],
        "precision_per_class": {
            task_cfg.class_names[0]: metrics["precision_per_class"][0],
            task_cfg.class_names[1]: metrics["precision_per_class"][1],
        },
        "recall_per_class": {
            task_cfg.class_names[0]: metrics["recall_per_class"][0],
            task_cfg.class_names[1]: metrics["recall_per_class"][1],
        },
        "f1_per_class": {
            task_cfg.class_names[0]: metrics["f1_per_class"][0],
            task_cfg.class_names[1]: metrics["f1_per_class"][1],
        },
        "confusion_matrix_rows_true_cols_pred": metrics["confusion_matrix"].tolist(),
    }

    out_json = task_cfg.output_dir / f"{task_cfg.task}_test_metrics.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    return out_json, summary

def print_metrics(task_cfg: TaskConfig, summary):
    print("\n" + "=" * 80)
    print(f"VALIDATION RESULT: {task_cfg.task.upper()}")
    print("=" * 80)
    print(f"Class 0 = {task_cfg.class_names[0]}")
    print(f"Class 1 = {task_cfg.class_names[1]}")
    print("Test samples:", summary["num_test_samples"])
    print("Label distribution:", summary["test_label_distribution"])
    print(f"Accuracy: {summary['accuracy']:.4f}")

    if summary["auc"] is not None:
        print(f"AUC: {summary['auc']:.4f}")
    else:
        print("AUC: not available, likely only one class exists in test set")

    print(f"Class-1 Precision: {summary['precision_binary_class1']:.4f}")
    print(f"Class-1 Recall:    {summary['recall_binary_class1']:.4f}")
    print(f"Class-1 F1:        {summary['f1_binary_class1']:.4f}")
    print("Confusion Matrix rows=true, cols=pred:")
    print(np.array(summary["confusion_matrix_rows_true_cols_pred"]))

def validate_one_task(task: str):
    task_cfg = TaskConfig(task)

    index_rows = read_index_csv(BaseConfig.INDEX_CSV)
    test_label_map = load_label_map(task_cfg.test_labels_json)

    test_ds = SingleTaskDataset(
        index_rows=index_rows,
        processed_root=BaseConfig.PROCESSED_ROOT,
        split="test",
        label_map=test_label_map,
        transform=test_transform,
    )

    print(f"\n[{task}] Test samples:", len(test_ds))
    print(f"[{task}] Test distribution:", Counter(test_ds.labels))

    if len(test_ds) == 0:
        raise RuntimeError(f"No test samples found for task={task}")

    test_loader = DataLoader(
        test_ds,
        batch_size=BaseConfig.BATCH_SIZE,
        shuffle=False,
        num_workers=BaseConfig.NUM_WORKERS,
        pin_memory=True,
    )

    model, model_path = load_model(task_cfg)
    metrics = evaluate_model(model, test_loader)

    out_csv = save_prediction_csv(task_cfg, metrics)
    out_json, summary = save_summary_json(task_cfg, metrics, model_path)

    cm_path = task_cfg.output_dir / f"{task}_confusion_matrix.png"
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        task_cfg.class_names,
        cm_path,
        title=f"{task.capitalize()} Confusion Matrix",
    )

    # Save sklearn text report
    report_txt = classification_report(
        metrics["y_true"],
        metrics["y_pred"],
        labels=[0, 1],
        target_names=task_cfg.class_names,
        zero_division=0,
    )

    report_path = task_cfg.output_dir / f"{task}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report_txt)

    print_metrics(task_cfg, summary)
    print("Saved metrics JSON:", out_json)
    print("Saved predictions CSV:", out_csv)
    print("Saved confusion matrix:", cm_path)
    print("Saved classification report:", report_path)

    return summary

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    all_summaries = {}

    for task in TASKS_TO_VALIDATE:
        all_summaries[task] = validate_one_task(task)

    BaseConfig.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    combined_path = BaseConfig.OUTPUT_ROOT / "combined_validation_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE - combined summary saved to:")
    print(combined_path)
    print("=" * 80)
