#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
from datetime import datetime

import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA


# =============================================================================
# Helpers
# =============================================================================

def list_image_files(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def load_gray3_pil_from_path(img_path: Path) -> Image.Image:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img3)


def init_feature_extractor(device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    return {
        "model": model,
        "transform": transform,
        "device": device,
        "name": "resnet50_imagenet_penultimate",
        "dim": 2048,
    }


@torch.no_grad()
def extract_feature_from_pil(pil_img: Image.Image, extractor: dict) -> np.ndarray:
    x = extractor["transform"](pil_img).unsqueeze(0).to(extractor["device"])
    feat = extractor["model"](x)
    feat = feat.squeeze(0).detach().cpu().numpy().astype(np.float64)
    return feat


def build_real_feature_distribution(real_ref_dir: Path, extractor: dict, pca_dim: int = 32):
    image_paths = list_image_files(real_ref_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in reference directory: {real_ref_dir}")

    feats = []
    used_paths = []

    for p in image_paths:
        try:
            pil_img = load_gray3_pil_from_path(p)
            feat = extract_feature_from_pil(pil_img, extractor)
            feats.append(feat)
            used_paths.append(str(p))
        except Exception as e:
            print(f"[ref] skipped {p} | reason={e}")

    if len(feats) < 2:
        raise RuntimeError(f"Need at least 2 valid reference images. Got {len(feats)}")

    X = np.stack(feats, axis=0)

    max_valid_dim = min(len(feats) - 1, X.shape[1])
    pca_dim = max(2, min(int(pca_dim), int(max_valid_dim)))

    pca = PCA(n_components=pca_dim, svd_solver="auto", whiten=False, random_state=0)
    Xp = pca.fit_transform(X)

    mu = Xp.mean(axis=0)
    cov = np.cov(Xp, rowvar=False)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-6
    cov_inv = np.linalg.pinv(cov)

    return {
        "feature_name": extractor["name"],
        "num_real_images": len(feats),
        "raw_feature_dim": int(X.shape[1]),
        "pca_dim": int(pca_dim),
        "pca_model": pca,
        "mean": mu,
        "cov_inv": cov_inv,
        "used_paths": used_paths,
    }


def mahalanobis_distance(feat: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> float:
    delta = feat - mu
    dist2 = float(delta.T @ cov_inv @ delta)
    dist2 = max(dist2, 0.0)
    return float(np.sqrt(dist2))


def summarize_values(values):
    vals = np.array(values, dtype=np.float64)
    if len(vals) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q10": None,
            "q25": None,
            "q50": None,
            "q75": None,
            "q90": None,
            "q95": None,
            "q99": None,
        }

    return {
        "count": int(len(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "q10": float(np.percentile(vals, 10)),
        "q25": float(np.percentile(vals, 25)),
        "q50": float(np.percentile(vals, 50)),
        "q75": float(np.percentile(vals, 75)),
        "q90": float(np.percentile(vals, 90)),
        "q95": float(np.percentile(vals, 95)),
        "q99": float(np.percentile(vals, 99)),
    }


def evaluate_target_images(target_dir: Path, real_dist: dict, extractor: dict):
    image_paths = list_image_files(target_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in target directory: {target_dir}")

    pca = real_dist["pca_model"]
    per_image = {}
    values = []

    for p in image_paths:
        stem = p.stem
        try:
            pil_img = load_gray3_pil_from_path(p)
            feat = extract_feature_from_pil(pil_img, extractor).reshape(1, -1)
            feat_pca = pca.transform(feat)[0]
            d = mahalanobis_distance(feat_pca, real_dist["mean"], real_dist["cov_inv"])

            per_image[stem] = {
                "path": str(p),
                "mahalanobis": float(d),
                "status": "ok",
            }
            values.append(float(d))
            print(f"[real-test] {p.name} | mahalanobis={d:.6f}")
        except Exception as e:
            per_image[stem] = {
                "path": str(p),
                "mahalanobis": None,
                "status": f"failed: {e}",
            }
            print(f"[real-test] {p.name} | failed: {e}")

    stats = summarize_values(values)
    return {
        "per_image": per_image,
        "stats": stats,
    }


def load_generated_scores_from_summary(summary_json: Path):
    data = json.loads(summary_json.read_text())

    # New format
    realism = data.get("realism_mahalanobis", {})
    per_image = realism.get("per_image", {})

    # Backward compatibility with old format
    if not per_image:
        mahal = data.get("mahalanobis", {})
        per_image = mahal.get("per_image", {})

    out = {}
    vals = []

    for stem, item in per_image.items():
        score = item.get("mahalanobis")
        status = item.get("status")
        if status == "ok" and score is not None:
            out[stem] = {
                "mahalanobis": float(score),
                "status": "ok",
            }
            vals.append(float(score))
        else:
            out[stem] = {
                "mahalanobis": None,
                "status": status,
            }

    return {
        "per_image": out,
        "stats": summarize_values(vals),
        "raw_summary": data,
    }

def deduce_thresholds(real_stats: dict):
    q95 = real_stats["q95"]
    q99 = real_stats["q99"]
    maxv = real_stats["max"]

    if q95 is None:
        raise RuntimeError("Cannot deduce thresholds because real stats are empty.")

    # Main rule:
    # <= q95            : real-like / good
    # q95 to q99        : borderline
    # > q99             : outlier / poor
    #
    # Fallback if q99 is unstable or too close to q95:
    if q99 is None or q99 <= q95:
        q99 = maxv

    return {
        "good_upper": float(q95),
        "borderline_upper": float(q99 if q99 is not None else maxv),
        "bad_lower": float(q99 if q99 is not None else maxv),
        "rule_text": (
            "good if score <= held-out real q95; "
            "borderline if q95 < score <= q99; "
            "poor/outlier if score > q99"
        ),
    }


def classify_score(score: float, thresholds: dict) -> str:
    if score is None:
        return "unscored"
    if score <= thresholds["good_upper"]:
        return "good_real_like"
    if score <= thresholds["borderline_upper"]:
        return "borderline"
    return "poor_outlier"


def compare_generated_against_real(real_eval: dict, generated_eval: dict):
    real_stats = real_eval["stats"]
    thresholds = deduce_thresholds(real_stats)

    classified = {}
    counts = {
        "good_real_like": 0,
        "borderline": 0,
        "poor_outlier": 0,
        "unscored": 0,
    }

    for stem, item in generated_eval["per_image"].items():
        score = item.get("mahalanobis")
        label = classify_score(score, thresholds)
        classified[stem] = {
            "mahalanobis": score,
            "status": item.get("status"),
            "classification": label,
        }
        counts[label] += 1

    return {
        "thresholds": thresholds,
        "generated_classification": classified,
        "classification_counts": counts,
    }

def fmt_stat(x):
    return "None" if x is None else f"{x:.6f}"


def print_report(real_eval: dict, generated_eval: dict | None, comparison: dict | None):
    rs = real_eval["stats"]
    print("\n================ REAL HELD-OUT STATS ================")
    print(
        f"count={rs['count']} | mean={rs['mean']:.6f} | std={rs['std']:.6f} | "
        f"min={rs['min']:.6f} | q95={rs['q95']:.6f} | q99={rs['q99']:.6f} | max={rs['max']:.6f}"
    )

    if generated_eval is not None:
        gs = generated_eval["stats"]
        print("\n================ GENERATED STATS ================")
        print(
            f"count={gs['count']} | mean={gs['mean']:.6f} | std={gs['std']:.6f} | "
            f"min={gs['min']:.6f} | q95={gs['q95']:.6f} | q99={gs['q99']:.6f} | max={gs['max']:.6f}"
        )

    if comparison is not None:
        th = comparison["thresholds"]
        cc = comparison["classification_counts"]
        print("\n================ DEDUCED STANDARD ================")
        print(f"good_real_like : score <= {th['good_upper']:.6f}")
        print(f"borderline     : {th['good_upper']:.6f} < score <= {th['borderline_upper']:.6f}")
        print(f"poor_outlier   : score > {th['bad_lower']:.6f}")
        print(f"rule           : {th['rule_text']}")

        print("\n================ GENERATED CLASS COUNTS ================")
        for k, v in cc.items():
            print(f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_ref_dir", required=True,
                    help="Reference real images used to fit the real feature distribution.")
    ap.add_argument("--target_dir", required=True,
                    help="Held-out real images to evaluate.")
    ap.add_argument("--out_json", required=True,
                    help="Where to save the final evaluation report JSON.")
    ap.add_argument("--mahal_pca_dim", type=int, default=32,
                    help="PCA dimension before Mahalanobis scoring.")
    ap.add_argument("--generated_summary_json", type=str, default="",
                    help="Optional generated summary JSON from generate_pix2pixV2_MAHADIST.py")
    args = ap.parse_args()

    real_ref_dir = Path(args.real_ref_dir)
    target_dir = Path(args.target_dir)
    out_json = Path(args.out_json)

    extractor = init_feature_extractor()

    real_dist = build_real_feature_distribution(
        real_ref_dir=real_ref_dir,
        extractor=extractor,
        pca_dim=args.mahal_pca_dim,
    )

    real_eval = evaluate_target_images(
        target_dir=target_dir,
        real_dist=real_dist,
        extractor=extractor,
    )

    generated_eval = None
    comparison = None
    if args.generated_summary_json:
        generated_eval = load_generated_scores_from_summary(Path(args.generated_summary_json))
        comparison = compare_generated_against_real(real_eval, generated_eval)

    report = {
        "feature_name": real_dist["feature_name"],
        "reference_real_dir": str(real_ref_dir),
        "target_real_dir": str(target_dir),
        "num_reference_images": real_dist["num_real_images"],
        "raw_feature_dim": real_dist["raw_feature_dim"],
        "pca_dim": real_dist["pca_dim"],
        "real_eval": real_eval,
        "generated_eval": generated_eval,
        "comparison": comparison,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    print_report(real_eval, generated_eval, comparison)
    print(f"\nSaved report JSON to: {out_json}")


if __name__ == "__main__":
    main()
"""
python Codes_Notebooks/Pix2Pix/realdataset_mahalanobis.py \
  --real_ref_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/train \
  --target_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test \
  --generated_summary_json results/_gen_stage23_combo_tray/generated/generated_combo_summary.json \
  --out_json results/_gen_stage23_combo_tray/generated/real_vs_generated_report.json \
  --mahal_pca_dim 32



"""