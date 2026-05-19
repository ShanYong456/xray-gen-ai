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


# ============================================================
# Image loading
# ============================================================

def list_image_files(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    if not folder.exists():
        raise RuntimeError(f"Folder does not exist: {folder}")

    paths = sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ])

    return paths


def load_gray3_pil(img_path: Path):
    """
    Load X-ray image as grayscale, then convert to 3-channel RGB.

    Reason:
    - Your X-ray images are grayscale.
    - ResNet50 expects 3-channel input.
    - So we copy grayscale into RGB channels.
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(img3)


# ============================================================
# Feature extractor
# ============================================================

def init_feature_extractor(device=None):
    """
    Uses ImageNet-pretrained ResNet50 as embedding model.

    Output feature:
    - ResNet50 penultimate layer
    - 2048-dimensional feature vector
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Remove final classification layer
    model.fc = torch.nn.Identity()

    model.eval()
    model.to(device)

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return {
        "model": model,
        "transform": transform,
        "device": device,
        "name": "resnet50_imagenet_penultimate",
        "dim": 2048,
    }


@torch.no_grad()
def extract_feature_from_pil(pil_img, extractor):
    x = extractor["transform"](pil_img).unsqueeze(0).to(extractor["device"])
    feat = extractor["model"](x)

    return feat.squeeze(0).detach().cpu().numpy().astype(np.float64)


def extract_features_from_dir(folder: Path, extractor):
    paths = list_image_files(folder)

    if not paths:
        raise RuntimeError(f"No images found in: {folder}")

    feats = {}
    failed = {}

    print(f"[info] Found {len(paths)} images in {folder}")

    for i, p in enumerate(paths, start=1):
        try:
            pil_img = load_gray3_pil(p)
            feats[p.stem] = extract_feature_from_pil(pil_img, extractor)

            if i % 50 == 0 or i == len(paths):
                print(f"[extract] {i}/{len(paths)}")

        except Exception as e:
            failed[p.stem] = str(e)
            print(f"[skip] {p.name} | {e}")

    if len(feats) < 2:
        raise RuntimeError(f"Need at least 2 valid images. Got {len(feats)}")

    return feats, failed


# ============================================================
# Mahalanobis distribution
# ============================================================

def build_mahal_distribution(ref_feats: dict, pca_dim: int):
    """
    Build the real-image reference distribution.

    Steps:
    1. Stack real reference image features
    2. Reduce feature dimension using PCA
    3. Compute mean and covariance
    4. Use inverse covariance for Mahalanobis distance
    """

    stems = list(ref_feats.keys())
    X = np.stack([ref_feats[s] for s in stems], axis=0)

    # PCA dim cannot be more than number of samples - 1
    max_valid_dim = min(len(stems) - 1, X.shape[1])
    pca_dim = max(2, min(int(pca_dim), int(max_valid_dim)))

    print(f"[mahal] Raw feature dim: {X.shape[1]}")
    print(f"[mahal] Requested PCA dim: {pca_dim}")

    pca = PCA(
        n_components=pca_dim,
        svd_solver="auto",
        whiten=False,
        random_state=0,
    )

    Xp = pca.fit_transform(X)

    mu = Xp.mean(axis=0)

    cov = np.cov(Xp, rowvar=False)

    # Small regularization to avoid unstable inverse covariance
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * 1e-6

    cov_inv = np.linalg.pinv(cov)

    return {
        "pca": pca,
        "mean": mu,
        "cov_inv": cov_inv,
        "pca_dim": pca_dim,
        "raw_dim": X.shape[1],
        "num_ref": len(stems),
        "ref_stems": stems,
    }


def mahalanobis_distance(feat_pca, mu, cov_inv):
    delta = feat_pca - mu
    dist2 = float(delta.T @ cov_inv @ delta)

    # Avoid tiny negative values due to numerical precision
    return float(np.sqrt(max(dist2, 0.0)))


def score_features(score_feats: dict, dist: dict):
    per_image = {}
    vals = []

    for stem, feat in score_feats.items():
        feat_pca = dist["pca"].transform(feat.reshape(1, -1))[0]

        d = mahalanobis_distance(
            feat_pca,
            dist["mean"],
            dist["cov_inv"],
        )

        per_image[stem] = {
            "mahalanobis": float(d),
            "status": "ok",
        }

        vals.append(float(d))

    vals = np.array(vals, dtype=np.float64)

    return {
        "per_image": per_image,
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.percentile(vals, 25)),
        "p75": float(np.percentile(vals, 75)),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Score real/test/generated images against a real-image Mahalanobis reference distribution."
    )

    ap.add_argument(
        "--real_eval_dir",
        required=True,
        help="Real images used to build the Mahalanobis reference distribution.",
    )

    ap.add_argument(
        "--score_dir",
        required=True,
        help="Images to score against the real reference distribution.",
    )

    ap.add_argument(
        "--mahal_pca_dim",
        type=int,
        default=32,
        help="PCA dimension used before Mahalanobis distance.",
    )

    ap.add_argument(
        "--out_json",
        required=True,
        help="Path to save output JSON result.",
    )

    args = ap.parse_args()

    real_eval_dir = Path(args.real_eval_dir)
    score_dir = Path(args.score_dir)
    out_json = Path(args.out_json)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("[device] Loading feature extractor...")
    extractor = init_feature_extractor()
    print(f"[device] Using: {extractor['device']}")

    print(f"\n[real] Extracting reference features from:")
    print(f"       {real_eval_dir}")
    ref_feats, ref_failed = extract_features_from_dir(real_eval_dir, extractor)

    print(f"\n[score] Extracting score features from:")
    print(f"        {score_dir}")
    score_feats, score_failed = extract_features_from_dir(score_dir, extractor)

    print("\n[mahal] Fitting PCA + covariance on real reference images...")
    dist = build_mahal_distribution(ref_feats, args.mahal_pca_dim)

    print("\n[mahal] Scoring images...")
    result = score_features(score_feats, dist)

    output = {
        "feature_name": extractor["name"],
        "real_eval_dir": str(real_eval_dir),
        "score_dir": str(score_dir),
        "num_real_reference_images": dist["num_ref"],
        "num_scored_images": len(score_feats),
        "raw_feature_dim": int(dist["raw_dim"]),
        "pca_dim": int(dist["pca_dim"]),

        "mean": result["mean"],
        "min": result["min"],
        "max": result["max"],
        "std": result["std"],
        "median": result["median"],
        "p25": result["p25"],
        "p75": result["p75"],

        "per_image": result["per_image"],
        "failed_reference_images": ref_failed,
        "failed_score_images": score_failed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    out_json.write_text(json.dumps(output, indent=2))

    print("\n===== MAHALANOBIS SCORE RESULT =====")
    print(f"Reference real images : {dist['num_ref']}")
    print(f"Scored images         : {len(score_feats)}")
    print(f"Raw feature dim       : {dist['raw_dim']}")
    print(f"PCA dim               : {dist['pca_dim']}")
    print(f"Mean distance         : {result['mean']:.4f}")
    print(f"Median distance       : {result['median']:.4f}")
    print(f"Min distance          : {result['min']:.4f}")
    print(f"Max distance          : {result['max']:.4f}")
    print(f"Std distance          : {result['std']:.4f}")
    print(f"P25 distance          : {result['p25']:.4f}")
    print(f"P75 distance          : {result['p75']:.4f}")
    print(f"Saved to              : {out_json}")

    if ref_failed:
        print(f"\n[warning] Failed reference images: {len(ref_failed)}")

    if score_failed:
        print(f"[warning] Failed score images: {len(score_failed)}")


if __name__ == "__main__":
    main()


"""
python Codes_Notebooks/Pix2Pix/generate_realscore_pix2pix_MAHADIST.py \
  --real_eval_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/train \
  --score_dir datasets/SHAMPOOBLADEWITHTRAY_COMPLETE/test \
  --mahal_pca_dim 32 \
  --out_json results/real_test_mahal_against_train.json

"""