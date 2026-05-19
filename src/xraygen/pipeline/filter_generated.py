from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile from an empty list")
    values = sorted(values)
    pct = max(0.0, min(100.0, pct))
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    candidates = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.stem.startswith(stem)
    )
    return candidates[0] if candidates else None


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter generated images using realism and novelty metrics.")
    ap.add_argument("--generated_dir", required=True)
    ap.add_argument("--mahal_json", required=True)
    ap.add_argument("--novelty_json", required=True)
    ap.add_argument("--accepted_dir", required=True)
    ap.add_argument("--rejected_dir", required=True)
    ap.add_argument("--report_json", required=True)
    ap.add_argument("--mahal_percentile", type=float, default=75.0)
    ap.add_argument("--mahal_max", type=float, default=None)
    ap.add_argument("--novelty_min", type=float, default=0.0)
    ap.add_argument("--copy_metrics", action="store_true")
    args = ap.parse_args()

    generated_dir = Path(args.generated_dir)
    accepted_dir = Path(args.accepted_dir)
    rejected_dir = Path(args.rejected_dir)
    report_json = Path(args.report_json)

    mahal = load_json(Path(args.mahal_json))
    novelty = load_json(Path(args.novelty_json))

    mahal_per_image = mahal.get("per_image", {})
    novelty_per_image = novelty.get("per_image", {})
    mahal_values = [
        float(v["mahalanobis"])
        for v in mahal_per_image.values()
        if v.get("status") == "ok" and "mahalanobis" in v
    ]

    mahal_max = args.mahal_max
    if mahal_max is None:
        mahal_max = percentile(mahal_values, args.mahal_percentile)

    reset_dir(accepted_dir)
    reset_dir(rejected_dir)

    accepted = []
    rejected = []
    missing_images = []

    for stem, mahal_info in sorted(mahal_per_image.items()):
        img_path = find_image_for_stem(generated_dir, stem)
        if img_path is None:
            missing_images.append(stem)
            continue

        mahal_score = float(mahal_info.get("mahalanobis", float("inf")))
        novelty_info = novelty_per_image.get(stem, {})
        novelty_score = float(novelty_info.get("nearest_real_distance", 0.0))

        reasons = []
        if mahal_info.get("status") != "ok":
            reasons.append("mahal_status_not_ok")
        if novelty_info.get("status", "ok") != "ok":
            reasons.append("novelty_status_not_ok")
        if mahal_score > float(mahal_max):
            reasons.append("mahal_above_threshold")
        if novelty_score < float(args.novelty_min):
            reasons.append("novelty_below_threshold")

        row = {
            "stem": stem,
            "image": img_path.name,
            "mahalanobis": mahal_score,
            "nearest_real_distance": novelty_score,
            "reasons": reasons,
        }

        if reasons:
            shutil.copy2(img_path, rejected_dir / img_path.name)
            rejected.append(row)
        else:
            shutil.copy2(img_path, accepted_dir / img_path.name)
            accepted.append(row)

    if args.copy_metrics:
        for src in [Path(args.mahal_json), Path(args.novelty_json)]:
            if src.exists():
                shutil.copy2(src, accepted_dir / src.name)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "generated_dir": str(generated_dir),
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
        "mahal_threshold": float(mahal_max),
        "mahal_threshold_source": (
            "explicit" if args.mahal_max is not None else f"generated_p{args.mahal_percentile:g}"
        ),
        "novelty_min": float(args.novelty_min),
        "num_generated_with_scores": len(mahal_per_image),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "acceptance_rate": (len(accepted) / len(mahal_per_image)) if mahal_per_image else 0.0,
        "missing_images": missing_images,
        "accepted": accepted,
        "rejected": rejected,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2))

    print(f"[filter] accepted={len(accepted)} rejected={len(rejected)}")
    print(f"[filter] mahal_threshold={float(mahal_max):.4f}")
    print(f"[filter] wrote {report_json}")


if __name__ == "__main__":
    main()
