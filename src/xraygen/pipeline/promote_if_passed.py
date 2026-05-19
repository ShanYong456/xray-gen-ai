from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing promotion artifact: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote model artifacts when DVC gates pass.")
    ap.add_argument("--candidate_eval", required=True)
    ap.add_argument("--production_dir", required=True)
    ap.add_argument("--generator_checkpoint_dir", required=True)
    ap.add_argument("--spatial_model_dir", required=True)
    ap.add_argument("--threat_model_dir", required=True)
    ap.add_argument("--allow_hold_output", action="store_true")
    args = ap.parse_args()

    candidate_eval_path = Path(args.candidate_eval)
    candidate_eval = load_json(candidate_eval_path)
    production_dir = Path(args.production_dir)

    reset_dir(production_dir)

    registry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "candidate_eval": str(candidate_eval_path),
        "passed": bool(candidate_eval.get("passed", False)),
        "decision": candidate_eval.get("decision", "hold"),
        "summary": candidate_eval.get("summary", {}),
        "artifacts": {},
    }

    if not candidate_eval.get("passed", False):
        registry["status"] = "not_promoted"
        registry["reason"] = "candidate gates did not pass"
        (production_dir / "model_registry.json").write_text(json.dumps(registry, indent=2))
        print("[promote] candidate held; wrote hold registry")
        if args.allow_hold_output:
            return
        raise SystemExit("Candidate gates did not pass; promotion stopped")

    generator_src = Path(args.generator_checkpoint_dir)
    spatial_src = Path(args.spatial_model_dir)
    threat_src = Path(args.threat_model_dir)

    generator_dst = production_dir / "generator" / generator_src.name
    spatial_dst = production_dir / "classifier" / "spatial_overlap_isolated"
    threat_dst = production_dir / "classifier" / "threat_contraband_noncontraband"

    for filename in ["latest_net_G.pth", "train_opt.txt", "test_opt.txt", "loss_log.txt"]:
        src = generator_src / filename
        if src.exists():
            copy_required_file(src, generator_dst / filename)

    for model_src, model_dst in [(spatial_src, spatial_dst), (threat_src, threat_dst)]:
        for filename in ["model.pt"]:
            copy_required_file(model_src / filename, model_dst / filename)
        for metrics_file in sorted(model_src.glob("metrics_*.json")):
            copy_required_file(metrics_file, model_dst / metrics_file.name)
        best = model_src / "checkpoints" / "best.pt"
        if best.exists():
            copy_required_file(best, model_dst / "checkpoints" / "best.pt")

    copy_required_file(candidate_eval_path, production_dir / "candidate_eval.json")

    registry["status"] = "promoted"
    registry["artifacts"] = {
        "generator": str(generator_dst),
        "spatial_classifier": str(spatial_dst),
        "threat_classifier": str(threat_dst),
    }
    (production_dir / "model_registry.json").write_text(json.dumps(registry, indent=2))
    print(f"[promote] promoted artifacts to {production_dir}")


if __name__ == "__main__":
    main()
