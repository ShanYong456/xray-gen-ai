from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def gate(name: str, value: float, op: str, threshold: float) -> dict:
    if op == ">=":
        passed = value >= threshold
    elif op == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unsupported gate operator: {op}")
    return {
        "name": name,
        "value": value,
        "operator": op,
        "threshold": threshold,
        "passed": bool(passed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate automated pipeline gates.")
    ap.add_argument("--filter_report", required=True)
    ap.add_argument("--fid_metrics", required=True)
    ap.add_argument("--classifier_summary", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--min_accepted", type=int, default=50)
    ap.add_argument("--min_acceptance_rate", type=float, default=0.50)
    ap.add_argument("--max_fid", type=float, default=80.0)
    ap.add_argument("--min_spatial_accuracy", type=float, default=0.85)
    ap.add_argument("--min_threat_accuracy", type=float, default=0.85)
    ap.add_argument("--min_spatial_auc", type=float, default=0.90)
    ap.add_argument("--min_threat_auc", type=float, default=0.90)
    args = ap.parse_args()

    filter_report = load_json(Path(args.filter_report))
    fid_metrics = load_json(Path(args.fid_metrics))
    classifier_summary = load_json(Path(args.classifier_summary))

    spatial = classifier_summary.get("spatial", {})
    threat = classifier_summary.get("threat", {})

    fid_value = fid_metrics.get("fid", fid_metrics.get("frechet_inception_distance"))
    if fid_value is None:
        fid_value = fid_metrics.get("raw_metrics", {}).get("frechet_inception_distance")
    if fid_value is None:
        raise ValueError("Could not find FID value in metrics JSON")

    gates = [
        gate("accepted_images", int(filter_report.get("num_accepted", 0)), ">=", args.min_accepted),
        gate("acceptance_rate", float(filter_report.get("acceptance_rate", 0.0)), ">=", args.min_acceptance_rate),
        gate("fid", float(fid_value), "<=", args.max_fid),
        gate("spatial_accuracy", float(spatial.get("accuracy", 0.0)), ">=", args.min_spatial_accuracy),
        gate("threat_accuracy", float(threat.get("accuracy", 0.0)), ">=", args.min_threat_accuracy),
        gate("spatial_auc", float(spatial.get("auc", 0.0)), ">=", args.min_spatial_auc),
        gate("threat_auc", float(threat.get("auc", 0.0)), ">=", args.min_threat_auc),
    ]

    passed = all(g["passed"] for g in gates)
    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "passed": bool(passed),
        "decision": "promote" if passed else "hold",
        "gates": gates,
        "summary": {
            "num_accepted": int(filter_report.get("num_accepted", 0)),
            "num_rejected": int(filter_report.get("num_rejected", 0)),
            "acceptance_rate": float(filter_report.get("acceptance_rate", 0.0)),
            "fid": float(fid_value),
            "spatial_accuracy": float(spatial.get("accuracy", 0.0)),
            "spatial_auc": float(spatial.get("auc", 0.0)),
            "threat_accuracy": float(threat.get("accuracy", 0.0)),
            "threat_auc": float(threat.get("auc", 0.0)),
        },
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, indent=2))

    print(f"[candidate] decision={output['decision']}")
    for item in gates:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"[candidate] {status} {item['name']}: "
            f"{item['value']} {item['operator']} {item['threshold']}"
        )


if __name__ == "__main__":
    main()
