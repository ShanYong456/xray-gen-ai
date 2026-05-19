from __future__ import annotations

import argparse
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xraygen.pipeline import dvc_live_dashboard as dashboard


ROOT = dashboard.ROOT
TEST_MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST"


def configure_test_dashboard() -> None:
    dashboard.DVC_YAML = ROOT / "dvc_test/dvc.yaml"
    dashboard.DVC_CWD = ROOT / "dvc_test"

    dashboard.METRIC_FILES = {
        "real_reference": ROOT / "reports/dvc_test/real_test_mahal_against_train.json",
        "real_ab_reference": ROOT / "results/_dvc_test_gen_real_ab_reference/real_ab_input_summary.json",
        "generated_summary": ROOT / "reports/dvc_test/generated_combo_summary.json",
        "filter_report": ROOT / "reports/dvc_test/filter_report.json",
        "fid_train": ROOT / f"reports/dvc_test/fid_eval_runs/train/{TEST_MODEL_NAME}/epoch_latest/metrics.json",
        "fid_test": ROOT / f"reports/dvc_test/fid_eval_runs/test/{TEST_MODEL_NAME}/epoch_latest/metrics.json",
        "candidate_eval": ROOT / "reports/dvc_test/candidate_eval.json",
        "production_registry": ROOT / "models/production_test/model_registry.json",
    }

    dashboard.IMAGE_DIRS = {
        "generated": ROOT / "datasets/_dvc_test_generated_combo/generated",
        "accepted": ROOT / "datasets/_dvc_test_generated_combo/accepted",
        "rejected": ROOT / "datasets/_dvc_test_generated_combo/rejected",
        "real_ab_fake": ROOT / "results/_dvc_test_gen_real_ab_reference/fake_images",
    }

    dashboard.STATE = dashboard.DashboardState()
    dashboard.INDEX_HTML = (
        dashboard.INDEX_HTML
        .replace("<title>X-ray DVC Pipeline</title>", "<title>X-ray DVC Test Pipeline</title>")
        .replace("<h1>X-ray DVC Pipeline</h1>", "<h1>X-ray DVC Test Pipeline</h1>")
        .replace(">Start DVC</button>", ">Start Test DVC</button>")
    )


def main() -> None:
    configure_test_dashboard()

    ap = argparse.ArgumentParser(description="Run a live dashboard for the experimental DVC test pipeline.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8700)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), dashboard.Handler)
    print(f"[dashboard:test] open http://{args.host}:{args.port}")
    print(f"[dashboard:test] dvc cwd: {dashboard.DVC_CWD}")
    print(f"[dashboard:test] dvc yaml: {dashboard.DVC_YAML}")
    print("[dashboard:test] click Start Test DVC to run the test pipeline")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
