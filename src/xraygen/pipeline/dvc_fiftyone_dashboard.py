from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

ROOT = Path(__file__).resolve().parents[3]
MAX_LOG_LINES = 1200
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

TEST_MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST"
DEFAULT_SOURCE_MODEL = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_COMPLETESyn"
DEFAULT_OUTPUT_MODEL = TEST_MODEL_NAME

DVC_YAML = ROOT / "dvc_test/dvc.yaml"
DVC_CWD = ROOT / "dvc_test"
CHECKPOINTS_DIR = ROOT / "checkpoints"

TEST_DATASET = "datasets/_dvc_test/SHAMPOOBLADEWITHTRAY_COMPLETE"
TRAIN_TRAY_MASKS = f"{TEST_DATASET}/matched_masks/train/tray"
TEST_TRAY_MASKS = f"{TEST_DATASET}/matched_masks/test/tray"
TRAIN_BLADE_MASKS = f"{TEST_DATASET}/matched_masks/train/blade"
TEST_BLADE_MASKS = f"{TEST_DATASET}/matched_masks/test/blade"

PIPELINE_METRIC_FILES = {
    "real_reference": ROOT / "reports/dvc_test/real_test_mahal_against_train.json",
    "real_ab_reference": ROOT / "results/_dvc_test_gen_real_ab_reference/real_ab_input_summary.json",
    "generated_summary": ROOT / "reports/dvc_test/generated_combo_summary.json",
    "filter_report": ROOT / "reports/dvc_test/filter_report.json",
    "fid_train": ROOT / f"reports/dvc_test/fid_eval_runs/train/{TEST_MODEL_NAME}/epoch_latest/metrics.json",
    "fid_test": ROOT / f"reports/dvc_test/fid_eval_runs/test/{TEST_MODEL_NAME}/epoch_latest/metrics.json",
    "candidate_eval": ROOT / "reports/dvc_test/candidate_eval.json",
    "production_registry": ROOT / "models/production_test/model_registry.json",
}

PIPELINE_IMAGE_DIRS = {
    "generated": ROOT / "datasets/_dvc_test_generated_combo/generated",
    "accepted": ROOT / "datasets/_dvc_test_generated_combo/accepted",
    "rejected": ROOT / "datasets/_dvc_test_generated_combo/rejected",
    "real_ab_fake": ROOT / "results/_dvc_test_gen_real_ab_reference/fake_images",
}


class PipelineState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.returncode: int | None = None
        self.current_stage: str | None = None
        self.logs: list[str] = []
        self.stages = self.load_stages()

    def load_stages(self) -> list[dict]:
        if not DVC_YAML.exists():
            return []
        stages: list[dict] = []
        in_stages = False
        for line in DVC_YAML.read_text().splitlines():
            if line.strip() == "stages:":
                in_stages = True
                continue
            if in_stages:
                match = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
                if match:
                    stages.append({"name": match.group(1), "status": "pending", "started_at": None, "finished_at": None})
        return stages

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        if len(self.logs) > MAX_LOG_LINES:
            self.logs = self.logs[-MAX_LOG_LINES:]

    def set_stage(self, name: str, status: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        for stage in self.stages:
            if stage["name"] == name:
                stage["status"] = status
                if status == "running" and stage.get("started_at") is None:
                    stage["started_at"] = now
                if status in {"done", "failed", "skipped"}:
                    stage["finished_at"] = now
                break

    def snapshot(self) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            total = len(self.stages)
            complete = sum(1 for stage in self.stages if stage["status"] in {"done", "skipped"})
            failed = any(stage["status"] == "failed" for stage in self.stages) or (
                self.returncode is not None and self.returncode != 0
            )
            if running:
                status = "running"
            elif failed:
                status = "failed"
            elif self.returncode == 0:
                status = "done"
            else:
                status = "idle"
            return {
                "status": status,
                "running": running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "current_stage": self.current_stage,
                "progress": (complete / total) if total else 0.0,
                "stages": self.stages,
                "logs": self.logs[-400:],
            }


class ControlState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.returncode: int | None = None
        self.action: str | None = None
        self.step: str | None = None
        self.config: dict = {}
        self.logs: list[str] = []

    def append_log(self, line: str) -> None:
        self.logs.append(line.rstrip("\n"))
        if len(self.logs) > MAX_LOG_LINES:
            self.logs = self.logs[-MAX_LOG_LINES:]

    def snapshot(self) -> dict:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            if running:
                status = "running"
            elif self.returncode == 0:
                status = "done"
            elif self.returncode is not None:
                status = "failed"
            else:
                status = "idle"
            return {
                "status": status,
                "running": running,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "returncode": self.returncode,
                "action": self.action,
                "step": self.step,
                "config": self.config,
                "logs": self.logs[-400:],
            }


PIPELINE_STATE = PipelineState()
CONTROL_STATE = ControlState()
FIFTYONE_STATE = ControlState()

DEFAULT_FIFTYONE_DATASET = TEST_DATASET
DEFAULT_FIFTYONE_EVAL_MODEL = DEFAULT_OUTPUT_MODEL
DEFAULT_FIFTYONE_EMBEDDING_MODEL = "inception-v3-imagenet-torch"
DEFAULT_FIFTYONE_PORT = 5151
FIFTYONE_SOURCE_COLORS = {
    "real": "#00ff00",
    "generated": "#ff0000",
}


def read_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        return {"error": str(exc)}
    return None


def extract_fid(metrics: dict) -> object:
    value = metrics.get("fid")
    if value is None:
        value = metrics.get("raw_metrics", {}).get("frechet_inception_distance")
    return value


def format_fid_sub(metrics: dict) -> str:
    parts = []
    phase = metrics.get("phase")
    num_images = metrics.get("num_images")
    if phase:
        parts.append(str(phase))
    if isinstance(num_images, int):
        parts.append(f"{num_images} images")
    return " | ".join(parts or ["lower is better"])


def format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "-"


def resolve_dvc_command(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("dvc")
    if found:
        return found
    local = ROOT / ".venv/bin/dvc"
    if local.exists():
        return str(local)
    return "dvc"


def collect_pipeline_metrics() -> dict:
    raw = {name: read_json(path) for name, path in PIPELINE_METRIC_FILES.items()}
    real_ab = raw.get("real_ab_reference") or []
    filter_report = raw.get("filter_report") or {}
    fid_train = raw.get("fid_train") or {}
    fid_test = raw.get("fid_test") or {}
    candidate = raw.get("candidate_eval") or {}
    registry = raw.get("production_registry") or {}

    if not isinstance(filter_report, dict):
        filter_report = {}
    if not isinstance(fid_train, dict):
        fid_train = {}
    if not isinstance(fid_test, dict):
        fid_test = {}
    if not isinstance(candidate, dict):
        candidate = {}
    if not isinstance(registry, dict):
        registry = {}
    summary = candidate.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    return {
        "raw": raw,
        "tiles": [
            {"label": "Real AB", "value": len(real_ab) if isinstance(real_ab, list) else "-", "sub": "reference inputs"},
            {"label": "Accepted", "value": filter_report.get("num_accepted", "-"), "sub": "synthetic images"},
            {"label": "Rejected", "value": filter_report.get("num_rejected", "-"), "sub": "filtered out"},
            {"label": "Acceptance", "value": format_percent(filter_report.get("acceptance_rate")), "sub": "quality gate"},
            {"label": "FID Train", "value": format_float(extract_fid(fid_train)), "sub": format_fid_sub(fid_train)},
            {"label": "FID Test", "value": format_float(extract_fid(fid_test)), "sub": format_fid_sub(fid_test)},
            {"label": "Spatial Acc", "value": format_percent(summary.get("spatial_accuracy")), "sub": "classifier"},
            {"label": "Threat Acc", "value": format_percent(summary.get("threat_accuracy")), "sub": "classifier"},
            {"label": "Decision", "value": candidate.get("decision", "-"), "sub": "candidate gate"},
            {"label": "Production", "value": registry.get("status", "-"), "sub": "promotion"},
        ],
        "gates": candidate.get("gates", []),
    }


def list_images_from_dirs(image_dirs: dict[str, Path], limit: int = 12) -> dict:
    galleries = {}
    for group, folder in image_dirs.items():
        items = []
        if folder.exists():
            paths = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)
            for path in paths[:limit]:
                items.append({"name": path.name, "url": f"/media?group={quote(group)}&name={quote(path.name)}"})
        galleries[group] = items
    return galleries


def resolve_image_path(image_dirs: dict[str, Path], group: str, name: str) -> Path | None:
    folder = image_dirs.get(group)
    if folder is None:
        return None
    path = folder / Path(name).name
    try:
        resolved = path.resolve()
        folder_resolved = folder.resolve()
    except FileNotFoundError:
        return None
    if folder_resolved not in resolved.parents:
        return None
    if not resolved.exists() or resolved.suffix.lower() not in IMAGE_EXTS:
        return None
    return resolved


def list_pipeline_gallery_images() -> dict:
    return list_images_from_dirs(PIPELINE_IMAGE_DIRS)


def resolve_pipeline_media_path(group: str, name: str) -> Path | None:
    return resolve_image_path(PIPELINE_IMAGE_DIRS, group, name)


def parse_dvc_line(line: str) -> None:
    running_match = re.search(r"Running stage '([^']+)'", line)
    skipped_match = re.search(r"Stage '([^']+)' didn't change, skipping", line)
    unchanged_match = re.search(r"Data and pipelines are up to date", line)
    with PIPELINE_STATE.lock:
        if running_match:
            name = running_match.group(1)
            if PIPELINE_STATE.current_stage and PIPELINE_STATE.current_stage != name:
                PIPELINE_STATE.set_stage(PIPELINE_STATE.current_stage, "done")
            PIPELINE_STATE.current_stage = name
            PIPELINE_STATE.set_stage(name, "running")
        elif skipped_match:
            PIPELINE_STATE.set_stage(skipped_match.group(1), "skipped")
        elif unchanged_match:
            for stage in PIPELINE_STATE.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"
        if line.startswith("ERROR:") and PIPELINE_STATE.current_stage:
            PIPELINE_STATE.set_stage(PIPELINE_STATE.current_stage, "failed")
        PIPELINE_STATE.append_log(line)


def run_pipeline_dvc(dvc_cmd: str, target: str | None = None, force: bool = False) -> None:
    cmd = [dvc_cmd, "repro"]
    if force:
        cmd.append("--force")
    if target:
        cmd.append(target)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(DVC_CWD),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        with PIPELINE_STATE.lock:
            PIPELINE_STATE.returncode = 1
            PIPELINE_STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            PIPELINE_STATE.logs = [f"$ {' '.join(cmd)}", f"[dashboard:test] ERROR: {exc}"]
            PIPELINE_STATE.process = None
        return
    with PIPELINE_STATE.lock:
        PIPELINE_STATE.process = process
        PIPELINE_STATE.started_at = datetime.now().isoformat(timespec="seconds")
        PIPELINE_STATE.finished_at = None
        PIPELINE_STATE.returncode = None
        PIPELINE_STATE.current_stage = None
        PIPELINE_STATE.logs = [f"$ {' '.join(cmd)}"]
        PIPELINE_STATE.stages = PIPELINE_STATE.load_stages()
    assert process.stdout is not None
    for line in process.stdout:
        parse_dvc_line(line)
    returncode = process.wait()
    with PIPELINE_STATE.lock:
        if PIPELINE_STATE.current_stage:
            PIPELINE_STATE.set_stage(PIPELINE_STATE.current_stage, "done" if returncode == 0 else "failed")
        if returncode == 0:
            for stage in PIPELINE_STATE.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"
        PIPELINE_STATE.returncode = returncode
        PIPELINE_STATE.finished_at = datetime.now().isoformat(timespec="seconds")
        PIPELINE_STATE.append_log(f"[dashboard:test] dvc repro exited with code {returncode}")
        PIPELINE_STATE.process = None


def stop_pipeline_dvc() -> bool:
    with PIPELINE_STATE.lock:
        proc = PIPELINE_STATE.process
    if proc is None or proc.poll() is not None:
        return False
    proc.send_signal(signal.SIGTERM)
    return True


def safe_model_name(value: object) -> str:
    name = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("Model names may only contain letters, numbers, dot, underscore, and dash.")
    return name


def safe_int(value: object, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def safe_float(value: object, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def list_checkpoint_models() -> list[dict]:
    models = []
    if not CHECKPOINTS_DIR.exists():
        return models
    for folder in sorted(path for path in CHECKPOINTS_DIR.iterdir() if path.is_dir()):
        checkpoints = sorted(folder.glob("*_net_G.pth"))
        if not checkpoints:
            continue
        epochs = sorted(
            {path.name.removesuffix("_net_G.pth") for path in checkpoints},
            key=lambda item: (item != "latest", item),
        )
        models.append({"name": folder.name, "epochs": epochs})
    return models


def model_exists(name: str) -> bool:
    return any(item["name"] == name for item in list_checkpoint_models())


def numeric_epoch(value: object) -> int | None:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def latest_numeric_epoch(model_name: str) -> int | None:
    for item in list_checkpoint_models():
        if item["name"] == model_name:
            values = [numeric_epoch(epoch) for epoch in item["epochs"]]
            numeric_values = [epoch for epoch in values if epoch is not None]
            return max(numeric_values) if numeric_values else None
    return None


def checkpoint_exists(model_name: str, epoch: str) -> bool:
    folder = CHECKPOINTS_DIR / model_name
    return (folder / f"{epoch}_net_G.pth").exists() and (folder / f"{epoch}_net_D.pth").exists()


def train_command(
    model_name: str,
    load_epoch: str,
    epoch_count: int,
    n_epochs: int,
    n_epochs_decay: int,
    max_dataset_size: int,
    lr: float,
    beta1: float,
    save_epoch_freq: int,
    synthetic_prob: float,
    synthetic_scale_min: float,
    synthetic_scale_max: float,
    synthetic_rot_min: float,
    synthetic_rot_max: float,
) -> list[str]:
    return [
        sys.executable,
        "external/pix2pix/train.py",
        "--dataroot", TEST_DATASET,
        "--name", model_name,
        "--model", "pix2pix",
        "--dataset_mode", "aligned",
        "--direction", "AtoB",
        "--input_nc", "7",
        "--output_nc", "3",
        "--class_nc", "3",
        "--thickness_nc", "1",
        "--netG", "unet_256",
        "--netD", "n_layers",
        "--n_layers_D", "4",
        "--norm", "instance",
        "--preprocess", "none",
        "--load_size", "0",
        "--crop_size", "0",
        "--no_flip",
        "--batch_size", "1",
        "--pool_size", "0",
        "--gan_mode", "lsgan",
        "--lr", f"{lr:g}",
        "--beta1", f"{beta1:g}",
        "--epoch_count", str(epoch_count),
        "--n_epochs", str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--max_dataset_size", str(max_dataset_size),
        "--save_latest_freq", "100000000",
        "--save_epoch_freq", str(save_epoch_freq),
        "--continue_train",
        "--epoch", load_epoch,
        "--no_html",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--return_instance_masks",
        "--use_masked_l1",
        "--lambda_L1", "25",
        "--lambda_bg", "0",
        "--use_grad_loss",
        "--lambda_grad", "5",
        "--use_lap_loss",
        "--lambda_lap", "2",
        "--use_ssim_loss",
        "--lambda_ssim", "2",
        "--d_label_smooth", "0.1",
        "--syn_gan_weight", "0.5",
        "--d_update_ratio", "1",
        "--use_tray_mask",
        "--tray_mask_dir", TRAIN_TRAY_MASKS,
        "--synthetic_prob", f"{synthetic_prob:g}",
        "--synthetic_combo_mode", "random",
        "--synthetic_prob_shampoo_only", "0.25",
        "--synthetic_prob_blade_only", "0.25",
        "--synthetic_prob_pair_no_overlap", "0.25",
        "--synthetic_prob_pair_overlap", "0.25",
        "--synthetic_place_tries", "100",
        "--synthetic_item_retries", "16",
        "--synthetic_erode_px", "2",
        "--synthetic_sort_large_first",
        "--pad_to_canvas",
        "--canvas_w", "1024",
        "--canvas_h", "1024",
        "--cutout_dir", "data/raw/Shampoo_nobackground/Cropped_Library",
        "--synthetic_blade_mask_dir", TRAIN_BLADE_MASKS,
        "--synthetic_scale_min", f"{synthetic_scale_min:g}",
        "--synthetic_scale_max", f"{synthetic_scale_max:g}",
        "--synthetic_rot_min", f"{synthetic_rot_min:g}",
        "--synthetic_rot_max", f"{synthetic_rot_max:g}",
    ]


def fid_command(model_name: str, epoch: str, phase: str, max_images: int) -> list[str]:
    tray_masks = TRAIN_TRAY_MASKS if phase == "train" else TEST_TRAY_MASKS
    blade_masks = TRAIN_BLADE_MASKS if phase == "train" else TEST_BLADE_MASKS
    return [
        sys.executable,
        "Codes_Notebooks/Pix2Pix/fid_eval.py",
        "--dataroot", TEST_DATASET,
        "--name", model_name,
        "--epoch", epoch,
        "--phase", phase,
        "--work_dir", f"reports/dvc_test/fid_eval_runs/{phase}",
        "--max_images", str(max_images),
        "--keep_images",
        "--input_nc", "7",
        "--output_nc", "3",
        "--netG", "unet_256",
        "--netD", "n_layers",
        "--n_layers_D", "4",
        "--norm", "instance",
        "--class_nc", "3",
        "--preprocess", "none",
        "--load_size", "0",
        "--crop_size", "0",
        "--pad_to_canvas",
        "--canvas_w", "1024",
        "--canvas_h", "1024",
        "--use_thickness_channel",
        "--use_edge_channel",
        "--use_coord_channels",
        "--use_tray_mask",
        "--tray_mask_dir", tray_masks,
        "--synthetic_blade_mask_dir", blade_masks,
    ]


def run_control_command(label: str, cmd: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        with CONTROL_STATE.lock:
            CONTROL_STATE.append_log(f"[dashboard] ERROR starting {label}: {exc}")
        return 1
    with CONTROL_STATE.lock:
        CONTROL_STATE.step = label
        CONTROL_STATE.append_log(f"[dashboard] {label}")
        CONTROL_STATE.append_log("$ " + " ".join(cmd))
        CONTROL_STATE.process = process
    assert process.stdout is not None
    for line in process.stdout:
        with CONTROL_STATE.lock:
            CONTROL_STATE.append_log(line)
    return process.wait()


def run_logged_process(state: ControlState, label: str, cmd: list[str], cwd: Path = ROOT) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
    except Exception as exc:
        with state.lock:
            state.append_log(f"[dashboard] ERROR starting {label}: {exc}")
        return 1
    with state.lock:
        state.step = label
        state.append_log(f"[dashboard] {label}")
        state.append_log("$ " + " ".join(cmd))
        state.process = process
    assert process.stdout is not None
    for line in process.stdout:
        with state.lock:
            state.append_log(line)
    return process.wait()


def run_control_action(action: str, config: dict) -> None:
    with CONTROL_STATE.lock:
        CONTROL_STATE.started_at = datetime.now().isoformat(timespec="seconds")
        CONTROL_STATE.finished_at = None
        CONTROL_STATE.returncode = None
        CONTROL_STATE.action = action
        CONTROL_STATE.step = "starting"
        CONTROL_STATE.config = config
        CONTROL_STATE.logs = []
    code = 0
    try:
        if action == "train":
            mode = str(config.get("train_mode") or "bootstrap")
            if mode not in {"bootstrap", "continue"}:
                raise ValueError("Training mode must be bootstrap or continue.")
            source = safe_model_name(config.get("source_model") or DEFAULT_SOURCE_MODEL)
            output = safe_model_name(config.get("output_model"))
            epoch = safe_model_name(config.get("checkpoint_epoch") or "latest")
            more_epochs = safe_int(config.get("more_epochs", config.get("n_epochs")), 5, 1, 1000)
            n_epochs_decay = safe_int(config.get("n_epochs_decay"), 0, 0, 1000)
            max_dataset_size = safe_int(config.get("max_dataset_size"), 128, 1, 1000000)
            lr = safe_float(config.get("lr"), 3e-6, 1e-9, 1.0)
            beta1 = safe_float(config.get("beta1"), 0.5, 0.0, 0.999)
            save_epoch_freq = safe_int(config.get("save_epoch_freq"), 50, 1, 1000)
            synthetic_prob = safe_float(config.get("synthetic_prob"), 0.5, 0.0, 1.0)
            synthetic_scale_min = safe_float(config.get("synthetic_scale_min"), 0.60, 0.01, 10.0)
            synthetic_scale_max = safe_float(config.get("synthetic_scale_max"), 0.72, 0.01, 10.0)
            synthetic_rot_min = safe_float(config.get("synthetic_rot_min"), -8.0, -360.0, 360.0)
            synthetic_rot_max = safe_float(config.get("synthetic_rot_max"), 8.0, -360.0, 360.0)
            if "_DVC_TEST" not in output:
                raise ValueError("Output model must include _DVC_TEST to keep this dashboard in the test area.")
            checkpoint_model = source if mode == "bootstrap" else output
            if not model_exists(checkpoint_model):
                raise ValueError(f"Checkpoint model does not exist: {checkpoint_model}")
            if not checkpoint_exists(checkpoint_model, epoch):
                raise ValueError(f"Checkpoint does not exist: checkpoints/{checkpoint_model}/{epoch}_net_G.pth")
            base_epoch = numeric_epoch(epoch)
            if base_epoch is None and epoch == "latest":
                base_epoch = latest_numeric_epoch(checkpoint_model)
            base_epoch = base_epoch or 0
            epoch_count = base_epoch + 1
            n_epochs = base_epoch + more_epochs
            if mode == "bootstrap":
                bootstrap = [
                    sys.executable,
                    "src/xraygen/pipeline/bootstrap_pix2pix_checkpoint.py",
                    "--source-dir", f"checkpoints/{source}",
                    "--dest-dir", f"checkpoints/{output}",
                    "--epoch", epoch,
                    "--force",
                ]
                code = run_control_command("bootstrap checkpoint", bootstrap)
            if code == 0:
                with CONTROL_STATE.lock:
                    CONTROL_STATE.append_log(
                        "[dashboard] continuing "
                        f"{output} from {checkpoint_model}:{epoch}; "
                        f"epoch_count={epoch_count}, n_epochs={n_epochs}, n_epochs_decay={n_epochs_decay}"
                    )
                code = run_control_command(
                    "continue train generator",
                    train_command(
                        output,
                        epoch,
                        epoch_count,
                        n_epochs,
                        n_epochs_decay,
                        max_dataset_size,
                        lr,
                        beta1,
                        save_epoch_freq,
                        synthetic_prob,
                        synthetic_scale_min,
                        synthetic_scale_max,
                        synthetic_rot_min,
                        synthetic_rot_max,
                    ),
                )
        elif action == "evaluate_fid":
            model = safe_model_name(config.get("eval_model"))
            epoch = safe_model_name(config.get("eval_epoch") or "latest")
            phase = str(config.get("phase") or "test")
            if phase not in {"train", "test"}:
                raise ValueError("Phase must be train or test.")
            max_images = safe_int(config.get("max_images"), 50, 1, 100000)
            if not model_exists(model):
                raise ValueError(f"Checkpoint does not exist: {model}")
            code = run_control_command("evaluate FID", fid_command(model, epoch, phase, max_images))
        else:
            raise ValueError(f"Unknown action: {action}")
    except Exception as exc:
        code = 1
        with CONTROL_STATE.lock:
            CONTROL_STATE.append_log(f"[dashboard] ERROR: {exc}")
    finally:
        with CONTROL_STATE.lock:
            CONTROL_STATE.returncode = code
            CONTROL_STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            CONTROL_STATE.append_log(f"[dashboard] action exited with code {code}")
            CONTROL_STATE.process = None
            CONTROL_STATE.step = None


def safe_path_text(value: object, default: str) -> str:
    text = str(value or default).strip()
    if not text:
        text = default
    path = Path(text)
    resolved = path if path.is_absolute() else ROOT / path
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        raise ValueError("Paths must stay inside the project workspace.")
    return str(path)


def safe_choice(value: object, allowed: set[str], default: str) -> str:
    text = str(value or default).strip()
    return text if text in allowed else default


def fiftyone_command(action: str, config: dict) -> list[str]:
    dataset_name = safe_model_name(config.get("dataset_name") or f"xray_{action}_embeddings")
    model = str(config.get("embedding_model") or DEFAULT_FIFTYONE_EMBEDDING_MODEL).strip()
    method = safe_choice(config.get("viz_method"), {"umap", "tsne", "pca"}, "umap")
    port = safe_int(config.get("port"), DEFAULT_FIFTYONE_PORT, 1024, 65535)
    dup_threshold = safe_float(config.get("dup_threshold"), 0.2, 0.0, 1.0)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_fiftyone",
        "--mode", action,
        "--dataset-name", dataset_name,
        "--embedding-model", model,
        "--viz-method", method,
        "--dup-threshold", f"{dup_threshold:g}",
        "--port", str(port),
    ]
    if bool(config.get("crop_right_half")):
        cmd.append("--crop-right-half")
    if bool(config.get("grayscale")):
        cmd.append("--grayscale")
    if action == "pretrain":
        dataset_root = safe_path_text(config.get("dataset_root"), DEFAULT_FIFTYONE_DATASET)
        cmd.extend(["--dataset-root", dataset_root])
        for split in str(config.get("splits") or "train,test").split(","):
            split = split.strip()
            if split:
                cmd.extend(["--split", split])
    elif action == "posteval":
        eval_model = safe_model_name(config.get("eval_model") or DEFAULT_FIFTYONE_EVAL_MODEL)
        eval_epoch = safe_model_name(config.get("eval_epoch") or "latest")
        phase = safe_choice(config.get("phase"), {"train", "test"}, "test")
        max_images = safe_int(config.get("max_images"), 200, 1, 100000)
        default_root = f"reports/dvc_test/fid_eval_runs/{phase}/{eval_model}/epoch_{eval_epoch}"
        real_dir = safe_path_text(config.get("real_dir"), f"{default_root}/real")
        fake_dir = safe_path_text(config.get("fake_dir"), f"{default_root}/fake")
        cmd.extend([
            "--real-dir", real_dir,
            "--fake-dir", fake_dir,
            "--fid-model", eval_model,
            "--fid-epoch", eval_epoch,
            "--fid-phase", phase,
            "--fid-max-images", str(max_images),
        ])
        cmd.append("--grayscale")
    else:
        raise ValueError(f"Unknown FiftyOne action: {action}")
    return cmd


def run_fiftyone_action(action: str, config: dict) -> None:
    with FIFTYONE_STATE.lock:
        FIFTYONE_STATE.started_at = datetime.now().isoformat(timespec="seconds")
        FIFTYONE_STATE.finished_at = None
        FIFTYONE_STATE.returncode = None
        FIFTYONE_STATE.action = action
        FIFTYONE_STATE.step = "starting"
        FIFTYONE_STATE.config = config
        FIFTYONE_STATE.logs = []
    code = 0
    try:
        code = run_logged_process(FIFTYONE_STATE, f"launch FiftyOne {action}", fiftyone_command(action, config))
    except Exception as exc:
        code = 1
        with FIFTYONE_STATE.lock:
            FIFTYONE_STATE.append_log(f"[dashboard:fiftyone] ERROR: {exc}")
    finally:
        with FIFTYONE_STATE.lock:
            FIFTYONE_STATE.returncode = code
            FIFTYONE_STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            FIFTYONE_STATE.append_log(f"[dashboard:fiftyone] action exited with code {code}")
            FIFTYONE_STATE.process = None
            FIFTYONE_STATE.step = None


def stop_control_run() -> bool:
    with CONTROL_STATE.lock:
        proc = CONTROL_STATE.process
    if proc is None or proc.poll() is not None:
        return False
    proc.send_signal(signal.SIGTERM)
    return True


def stop_fiftyone_run() -> bool:
    with FIFTYONE_STATE.lock:
        proc = FIFTYONE_STATE.process
    if proc is None or proc.poll() is not None:
        return False
    proc.send_signal(signal.SIGTERM)
    return True


def collect_control_metrics() -> dict:
    config = CONTROL_STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    metrics_path = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}/metrics.json"
    metrics = read_json(metrics_path) or {}
    if not isinstance(metrics, dict):
        metrics = {}
    fid = extract_fid(metrics)
    return {
        "path": str(metrics_path.relative_to(ROOT)),
        "tiles": [
            {"label": "Selected Model", "value": model, "sub": "checkpoint"},
            {"label": "Phase", "value": phase, "sub": "FID split"},
            {"label": "Epoch", "value": epoch, "sub": "checkpoint epoch"},
            {"label": "FID", "value": format_float(fid), "sub": format_fid_sub(metrics)},
        ],
        "raw": metrics,
    }


def list_control_gallery_images(limit: int = 12) -> dict:
    config = CONTROL_STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    root = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}"
    return list_images_from_dirs({"fake": root / "fake", "real": root / "real", "debug": root / "debug"}, limit=limit)


def resolve_control_media_path(group: str, name: str) -> Path | None:
    config = CONTROL_STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    root = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}"
    return resolve_image_path({"fake": root / "fake", "real": root / "real", "debug": root / "debug"}, group, name)


def collect_fiftyone_metrics() -> dict:
    state = FIFTYONE_STATE.snapshot()
    config = state.get("config", {})
    mode = state.get("action") or "-"
    port = config.get("port") or DEFAULT_FIFTYONE_PORT
    if mode == "posteval":
        eval_model = config.get("eval_model") or DEFAULT_FIFTYONE_EVAL_MODEL
        eval_epoch = config.get("eval_epoch") or "latest"
        phase = config.get("phase") or "test"
        default_root = f"reports/dvc_test/fid_eval_runs/{phase}/{eval_model}/epoch_{eval_epoch}"
        real_dir = config.get("real_dir") or f"{default_root}/real"
        fake_dir = config.get("fake_dir") or f"{default_root}/fake"
        source_text = f"{real_dir} vs {fake_dir}"
    else:
        source_text = config.get("dataset_root") or DEFAULT_FIFTYONE_DATASET
    return {
        "tiles": [
            {"label": "Mode", "value": mode, "sub": "FiftyOne workflow"},
            {"label": "Dataset", "value": config.get("dataset_name", "-"), "sub": "FiftyOne name"},
            {"label": "Embedding", "value": config.get("embedding_model", DEFAULT_FIFTYONE_EMBEDDING_MODEL), "sub": "zoo model"},
            {"label": "App", "value": f"localhost:{port}", "sub": source_text},
        ],
    }


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS.union({".tif", ".tiff"})


def prepare_fiftyone_image(src: Path, cache_dir: Path, crop_right_half: bool, grayscale: bool) -> Path:
    if not crop_right_half and not grayscale:
        return src
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".png"
    flags = []
    if crop_right_half:
        flags.append("rightB")
    if grayscale:
        flags.append("gray")
    out = cache_dir / f"{src.stem}_{'_'.join(flags)}{suffix}"
    if out.exists():
        return out
    with Image.open(src) as img:
        img = img.convert("RGB")
        if crop_right_half:
            width, height = img.size
            img = img.crop((width // 2, 0, width, height))
        if grayscale:
            img = img.convert("L").convert("RGB")
        img.save(out)
    return out


def prepare_fiftyone_images(
    image_paths: list[Path],
    processed_dir: Path,
    crop_right_half: bool,
    grayscale: bool,
) -> list[Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared_paths = []
    for path in image_paths:
        prepared_paths.append(prepare_fiftyone_image(path, processed_dir, crop_right_half, grayscale))
    return prepared_paths


def split_debug_triplet(src: Path, cache_dir: Path) -> tuple[Path, Path]:
    from PIL import Image

    cache_dir.mkdir(parents=True, exist_ok=True)
    fake_out = cache_dir / f"{src.stem}_fake.png"
    real_out = cache_dir / f"{src.stem}_real.png"
    if fake_out.exists() and real_out.exists():
        return fake_out, real_out
    with Image.open(src) as img:
        img = img.convert("RGB")
        width, height = img.size
        third = width // 3
        fake = img.crop((third, 0, third * 2, height))
        real = img.crop((third * 2, 0, width, height))
        fake.save(fake_out)
        real.save(real_out)
    return fake_out, real_out


def iter_fiftyone_split_images(dataset_root: Path, splits: list[str]) -> list[dict]:
    rows: list[dict] = []
    for split in splits:
        folder = dataset_root / split
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if is_image_file(path):
                rows.append({"path": path, "split": split, "source": "pretrain", "pair_id": path.stem})
    return rows


def list_fiftyone_eval_images(real_dir: Path, fake_dir: Path, cache_dir: Path) -> tuple[list[Path], list[Path], str]:
    real_paths = [path for path in sorted(real_dir.rglob("*")) if is_image_file(path)] if real_dir.exists() else []
    fake_paths = [path for path in sorted(fake_dir.rglob("*")) if is_image_file(path)] if fake_dir.exists() else []
    if real_paths or fake_paths:
        return real_paths, fake_paths, "real_fake_dirs"

    debug_candidates = [
        real_dir.parent / "debug",
        fake_dir.parent / "debug",
        real_dir.parent,
        fake_dir.parent,
    ]
    seen: set[Path] = set()
    for debug_dir in debug_candidates:
        if not debug_dir.exists() or debug_dir in seen:
            continue
        seen.add(debug_dir)
        debug_images = [path for path in sorted(debug_dir.rglob("*")) if is_image_file(path)]
        if not debug_images:
            continue
        split_cache = cache_dir / "debug_split"
        real_paths = []
        fake_paths = []
        for path in debug_images:
            fake_path, real_path = split_debug_triplet(path, split_cache)
            real_paths.append(real_path)
            fake_paths.append(fake_path)
        return real_paths, fake_paths, "debug_triplet_split"
    return real_paths, fake_paths, "none"


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if is_image_file(path))


def ensure_fid_eval_images(args: argparse.Namespace, real_dir: Path, fake_dir: Path) -> None:
    if count_images(real_dir) > 0 and count_images(fake_dir) > 0:
        return
    if not args.fid_model:
        return

    print("Real/fake image folders are missing or empty.")
    print("Regenerating FID real/fake image folders with --keep_images before launching FiftyOne...")
    cmd = fid_command(args.fid_model, args.fid_epoch, args.fid_phase, args.fid_max_images)
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FID image regeneration failed with code {result.returncode}")


def launch_fiftyone_embeddings(args: argparse.Namespace) -> None:
    import fiftyone as fo
    import fiftyone.brain as fob
    import fiftyone.zoo as foz

    cache_dir = ROOT / "reports/fiftyone_cache" / args.dataset_name
    samples = []
    if args.mode == "pretrain":
        dataset_root = (ROOT / args.dataset_root).resolve() if not Path(args.dataset_root).is_absolute() else Path(args.dataset_root)
        rows = iter_fiftyone_split_images(dataset_root, args.split)
        if not rows:
            raise RuntimeError(
                "No images found for the requested FiftyOne workflow. "
                "Check the dataset root and split names."
            )
        for row in rows:
            path = row["path"]
            prepared = prepare_fiftyone_image(path, cache_dir / "pretrain_processed", args.crop_right_half, args.grayscale)
            samples.append(
                fo.Sample(
                    filepath=str(prepared.resolve()),
                    source=row["source"],
                    split=row["split"],
                    original_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                    stem=path.stem,
                )
            )
    else:
        real_dir = (ROOT / args.real_dir).resolve() if not Path(args.real_dir).is_absolute() else Path(args.real_dir)
        fake_dir = (ROOT / args.fake_dir).resolve() if not Path(args.fake_dir).is_absolute() else Path(args.fake_dir)
        ensure_fid_eval_images(args, real_dir, fake_dir)
        real_image_paths, fake_image_paths, image_source = list_fiftyone_eval_images(real_dir, fake_dir, cache_dir)
        processed_real_paths = prepare_fiftyone_images(
            real_image_paths,
            cache_dir / "real_B_processed",
            args.crop_right_half,
            args.grayscale,
        )
        processed_fake_paths = prepare_fiftyone_images(
            fake_image_paths,
            cache_dir / "fake_B_processed",
            args.crop_right_half,
            args.grayscale,
        )
        print(f"Prepared real images: {len(processed_real_paths)}")
        print(f"Prepared fake images: {len(processed_fake_paths)}")
        print(f"Image source: {image_source}")

        if len(processed_real_paths) == 0:
            raise RuntimeError(f"No real images found from: {real_dir}")
        if len(processed_fake_paths) == 0:
            raise RuntimeError(f"No generated images found from: {fake_dir}")

        for path in processed_real_paths:
            samples.append(
                fo.Sample(
                    filepath=str(path.resolve()),
                    source="real",
                    stem=path.stem,
                )
            )
        for path in processed_fake_paths:
            samples.append(
                fo.Sample(
                    filepath=str(path.resolve()),
                    source="generated",
                    stem=path.stem,
                )
            )

    if args.dataset_name in fo.list_datasets():
        fo.delete_dataset(args.dataset_name)
    dataset = fo.Dataset(args.dataset_name)
    dataset.add_samples(samples)
    dataset.compute_metadata()

    print(dataset)
    print(f"\nLoading embedding model: {args.embedding_model}")
    model = foz.load_zoo_model(args.embedding_model)
    if not model.has_embeddings:
        raise RuntimeError(f"Model does not expose embeddings: {args.embedding_model}")

    print("Computing embeddings...")
    dataset.compute_embeddings(model, embeddings_field="img_embeddings")

    print("\nComputing similarity...")
    sim_index = fob.compute_similarity(
        dataset,
        embeddings="img_embeddings",
        brain_key="img_similarity",
    )

    print(f"\nComputing visualization with method={args.viz_method}...")
    fob.compute_visualization(
        dataset,
        embeddings="img_embeddings",
        method=args.viz_method,
        num_dims=2,
        brain_key="img_viz",
    )

    print("\nComputing uniqueness...")
    fob.compute_uniqueness(dataset, uniqueness_field="uniqueness")

    print("\nComputing near-duplicates...")
    dups_index = fob.compute_near_duplicates(
        dataset,
        threshold=args.dup_threshold,
        similarity_index=sim_index,
    )

    print("\nDone.")
    print("Near-duplicate index computed:", dups_index is not None)
    if args.mode == "posteval":
        num_real = dataset.match(fo.ViewField("source") == "real").count()
        num_fake = dataset.match(fo.ViewField("source") == "generated").count()
        print(f"Real samples: {num_real}")
        print(f"Generated samples: {num_fake}")
    else:
        print("Samples:", dataset.count())

    print("\nBrain runs available:")
    print(dataset.list_brain_runs())

    color_scheme = None
    if args.mode == "posteval":
        color_scheme = fo.ColorScheme(
            fields=[
                {
                    "path": "source",
                    "valueColors": [
                        {"value": "real", "color": FIFTYONE_SOURCE_COLORS["real"]},
                        {"value": "generated", "color": FIFTYONE_SOURCE_COLORS["generated"]},
                    ],
                }
            ]
        )

    session = fo.launch_app(dataset, port=args.port, color_scheme=color_scheme)

    print("\nApp launched.")
    print(f"Open: http://localhost:{args.port}")
    print("In the app, open the Brain/Visualization result named 'img_viz'.")
    if args.mode == "posteval":
        print("Color by the 'source' field so real/generated use your chosen colors.")
    else:
        print("Color by the 'split' field to inspect train/test coverage.")
    session.wait()


def run_fiftyone_cli(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Build and launch FiftyOne embedding views for the X-ray pipeline.")
    ap.add_argument("--mode", choices=["pretrain", "posteval"], required=True)
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--embedding-model", default=DEFAULT_FIFTYONE_EMBEDDING_MODEL)
    ap.add_argument("--viz-method", choices=["umap", "tsne", "pca"], default="umap")
    ap.add_argument("--dup-threshold", type=float, default=0.2)
    ap.add_argument("--port", type=int, default=DEFAULT_FIFTYONE_PORT)
    ap.add_argument("--dataset-root", default=DEFAULT_FIFTYONE_DATASET)
    ap.add_argument("--split", action="append", default=[])
    ap.add_argument("--real-dir", default="")
    ap.add_argument("--fake-dir", default="")
    ap.add_argument("--fid-model", default="")
    ap.add_argument("--fid-epoch", default="latest")
    ap.add_argument("--fid-phase", choices=["train", "test"], default="test")
    ap.add_argument("--fid-max-images", type=int, default=200)
    ap.add_argument("--crop-right-half", action="store_true")
    ap.add_argument("--grayscale", action="store_true")
    args = ap.parse_args(argv)
    if not args.split:
        args.split = ["train", "test"]
    launch_fiftyone_embeddings(args)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>X-ray DVC Test + FiftyOne Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#111317; --panel:#1b2028; --field:#232a34; --text:#edf2f7; --muted:#9aa7b5; --line:#343d4b; --run:#61a8ff; --ok:#48c78e; --bad:#ff6b6b; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); letter-spacing:0; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 24px; border-bottom:1px solid var(--line); background:#151922; position:sticky; top:0; z-index:3; }
    h1 { font-size:20px; margin:0; }
    h2 { font-size:14px; margin:0 0 12px; }
    .sub { color:var(--muted); font-size:13px; margin-top:3px; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; }
    button { border:1px solid var(--line); background:var(--field); color:var(--text); border-radius:6px; padding:9px 12px; font:inherit; cursor:pointer; }
    button.primary, button.active { background:#245fa8; border-color:#3576c8; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    main { padding:18px 24px 24px; }
    .view { display:none; }
    .view.active { display:grid; grid-template-columns:minmax(300px, 430px) minmax(0, 1fr); gap:18px; }
    .section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; min-width:0; }
    .section + .section { margin-top:18px; }
    label { display:block; color:var(--muted); font-size:12px; margin:10px 0 5px; }
    input, select { width:100%; border:1px solid var(--line); border-radius:6px; background:var(--field); color:var(--text); padding:9px 10px; font:inherit; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
    .progress { height:8px; background:#0c0f14; border-radius:999px; overflow:hidden; margin-bottom:10px; }
    .bar { height:100%; width:0; background:var(--run); transition:width .25s ease; }
    .stage { display:grid; grid-template-columns:12px 1fr auto; align-items:center; gap:9px; padding:8px; border-radius:6px; }
    .stage.running { background:rgba(97,168,255,.11); }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--muted); }
    .done .dot, .skipped .dot { background:var(--ok); }
    .running .dot { background:var(--run); }
    .failed .dot { background:var(--bad); }
    .badge { color:var(--muted); border:1px solid var(--line); border-radius:999px; padding:3px 7px; font-size:11px; text-transform:uppercase; }
    .metrics { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }
    .tile { background:var(--field); border:1px solid var(--line); border-radius:8px; padding:11px; min-height:76px; min-width:0; }
    .tile-label, .tile-sub { color:var(--muted); font-size:12px; }
    .tile-value { font-size:19px; font-weight:750; margin-top:5px; overflow-wrap:anywhere; }
    .gallery-tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
    .gallery-tabs button.active { background:#245fa8; border-color:#3576c8; }
    .gallery { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }
    .thumb { background:var(--field); border:1px solid var(--line); border-radius:8px; overflow:hidden; min-width:0; }
    .thumb img { display:block; width:100%; aspect-ratio:1/1; object-fit:contain; background:#05070a; }
    .thumb-name { color:var(--muted); font-size:11px; padding:7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .status-line { color:var(--muted); border-top:1px solid var(--line); margin-top:12px; padding-top:10px; font-size:13px; }
    pre { margin:0; padding:12px 14px; height:calc(100vh - 500px); min-height:300px; overflow:auto; background:#07090d; color:#d9e2ee; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
    @media (max-width: 980px) { header, main { padding-left:14px; padding-right:14px; } .view.active { grid-template-columns:1fr; } .metrics, .gallery { grid-template-columns:repeat(2, minmax(120px, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>X-ray DVC Test + FiftyOne Dashboard</h1>
      <div class="sub" id="subtitle">Full pipeline, retrain/evaluate, and embedding inspection</div>
    </div>
    <div class="tabs">
      <button class="active" data-tab="pipelineView">Full Pipeline</button>
      <button data-tab="controlView">Retrain / Evaluate</button>
      <button data-tab="fiftyoneView">FiftyOne Embeddings</button>
    </div>
  </header>

  <main>
    <div class="view active" id="pipelineView">
      <aside>
        <div class="section">
          <h2>Pipeline Run</h2>
          <div class="actions">
            <button class="primary" id="pipelineStart">Start Test DVC</button>
            <button id="pipelineForce">Force Run</button>
            <button id="pipelineStop">Stop</button>
          </div>
          <div class="status-line" id="pipelineStatus">Waiting for run</div>
        </div>
        <div class="section">
          <h2>Stages</h2>
          <div class="progress"><div class="bar" id="pipelineBar"></div></div>
          <div id="pipelineStages"></div>
        </div>
      </aside>
      <section>
        <div class="section">
          <h2>Pipeline Metrics</h2>
          <div class="metrics" id="pipelineMetrics"></div>
        </div>
        <div class="section">
          <h2>Pipeline Images</h2>
          <div class="gallery-tabs" id="pipelineGalleryTabs"></div>
          <div class="gallery" id="pipelineGallery"></div>
        </div>
        <div class="section">
          <h2>Pipeline Logs</h2>
          <pre id="pipelineLogs"></pre>
        </div>
      </section>
    </div>

    <div class="view" id="controlView">
      <aside>
        <div class="section">
          <h2>Continue Training</h2>
          <label for="trainMode">Training mode</label>
          <select id="trainMode"><option value="continue">Continue existing test model</option><option value="bootstrap">Start from source checkpoint</option></select>
          <label for="sourceModel" id="sourceModelLabel">Source model</label>
          <select id="sourceModel"></select>
          <label for="checkpointEpoch">Checkpoint epoch</label>
          <select id="checkpointEpoch"></select>
          <label for="outputModel">Train into test model</label>
          <input id="outputModel" list="modelNames" value="Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST">
          <datalist id="modelNames"></datalist>
          <div class="row">
            <div><label for="moreEpochs">More fixed-LR epochs</label><input id="moreEpochs" type="number" min="1" value="5"></div>
            <div><label for="nEpochsDecay">More decay epochs</label><input id="nEpochsDecay" type="number" min="0" value="0"></div>
          </div>
          <div class="row">
            <div><label for="maxDatasetSize">Max dataset size</label><input id="maxDatasetSize" type="number" min="1" value="128"></div>
            <div><label for="saveEpochFreq">Save every N epochs</label><input id="saveEpochFreq" type="number" min="1" value="50"></div>
          </div>
          <div class="row">
            <div><label for="learningRate">Learning rate</label><input id="learningRate" type="number" min="0" step="0.000001" value="0.000003"></div>
            <div><label for="beta1">Adam beta1</label><input id="beta1" type="number" min="0" max="0.999" step="0.001" value="0.5"></div>
          </div>
          <div class="row">
            <div><label for="syntheticProb">Synthetic probability</label><input id="syntheticProb" type="number" min="0" max="1" step="0.05" value="0.5"></div>
            <div><label for="scaleMin">Scale min</label><input id="scaleMin" type="number" min="0.01" step="0.01" value="0.60"></div>
          </div>
          <div class="row">
            <div><label for="scaleMax">Scale max</label><input id="scaleMax" type="number" min="0.01" step="0.01" value="0.72"></div>
            <div><label for="rotMin">Rotation min</label><input id="rotMin" type="number" step="1" value="-8"></div>
          </div>
          <label for="rotMax">Rotation max</label>
          <input id="rotMax" type="number" step="1" value="8">
          <div class="actions"><button class="primary" id="trainBtn">Start Training</button><button id="controlStop">Stop</button></div>
        </div>
        <div class="section">
          <h2>Evaluate Model</h2>
          <label for="evalModel">Model to evaluate</label>
          <select id="evalModel"></select>
          <div class="row">
            <div><label for="evalEpoch">Eval epoch</label><select id="evalEpoch"></select></div>
            <div><label for="phase">Phase</label><select id="phase"><option>test</option><option>train</option></select></div>
          </div>
          <label for="maxImages">Max images</label>
          <input id="maxImages" type="number" min="1" value="50">
          <div class="actions"><button class="primary" id="evalBtn">Run FID Evaluation</button></div>
        </div>
      </aside>
      <section>
        <div class="section">
          <h2>Control Metrics</h2>
          <div class="metrics" id="controlMetrics"></div>
          <div class="status-line" id="controlStatus">Waiting for run</div>
        </div>
        <div class="section">
          <h2>FID Images</h2>
          <div class="gallery-tabs" id="controlGalleryTabs"></div>
          <div class="gallery" id="controlGallery"></div>
        </div>
        <div class="section">
          <h2>Control Logs</h2>
          <pre id="controlLogs"></pre>
        </div>
      </section>
    </div>

    <div class="view" id="fiftyoneView">
      <aside>
        <div class="section">
          <h2>Before Training Data</h2>
          <label for="foDatasetRoot">Pix2Pix dataset root</label>
          <input id="foDatasetRoot" value="datasets/_dvc_test/SHAMPOOBLADEWITHTRAY_COMPLETE">
          <label for="foSplits">Splits</label>
          <input id="foSplits" value="train,test">
          <label for="foPreDatasetName">FiftyOne dataset name</label>
          <input id="foPreDatasetName" value="xray_pretrain_embeddings">
          <div class="row">
            <div><label for="foCropPre">Crop right half</label><select id="foCropPre"><option value="true">Yes</option><option value="false">No</option></select></div>
            <div><label for="foGrayPre">Grayscale</label><select id="foGrayPre"><option value="true">Yes</option><option value="false">No</option></select></div>
          </div>
          <div class="actions"><button class="primary" id="foPreBtn">Open Pre-training Embeddings</button></div>
        </div>
        <div class="section">
          <h2>After Training Real Vs Fake</h2>
          <label for="foEvalModel">Evaluated model</label>
          <select id="foEvalModel"></select>
          <div class="row">
            <div><label for="foEvalEpoch">Epoch</label><select id="foEvalEpoch"></select></div>
            <div><label for="foPhase">Phase</label><select id="foPhase"><option>test</option><option>train</option></select></div>
          </div>
          <label for="foRealDir">Real image folder</label>
          <input id="foRealDir" placeholder="auto: reports/dvc_test/fid_eval_runs/.../real">
          <label for="foFakeDir">Generated image folder</label>
          <input id="foFakeDir" placeholder="auto: reports/dvc_test/fid_eval_runs/.../fake">
          <label for="foPostDatasetName">FiftyOne dataset name</label>
          <input id="foPostDatasetName" value="xray_posteval_embeddings">
          <label for="foMaxImages">Image pairs to generate if missing</label>
          <input id="foMaxImages" type="number" min="1" value="200">
          <div class="actions"><button class="primary" id="foPostBtn">Compare Real Vs Fake</button><button id="foStop">Stop FiftyOne</button></div>
        </div>
        <div class="section">
          <h2>Embedding Settings</h2>
          <label for="foEmbeddingModel">Embedding model</label>
          <input id="foEmbeddingModel" value="inception-v3-imagenet-torch">
          <div class="row">
            <div><label for="foVizMethod">Visualization</label><select id="foVizMethod"><option>umap</option><option>tsne</option><option>pca</option></select></div>
            <div><label for="foDupThreshold">Duplicate threshold</label><input id="foDupThreshold" type="number" min="0" max="1" step="0.01" value="0.2"></div>
          </div>
          <label for="foPort">FiftyOne port</label>
          <input id="foPort" type="number" min="1024" max="65535" value="5151">
        </div>
      </aside>
      <section>
        <div class="section">
          <h2>FiftyOne Status</h2>
          <div class="metrics" id="fiftyoneMetrics"></div>
          <div class="status-line" id="fiftyoneStatus">Waiting for FiftyOne run</div>
        </div>
        <div class="section">
          <h2>FiftyOne Logs</h2>
          <pre id="fiftyoneLogs"></pre>
        </div>
      </section>
    </div>
  </main>

  <script>
    const tabs = document.querySelectorAll('[data-tab]');
    tabs.forEach(btn => btn.onclick = () => {
      tabs.forEach(item => item.classList.toggle('active', item === btn));
      document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === btn.dataset.tab));
    });

    let models = [];
    let pipelineGalleryActive = 'accepted';
    let controlGalleryActive = 'fake';
    const sourceModel = document.getElementById('sourceModel');
    const evalModel = document.getElementById('evalModel');
    const checkpointEpoch = document.getElementById('checkpointEpoch');
    const evalEpoch = document.getElementById('evalEpoch');
    const trainMode = document.getElementById('trainMode');
    const outputModel = document.getElementById('outputModel');
    const pipelineLogs = document.getElementById('pipelineLogs');
    const controlLogs = document.getElementById('controlLogs');
    const fiftyoneLogs = document.getElementById('fiftyoneLogs');
    const foEvalModel = document.getElementById('foEvalModel');
    const foEvalEpoch = document.getElementById('foEvalEpoch');

    function epochsFor(name) {
      const item = models.find(m => m.name === name);
      return item ? item.epochs : ['latest'];
    }
    function fillEpochSelect(select, name) {
      const current = select.value || 'latest';
      select.innerHTML = epochsFor(name).map(e => `<option ${e === current ? 'selected' : ''}>${e}</option>`).join('');
      if (!select.value) select.value = 'latest';
    }
    function checkpointSourceName() {
      return trainMode.value === 'continue' ? outputModel.value : sourceModel.value;
    }
    function refreshCheckpointEpochs() {
      fillEpochSelect(checkpointEpoch, checkpointSourceName());
    }
    function updateTrainingModeUI() {
      const continuing = trainMode.value === 'continue';
      sourceModel.disabled = continuing;
      document.getElementById('sourceModelLabel').textContent = continuing ? 'Source model (not used)' : 'Source model';
      refreshCheckpointEpochs();
    }
    async function loadOptions() {
      const data = await fetch('/api/control/options').then(r => r.json());
      models = data.models || [];
      const names = models.map(m => m.name);
      const options = names.map(name => `<option>${name}</option>`).join('');
      sourceModel.innerHTML = options;
      evalModel.innerHTML = options;
      foEvalModel.innerHTML = options;
      document.getElementById('modelNames').innerHTML = options;
      sourceModel.value = names.includes(data.defaults.source_model) ? data.defaults.source_model : (names[0] || '');
      evalModel.value = names.includes(data.defaults.output_model) ? data.defaults.output_model : sourceModel.value;
      foEvalModel.value = evalModel.value;
      outputModel.value = data.defaults.output_model;
      updateTrainingModeUI();
      fillEpochSelect(evalEpoch, evalModel.value);
      fillEpochSelect(foEvalEpoch, foEvalModel.value);
    }
    trainMode.onchange = updateTrainingModeUI;
    sourceModel.onchange = refreshCheckpointEpochs;
    outputModel.onchange = refreshCheckpointEpochs;
    outputModel.oninput = refreshCheckpointEpochs;
    evalModel.onchange = () => fillEpochSelect(evalEpoch, evalModel.value);
    foEvalModel.onchange = () => fillEpochSelect(foEvalEpoch, foEvalModel.value);

    async function postJson(url, body={}) {
      await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    }
    document.getElementById('pipelineStart').onclick = () => postJson('/api/pipeline/start', {force:false});
    document.getElementById('pipelineForce').onclick = () => postJson('/api/pipeline/start', {force:true});
    document.getElementById('pipelineStop').onclick = () => postJson('/api/pipeline/stop');
    document.getElementById('controlStop').onclick = () => postJson('/api/control/stop');
    document.getElementById('trainBtn').onclick = () => postJson('/api/control/start_action', {action:'train', config:{
      train_mode: trainMode.value,
      source_model: sourceModel.value,
      checkpoint_epoch: checkpointEpoch.value,
      output_model: outputModel.value,
      more_epochs: document.getElementById('moreEpochs').value,
      n_epochs_decay: document.getElementById('nEpochsDecay').value,
      max_dataset_size: document.getElementById('maxDatasetSize').value,
      save_epoch_freq: document.getElementById('saveEpochFreq').value,
      lr: document.getElementById('learningRate').value,
      beta1: document.getElementById('beta1').value,
      synthetic_prob: document.getElementById('syntheticProb').value,
      synthetic_scale_min: document.getElementById('scaleMin').value,
      synthetic_scale_max: document.getElementById('scaleMax').value,
      synthetic_rot_min: document.getElementById('rotMin').value,
      synthetic_rot_max: document.getElementById('rotMax').value
    }});
    document.getElementById('evalBtn').onclick = () => postJson('/api/control/start_action', {action:'evaluate_fid', config:{
      eval_model: evalModel.value,
      eval_epoch: evalEpoch.value,
      phase: document.getElementById('phase').value,
      max_images: document.getElementById('maxImages').value
    }});
    function fiftyoneSharedConfig(datasetName) {
      return {
        dataset_name: datasetName,
        embedding_model: document.getElementById('foEmbeddingModel').value,
        viz_method: document.getElementById('foVizMethod').value,
        dup_threshold: document.getElementById('foDupThreshold').value,
        port: document.getElementById('foPort').value
      };
    }
    document.getElementById('foPreBtn').onclick = () => postJson('/api/fiftyone/start_action', {action:'pretrain', config:{
      ...fiftyoneSharedConfig(document.getElementById('foPreDatasetName').value),
      dataset_root: document.getElementById('foDatasetRoot').value,
      splits: document.getElementById('foSplits').value,
      crop_right_half: document.getElementById('foCropPre').value === 'true',
      grayscale: document.getElementById('foGrayPre').value === 'true'
    }});
    document.getElementById('foPostBtn').onclick = () => postJson('/api/fiftyone/start_action', {action:'posteval', config:{
      ...fiftyoneSharedConfig(document.getElementById('foPostDatasetName').value),
      eval_model: foEvalModel.value,
      eval_epoch: foEvalEpoch.value,
      phase: document.getElementById('foPhase').value,
      max_images: document.getElementById('foMaxImages').value,
      real_dir: document.getElementById('foRealDir').value,
      fake_dir: document.getElementById('foFakeDir').value
    }});
    document.getElementById('foStop').onclick = () => postJson('/api/fiftyone/stop');

    function galleryLabel(key) {
      return {generated:'Generated', accepted:'Accepted', rejected:'Rejected', real_ab_fake:'Real AB Fake', fake:'Fake', real:'Real', debug:'Debug'}[key] || key;
    }
    function renderGallery(images, prefix, mediaPrefix, activeValue, setActive) {
      const groups = Object.keys(images || {});
      if (!groups.includes(activeValue)) activeValue = groups[0] || '';
      document.getElementById(`${prefix}GalleryTabs`).innerHTML = groups.map(group => `<button class="${group === activeValue ? 'active' : ''}" data-${prefix}-gallery="${group}">${galleryLabel(group)} (${(images[group] || []).length})</button>`).join('');
      document.querySelectorAll(`[data-${prefix}-gallery]`).forEach(btn => btn.onclick = () => setActive(btn.dataset[`${prefix}Gallery`]));
      const selected = images[activeValue] || [];
      document.getElementById(`${prefix}Gallery`).innerHTML = selected.length ? selected.map(item => {
        const url = item.url.replace('/media?', `/${mediaPrefix}/media?`);
        return `<div class="thumb" title="${item.name}"><img src="${url}" alt="${item.name}" loading="lazy"><div class="thumb-name">${item.name}</div></div>`;
      }).join('') : '<div class="sub">No images yet</div>';
    }
    function renderPipeline(data) {
      const state = data.state;
      document.getElementById('pipelineStatus').textContent = `Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'}`;
      document.getElementById('pipelineBar').style.width = `${Math.round((state.progress || 0) * 100)}%`;
      document.getElementById('pipelineStart').disabled = state.running || data.any_running;
      document.getElementById('pipelineForce').disabled = state.running || data.any_running;
      document.getElementById('pipelineStop').disabled = !state.running;
      document.getElementById('pipelineStages').innerHTML = state.stages.map(s => `<div class="stage ${s.status}"><span class="dot"></span><span>${s.name}</span><span class="badge">${s.status}</span></div>`).join('');
      document.getElementById('pipelineMetrics').innerHTML = data.metrics.tiles.map(t => `<div class="tile"><div class="tile-label">${t.label}</div><div class="tile-value">${t.value}</div><div class="tile-sub">${t.sub}</div></div>`).join('');
      renderGallery(data.images || {}, 'pipeline', 'pipeline', pipelineGalleryActive, value => { pipelineGalleryActive = value; renderPipeline(data); });
      const atBottom = pipelineLogs.scrollTop + pipelineLogs.clientHeight >= pipelineLogs.scrollHeight - 24;
      pipelineLogs.textContent = state.logs.join('\n');
      if (atBottom) pipelineLogs.scrollTop = pipelineLogs.scrollHeight;
    }
    function renderControl(data) {
      const state = data.state;
      document.getElementById('controlStatus').textContent = `Status: ${state.status} · Action: ${state.action || '-'} · Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'}`;
      document.getElementById('trainBtn').disabled = state.running || data.any_running;
      document.getElementById('evalBtn').disabled = state.running || data.any_running;
      document.getElementById('controlStop').disabled = !state.running;
      document.getElementById('controlMetrics').innerHTML = data.metrics.tiles.map(t => `<div class="tile"><div class="tile-label">${t.label}</div><div class="tile-value">${t.value}</div><div class="tile-sub">${t.sub}</div></div>`).join('');
      renderGallery(data.images || {}, 'control', 'control', controlGalleryActive, value => { controlGalleryActive = value; renderControl(data); });
      const atBottom = controlLogs.scrollTop + controlLogs.clientHeight >= controlLogs.scrollHeight - 24;
      controlLogs.textContent = state.logs.join('\n');
      if (atBottom) controlLogs.scrollTop = controlLogs.scrollHeight;
    }
    function renderFiftyOne(data) {
      const state = data.state;
      document.getElementById('fiftyoneStatus').textContent = `Status: ${state.status} · Mode: ${state.action || '-'} · Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'}`;
      document.getElementById('foPreBtn').disabled = state.running;
      document.getElementById('foPostBtn').disabled = state.running;
      document.getElementById('foStop').disabled = !state.running;
      document.getElementById('fiftyoneMetrics').innerHTML = data.metrics.tiles.map(t => `<div class="tile"><div class="tile-label">${t.label}</div><div class="tile-value">${t.value}</div><div class="tile-sub">${t.sub}</div></div>`).join('');
      const atBottom = fiftyoneLogs.scrollTop + fiftyoneLogs.clientHeight >= fiftyoneLogs.scrollHeight - 24;
      fiftyoneLogs.textContent = state.logs.join('\n');
      if (atBottom) fiftyoneLogs.scrollTop = fiftyoneLogs.scrollHeight;
    }
    function render(data) {
      document.getElementById('subtitle').textContent = `Pipeline: ${data.pipeline.state.status} · Control: ${data.control.state.status} · FiftyOne: ${data.fiftyone.state.status}`;
      renderPipeline(data.pipeline);
      renderControl(data.control);
      renderFiftyOne(data.fiftyone);
    }
    loadOptions();
    const source = new EventSource('/events');
    source.onmessage = event => render(JSON.parse(event.data));
  </script>
</body>
</html>
"""


def pipeline_running() -> bool:
    with PIPELINE_STATE.lock:
        return PIPELINE_STATE.process is not None and PIPELINE_STATE.process.poll() is None


def control_running() -> bool:
    with CONTROL_STATE.lock:
        return CONTROL_STATE.process is not None and CONTROL_STATE.process.poll() is None


def fiftyone_running() -> bool:
    with FIFTYONE_STATE.lock:
        return FIFTYONE_STATE.process is not None and FIFTYONE_STATE.process.poll() is None


def snapshot() -> dict:
    any_running = pipeline_running() or control_running()
    return {
        "pipeline": {
            "state": PIPELINE_STATE.snapshot(),
            "metrics": collect_pipeline_metrics(),
            "images": list_pipeline_gallery_images(),
            "any_running": any_running and not pipeline_running(),
        },
        "control": {
            "state": CONTROL_STATE.snapshot(),
            "metrics": collect_control_metrics(),
            "images": list_control_gallery_images(),
            "any_running": any_running and not control_running(),
        },
        "fiftyone": {
            "state": FIFTYONE_STATE.snapshot(),
            "metrics": collect_fiftyone_metrics(),
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "XrayDvcTestUnifiedDashboard/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_media(self, media_path: Path | None) -> None:
        if media_path is None:
            self.send_json({"error": "image not found"}, status=404)
            return
        body = media_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(media_path))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/state":
            self.send_json(snapshot())
            return
        if path == "/api/control/options":
            self.send_json({
                "models": list_checkpoint_models(),
                "defaults": {
                    "source_model": DEFAULT_SOURCE_MODEL,
                    "output_model": DEFAULT_OUTPUT_MODEL,
                },
            })
            return
        if path == "/pipeline/media":
            query = parse_qs(urlparse(self.path).query)
            self.send_media(resolve_pipeline_media_path((query.get("group") or [""])[0], (query.get("name") or [""])[0]))
            return
        if path == "/control/media":
            query = parse_qs(urlparse(self.path).query)
            self.send_media(resolve_control_media_path((query.get("group") or [""])[0], (query.get("name") or [""])[0]))
            return
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(snapshot())}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if path == "/api/pipeline/start":
            if pipeline_running() or control_running():
                self.send_json({"started": False, "reason": "another run is already active"}, status=409)
                return
            dvc_cmd = resolve_dvc_command(payload.get("dvc_cmd"))
            target = payload.get("target") or None
            force = bool(payload.get("force", False))
            thread = threading.Thread(target=run_pipeline_dvc, args=(dvc_cmd, target, force), daemon=True)
            thread.start()
            self.send_json({"started": True, "dvc_cmd": dvc_cmd, "target": target, "force": force})
            return
        if path == "/api/pipeline/stop":
            self.send_json({"stopped": stop_pipeline_dvc()})
            return
        if path == "/api/control/start_action":
            if pipeline_running() or control_running():
                self.send_json({"started": False, "reason": "another run is already active"}, status=409)
                return
            action = str(payload.get("action") or "")
            config = payload.get("config") or {}
            thread = threading.Thread(target=run_control_action, args=(action, config), daemon=True)
            thread.start()
            self.send_json({"started": True, "action": action})
            return
        if path == "/api/control/stop":
            self.send_json({"stopped": stop_control_run()})
            return
        if path == "/api/fiftyone/start_action":
            if fiftyone_running():
                self.send_json({"started": False, "reason": "FiftyOne is already active"}, status=409)
                return
            action = str(payload.get("action") or "")
            config = payload.get("config") or {}
            thread = threading.Thread(target=run_fiftyone_action, args=(action, config), daemon=True)
            thread.start()
            self.send_json({"started": True, "action": action})
            return
        if path == "/api/fiftyone/stop":
            self.send_json({"stopped": stop_fiftyone_run()})
            return
        self.send_json({"error": "not found"}, status=404)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "_fiftyone":
        run_fiftyone_cli(sys.argv[2:])
        return

    ap = argparse.ArgumentParser(description="Run a unified dashboard for DVC test runs, manual retrain/evaluate loops, and FiftyOne embedding inspection.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8771)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard:test-fiftyone] open http://{args.host}:{args.port}")
    print(f"[dashboard:test-fiftyone] dvc cwd: {DVC_CWD}")
    print("[dashboard:test-fiftyone] use Full Pipeline, Retrain / Evaluate, or FiftyOne Embeddings")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
