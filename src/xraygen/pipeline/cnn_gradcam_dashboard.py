from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import os
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
EVAL_SCRIPT = ROOT / "Codes_Notebooks/Pix2Pix/evaluate_generated_cnn_gradcam.py"
DEFAULT_IMAGE_DIR = "datasets/_dvc_generated_combo/generated"
DEFAULT_OUT_DIR = "reports/generated_cnn_gradcam_dashboard"
DEFAULT_MODEL = (
    "models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2/"
    "gray_multihead_itemmask_optional_slow/checkpoints/train_best_checkpoint.pt"
)
MAX_LOG_LINES = 1200
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
SPATIAL_CLASSES = {"", "isolated", "overlap"}
THREAT_CLASSES = {"", "non_contraband", "contraband"}
GRADCAM_MODES = {"first", "contraband", "overlap_or_contraband", "interesting"}


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen[str] | None = None
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.returncode: int | None = None
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
                "config": self.config,
                "logs": self.logs[-400:],
            }


STATE = RunState()


def resolve_root_path(value: object, default: str) -> Path:
    text = str(value or default).strip()
    path = Path(text)
    if not path.is_absolute():
        path = ROOT / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_int(value: object, default: int, min_value: int, max_value: int) -> int:
    if value in (None, ""):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def safe_float(value: object, default: float, min_value: float, max_value: float) -> float:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(min_value, min(max_value, parsed))


def normalize_config(raw: dict) -> dict:
    gradcam_mode = str(raw.get("gradcam_mode") or "first")
    if gradcam_mode not in GRADCAM_MODES:
        gradcam_mode = "first"

    expected_spatial = str(raw.get("expected_spatial") or "")
    if expected_spatial not in SPATIAL_CLASSES:
        expected_spatial = ""

    expected_threat = str(raw.get("expected_threat") or "")
    if expected_threat not in THREAT_CLASSES:
        expected_threat = ""

    return {
        "image_dir": display_path(resolve_root_path(raw.get("image_dir"), DEFAULT_IMAGE_DIR)),
        "out_dir": display_path(resolve_root_path(raw.get("out_dir"), DEFAULT_OUT_DIR)),
        "model": display_path(resolve_root_path(raw.get("model"), DEFAULT_MODEL)),
        "recursive": bool(raw.get("recursive")),
        "max_images": safe_int(raw.get("max_images"), 0, 0, 1_000_000),
        "image_size": safe_int(raw.get("image_size"), 512, 64, 4096),
        "gradcam_mode": gradcam_mode,
        "gradcam_limit": safe_int(raw.get("gradcam_limit"), 32, 0, 100_000),
        "gradcam_alpha": safe_float(raw.get("gradcam_alpha"), 0.40, 0.0, 1.0),
        "expected_spatial": expected_spatial,
        "expected_threat": expected_threat,
        "clean_border": bool(raw.get("clean_border", True)),
        "crop_border": bool(raw.get("crop_border", False)),
        "crop_margin": safe_int(raw.get("crop_margin"), 0, 0, 256),
    }


def build_command(config: dict) -> list[str]:
    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--image_dir",
        config["image_dir"],
        "--out_dir",
        config["out_dir"],
        "--model_mode",
        "multihead_itemmask",
        "--model",
        config["model"],
        "--image_size",
        str(config["image_size"]),
        "--input_channels",
        "2",
        "--gradcam_mode",
        config["gradcam_mode"],
        "--gradcam_limit",
        str(config["gradcam_limit"]),
        "--gradcam_alpha",
        str(config["gradcam_alpha"]),
        "--crop_margin",
        str(config["crop_margin"]),
    ]
    if config["recursive"]:
        cmd.append("--recursive")
    if config["max_images"] > 0:
        cmd.extend(["--max_images", str(config["max_images"])])
    if not config["clean_border"]:
        cmd.append("--no_clean_border")
    if not config["crop_border"]:
        cmd.append("--no_crop_border")
    if config["expected_spatial"]:
        cmd.extend(["--expected_spatial", config["expected_spatial"]])
    if config["expected_threat"]:
        cmd.extend(["--expected_threat", config["expected_threat"]])
    return cmd


def run_eval(config: dict) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cmd = build_command(config)

    with STATE.lock:
        STATE.started_at = datetime.now().isoformat(timespec="seconds")
        STATE.finished_at = None
        STATE.returncode = None
        STATE.config = config
        STATE.logs = ["$ " + " ".join(cmd)]
        STATE.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    assert STATE.process.stdout is not None
    for line in STATE.process.stdout:
        with STATE.lock:
            STATE.append_log(line)

    returncode = STATE.process.wait()
    with STATE.lock:
        STATE.returncode = returncode
        STATE.finished_at = datetime.now().isoformat(timespec="seconds")
        STATE.append_log(f"[dashboard] evaluator exited with code {returncode}")
        STATE.process = None


def stop_run() -> bool:
    with STATE.lock:
        proc = STATE.process
    if proc is None or proc.poll() is not None:
        return False
    proc.send_signal(signal.SIGTERM)
    return True


def read_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text())
    except Exception as exc:
        return {"error": str(exc)}
    return None


def read_prediction_rows(path: Path, limit: int | None = 40) -> list[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    if limit is None:
        return rows
    return rows[:limit]


def is_allowed_media_path(path: Path, allowed_root: Path) -> bool:
    try:
        resolved = path.resolve()
        root = allowed_root.resolve()
    except FileNotFoundError:
        return False
    return resolved == root or root in resolved.parents


def path_from_prediction_value(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "-"


def current_out_dir() -> Path:
    config = STATE.snapshot().get("config") or {}
    return resolve_root_path(config.get("out_dir"), DEFAULT_OUT_DIR)


def collect_metrics() -> dict:
    out_dir = current_out_dir()
    summary_path = out_dir / "summary.json"
    predictions_path = out_dir / "predictions.csv"
    summary = read_json(summary_path) or {}
    rows = read_prediction_rows(predictions_path)
    spatial_counts = summary.get("spatial_class_counts") or {}
    threat_counts = summary.get("threat_class_counts") or {}
    return {
        "summary_path": display_path(summary_path),
        "predictions_path": display_path(predictions_path),
        "raw": summary,
        "rows": rows,
        "tiles": [
            {"label": "Images", "value": summary.get("num_images", "-"), "sub": "evaluated"},
            {"label": "Spatial Overlap", "value": spatial_counts.get("overlap", 0), "sub": "predicted count"},
            {"label": "Threat", "value": threat_counts.get("contraband", 0), "sub": "contraband count"},
            {"label": "Mean Overlap", "value": format_float(summary.get("mean_spatial_prob_overlap")), "sub": "probability"},
            {"label": "Mean Threat", "value": format_float(summary.get("mean_threat_prob_contraband")), "sub": "probability"},
            {"label": "Spatial Match", "value": format_percent(summary.get("spatial_expected_match_rate")), "sub": summary.get("expected_spatial", "optional target")},
            {"label": "Threat Match", "value": format_percent(summary.get("threat_expected_match_rate")), "sub": summary.get("expected_threat", "optional target")},
            {"label": "Grad-CAM", "value": summary.get("gradcam_limit", "-"), "sub": summary.get("gradcam_mode", "mode")},
        ],
    }


def collect_review_items(limit: int | None = None) -> list[dict]:
    out_dir = current_out_dir()
    predictions_path = out_dir / "predictions.csv"
    rows = read_prediction_rows(predictions_path, limit=limit)
    items = []
    for idx, row in enumerate(rows):
        items.append(
            {
                "index": idx,
                "image": row.get("image", ""),
                "original_url": f"/review_media?kind=original&idx={idx}",
                "spatial_url": f"/review_media?kind=spatial&idx={idx}",
                "threat_url": f"/review_media?kind=threat&idx={idx}",
                "spatial_pred": row.get("spatial_pred", ""),
                "spatial_prob_overlap": row.get("spatial_prob_overlap", ""),
                "threat_pred": row.get("threat_pred", ""),
                "threat_prob_contraband": row.get("threat_prob_contraband", ""),
                "has_spatial_gradcam": bool(row.get("spatial_gradcam")),
                "has_threat_gradcam": bool(row.get("threat_gradcam")),
            }
        )
    return items


def list_gallery_images(limit: int = 24) -> dict:
    gradcam_root = current_out_dir() / "gradcam"
    folders = {
        "spatial": gradcam_root / "spatial_overlap_isolated",
        "threat": gradcam_root / "threat_contraband_noncontraband",
    }
    galleries = {}
    for group, folder in folders.items():
        items = []
        if folder.exists():
            paths = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
            for path in paths[:limit]:
                items.append({"name": path.name, "url": f"/media?group={quote(group)}&name={quote(path.name)}"})
        galleries[group] = items
    return galleries


def resolve_review_media_path(kind: str, idx_text: str) -> Path | None:
    try:
        idx = int(idx_text)
    except (TypeError, ValueError):
        return None

    out_dir = current_out_dir()
    rows = read_prediction_rows(out_dir / "predictions.csv", limit=max(idx + 1, 1))
    if idx < 0 or idx >= len(rows):
        return None
    row = rows[idx]

    if kind == "original":
        value = row.get("image") or ""
        allowed_root = ROOT
    elif kind == "spatial":
        value = row.get("spatial_gradcam") or ""
        allowed_root = out_dir
    elif kind == "threat":
        value = row.get("threat_gradcam") or ""
        allowed_root = out_dir
    else:
        return None

    if not value:
        return None
    path = path_from_prediction_value(value)
    if not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
        return None
    if not is_allowed_media_path(path, allowed_root):
        return None
    return path


def resolve_media_path(group: str, name: str) -> Path | None:
    gradcam_root = current_out_dir() / "gradcam"
    folder = {
        "spatial": gradcam_root / "spatial_overlap_isolated",
        "threat": gradcam_root / "threat_contraband_noncontraband",
    }.get(group)
    if folder is None:
        return None
    path = folder / Path(name).name
    try:
        resolved = path.resolve()
        folder_resolved = folder.resolve()
    except FileNotFoundError:
        return None
    if folder_resolved not in resolved.parents or not resolved.exists() or resolved.suffix.lower() not in IMAGE_EXTS:
        return None
    return resolved


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>X-ray CNN Grad-CAM Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#111317; --panel:#1b2028; --field:#232a34; --text:#edf2f7; --muted:#9aa7b5; --line:#343d4b; --run:#61a8ff; --ok:#48c78e; --bad:#ff6b6b; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); letter-spacing:0; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 24px; border-bottom:1px solid var(--line); background:#151922; position:sticky; top:0; z-index:2; }
    h1 { font-size:20px; margin:0; }
    h2 { font-size:14px; margin:0 0 12px; }
    .sub { color:var(--muted); font-size:13px; margin-top:3px; }
    main { display:grid; grid-template-columns:minmax(340px, 460px) minmax(0, 1fr); gap:18px; padding:18px 24px 24px; }
    .section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    .section + .section { margin-top:18px; }
    label { display:block; color:var(--muted); font-size:12px; margin:10px 0 5px; }
    input, select { width:100%; border:1px solid var(--line); border-radius:6px; background:var(--field); color:var(--text); padding:9px 10px; font:inherit; }
    input[type="checkbox"] { width:auto; margin-right:7px; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { border:1px solid var(--line); background:var(--field); color:var(--text); border-radius:6px; padding:9px 12px; font:inherit; cursor:pointer; }
    button.primary { background:#245fa8; border-color:#3576c8; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
    .metrics { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }
    .tile { background:var(--field); border:1px solid var(--line); border-radius:8px; padding:11px; min-height:76px; min-width:0; }
    .tile-label, .tile-sub { color:var(--muted); font-size:12px; }
    .tile-value { font-size:20px; font-weight:750; margin-top:5px; overflow-wrap:anywhere; }
    .status-line { color:var(--muted); border-top:1px solid var(--line); margin-top:12px; padding-top:10px; font-size:13px; overflow-wrap:anywhere; }
    .review-grid { display:grid; grid-template-columns:1fr; gap:12px; }
    .review-card { background:var(--field); border:1px solid var(--line); border-radius:8px; overflow:hidden; min-width:0; }
    .review-head { display:grid; grid-template-columns:1fr auto auto; gap:10px; align-items:center; padding:10px 11px; border-bottom:1px solid var(--line); }
    .review-name { font-size:12px; color:var(--muted); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .pill { border:1px solid var(--line); border-radius:999px; padding:4px 8px; font-size:12px; white-space:nowrap; }
    .pill.overlap, .pill.contraband { color:var(--bad); border-color:rgba(255, 107, 107, .55); }
    .pill.isolated, .pill.non_contraband { color:var(--ok); border-color:rgba(72, 199, 142, .45); }
    .review-images { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:1px; background:var(--line); }
    .review-shot { background:#05070a; min-width:0; }
    .review-shot img { display:block; width:100%; aspect-ratio:1/1; object-fit:contain; }
    .shot-label { color:var(--muted); font-size:11px; padding:7px 8px; background:#11151c; border-top:1px solid var(--line); }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    th, td { border-top:1px solid var(--line); padding:7px 6px; text-align:left; overflow-wrap:anywhere; }
    th { color:var(--muted); font-weight:600; }
    pre { margin:0; padding:12px 14px; height:calc(100vh - 560px); min-height:300px; overflow:auto; background:#07090d; color:#d9e2ee; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
    @media (max-width: 980px) { header, main { padding-left:14px; padding-right:14px; } main { grid-template-columns:1fr; } .metrics { grid-template-columns:repeat(2, minmax(120px, 1fr)); } .review-head { grid-template-columns:1fr; } .review-images { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>X-ray CNN Grad-CAM Dashboard</h1>
      <div class="sub" id="subtitle">Idle</div>
    </div>
    <button id="stopBtn">Stop</button>
  </header>
  <main>
    <aside>
      <div class="section">
        <h2>Evaluation Settings</h2>
        <label for="imageDir">Generated image directory</label>
        <input id="imageDir" value="datasets/_dvc_generated_combo/generated">
        <label for="outDir">Output directory</label>
        <input id="outDir" value="reports/generated_cnn_gradcam_dashboard">
        <label for="modelPath">SimpleCNN multi-head model</label>
        <input id="modelPath" value="models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2/gray_multihead_itemmask_optional_slow/checkpoints/train_best_checkpoint.pt">
        <div class="row">
          <div><label for="maxImages">Max images</label><input id="maxImages" type="number" min="0" value="0"></div>
          <div><label for="imageSize">Image size</label><input id="imageSize" type="number" min="64" value="512"></div>
        </div>
        <div class="row">
          <div><label for="gradcamMode">Grad-CAM mode</label><select id="gradcamMode"><option>first</option><option>contraband</option><option>overlap_or_contraband</option><option>interesting</option></select></div>
          <div><label for="gradcamLimit">Grad-CAM limit</label><input id="gradcamLimit" type="number" min="0" value="32"></div>
        </div>
        <div class="row">
          <div><label for="expectedSpatial">Expected spatial</label><select id="expectedSpatial"><option value="">none</option><option>isolated</option><option>overlap</option></select></div>
          <div><label for="expectedThreat">Expected threat</label><select id="expectedThreat"><option value="">none</option><option>non_contraband</option><option>contraband</option></select></div>
        </div>
        <div class="row">
          <div><label for="gradcamAlpha">Overlay alpha</label><input id="gradcamAlpha" type="number" min="0" max="1" step="0.05" value="0.40"></div>
          <div><label for="cropMargin">Crop margin</label><input id="cropMargin" type="number" min="0" value="0"></div>
        </div>
        <label><input id="recursive" type="checkbox"> Scan recursively</label>
        <label><input id="cleanBorder" type="checkbox" checked> Remove large black bands</label>
        <label><input id="cropBorder" type="checkbox"> Crop white border, two-model mode only</label>
        <div class="actions"><button class="primary" id="startBtn">Run CNN + Grad-CAM</button></div>
      </div>
    </aside>
    <section>
      <div class="section">
        <h2>Current Metrics</h2>
        <div class="metrics" id="metrics"></div>
        <div class="status-line" id="statusLine">Waiting for run</div>
      </div>
      <div class="section">
        <h2>Generated Image Review</h2>
        <div id="review" class="review-grid"></div>
      </div>
      <div class="section">
        <h2>Predictions</h2>
        <div id="predictions"></div>
      </div>
      <div class="section">
        <h2>Log Stream</h2>
        <pre id="logs"></pre>
      </div>
    </section>
  </main>
  <script>
    const logs = document.getElementById('logs');

    function config() {
      return {
        image_dir: document.getElementById('imageDir').value,
        out_dir: document.getElementById('outDir').value,
        model: document.getElementById('modelPath').value,
        recursive: document.getElementById('recursive').checked,
        max_images: document.getElementById('maxImages').value,
        image_size: document.getElementById('imageSize').value,
        gradcam_mode: document.getElementById('gradcamMode').value,
        gradcam_limit: document.getElementById('gradcamLimit').value,
        gradcam_alpha: document.getElementById('gradcamAlpha').value,
        expected_spatial: document.getElementById('expectedSpatial').value,
        expected_threat: document.getElementById('expectedThreat').value,
        clean_border: document.getElementById('cleanBorder').checked,
        crop_border: document.getElementById('cropBorder').checked,
        crop_margin: document.getElementById('cropMargin').value
      };
    }
    document.getElementById('startBtn').onclick = async () => {
      await fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({config: config()})});
    };
    document.getElementById('stopBtn').onclick = () => fetch('/api/stop', {method:'POST'});

    function prob(value) {
      const n = Number(value || 0);
      return Number.isFinite(n) ? n.toFixed(3) : '-';
    }
    function renderReview(items) {
      if (!items || !items.length) {
        document.getElementById('review').innerHTML = '<div class="sub">No reviewed images yet</div>';
        return;
      }
      document.getElementById('review').innerHTML = items.map(item => `
        <div class="review-card">
          <div class="review-head">
            <div class="review-name" title="${item.image}">${item.image}</div>
            <div class="pill ${item.spatial_pred}">Spatial: ${item.spatial_pred || '-'} · P overlap ${prob(item.spatial_prob_overlap)}</div>
            <div class="pill ${item.threat_pred}">Threat: ${item.threat_pred || '-'} · P contraband ${prob(item.threat_prob_contraband)}</div>
          </div>
          <div class="review-images">
            <div class="review-shot">
              <img src="${item.original_url}" alt="generated image" loading="lazy">
              <div class="shot-label">Generated image</div>
            </div>
            <div class="review-shot">
              ${item.has_spatial_gradcam ? `<img src="${item.spatial_url}" alt="spatial gradcam" loading="lazy">` : '<div class="sub" style="padding:20px">No spatial Grad-CAM</div>'}
              <div class="shot-label">Spatial Grad-CAM</div>
            </div>
            <div class="review-shot">
              ${item.has_threat_gradcam ? `<img src="${item.threat_url}" alt="threat gradcam" loading="lazy">` : '<div class="sub" style="padding:20px">No threat Grad-CAM</div>'}
              <div class="shot-label">Threat Grad-CAM</div>
            </div>
          </div>
        </div>
      `).join('');
    }
    function renderPredictions(rows) {
      if (!rows || !rows.length) {
        document.getElementById('predictions').innerHTML = '<div class="sub">No predictions yet</div>';
        return;
      }
      document.getElementById('predictions').innerHTML = `<table><thead><tr><th>Image</th><th>Spatial</th><th>P overlap</th><th>Threat</th><th>P contraband</th></tr></thead><tbody>${rows.map(r => `<tr><td>${r.image}</td><td>${r.spatial_pred}</td><td>${Number(r.spatial_prob_overlap || 0).toFixed(3)}</td><td>${r.threat_pred}</td><td>${Number(r.threat_prob_contraband || 0).toFixed(3)}</td></tr>`).join('')}</tbody></table>`;
    }
    function render(data) {
      const state = data.state;
      document.getElementById('subtitle').textContent = state.status.toUpperCase();
      document.getElementById('startBtn').disabled = state.running;
      document.getElementById('stopBtn').disabled = !state.running;
      document.getElementById('statusLine').textContent = `Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'} · Summary: ${data.metrics.summary_path}`;
      document.getElementById('metrics').innerHTML = data.metrics.tiles.map(t => `<div class="tile"><div class="tile-label">${t.label}</div><div class="tile-value">${t.value}</div><div class="tile-sub">${t.sub}</div></div>`).join('');
      renderReview(data.review || []);
      renderPredictions(data.metrics.rows || []);
      const atBottom = logs.scrollTop + logs.clientHeight >= logs.scrollHeight - 24;
      logs.textContent = state.logs.join('\n');
      if (atBottom) logs.scrollTop = logs.scrollHeight;
    }
    const source = new EventSource('/events');
    source.onmessage = event => render(JSON.parse(event.data));
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "XrayCnnGradcamDashboard/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def event_payload(self) -> dict:
        return {
            "state": STATE.snapshot(),
            "metrics": collect_metrics(),
            "images": list_gallery_images(),
            "review": collect_review_items(),
        }

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
            self.send_json(self.event_payload())
            return
        if path == "/media":
            query = parse_qs(urlparse(self.path).query)
            media_path = resolve_media_path((query.get("group") or [""])[0], (query.get("name") or [""])[0])
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
            return
        if path == "/review_media":
            query = parse_qs(urlparse(self.path).query)
            media_path = resolve_review_media_path((query.get("kind") or [""])[0], (query.get("idx") or [""])[0])
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
            return
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(f"data: {json.dumps(self.event_payload())}\n\n".encode("utf-8"))
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
        if path == "/api/start":
            with STATE.lock:
                running = STATE.process is not None and STATE.process.poll() is None
            if running:
                self.send_json({"started": False, "reason": "already running"}, status=409)
                return
            config = normalize_config(payload.get("config") or {})
            thread = threading.Thread(target=run_eval, args=(config,), daemon=True)
            thread.start()
            self.send_json({"started": True, "config": config})
            return
        if path == "/api/stop":
            self.send_json({"stopped": stop_run()})
            return
        self.send_json({"error": "not found"}, status=404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a local dashboard for CNN classifier and Grad-CAM evaluation.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8750)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard:cnn-gradcam] open http://{args.host}:{args.port}")
    print("[dashboard:cnn-gradcam] choose generated image and Grad-CAM settings in the browser")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
