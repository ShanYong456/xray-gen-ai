from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xraygen.pipeline import dvc_live_dashboard as dashboard


ROOT = dashboard.ROOT
CHECKPOINTS_DIR = ROOT / "checkpoints"
TEST_DATASET = "datasets/_dvc_test/SHAMPOOBLADEWITHTRAY_COMPLETE"
TRAIN_TRAY_MASKS = f"{TEST_DATASET}/matched_masks/train/tray"
TEST_TRAY_MASKS = f"{TEST_DATASET}/matched_masks/test/tray"
TRAIN_BLADE_MASKS = f"{TEST_DATASET}/matched_masks/train/blade"
TEST_BLADE_MASKS = f"{TEST_DATASET}/matched_masks/test/blade"
DEFAULT_SOURCE_MODEL = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_COMPLETESyn"
DEFAULT_OUTPUT_MODEL = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST"
MAX_LOG_LINES = 1200
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class RunState:
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


STATE = RunState()


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


def list_checkpoint_models() -> list[dict]:
    models = []
    if not CHECKPOINTS_DIR.exists():
        return models
    for folder in sorted(p for p in CHECKPOINTS_DIR.iterdir() if p.is_dir()):
        checkpoints = sorted(folder.glob("*_net_G.pth"))
        if not checkpoints:
            continue
        epochs = sorted(
            {p.name.removesuffix("_net_G.pth") for p in checkpoints},
            key=lambda item: (item != "latest", item),
        )
        models.append({"name": folder.name, "epochs": epochs})
    return models


def model_exists(name: str) -> bool:
    return any(item["name"] == name for item in list_checkpoint_models())


def train_command(model_name: str, n_epochs: int, n_epochs_decay: int, max_dataset_size: int) -> list[str]:
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
        "--lr", "3e-6",
        "--beta1", "0.5",
        "--n_epochs", str(n_epochs),
        "--n_epochs_decay", str(n_epochs_decay),
        "--max_dataset_size", str(max_dataset_size),
        "--save_latest_freq", "100000000",
        "--save_epoch_freq", "50",
        "--continue_train",
        "--epoch", "latest",
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
        "--synthetic_prob", "0.5",
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
        "--synthetic_scale_min", "0.60",
        "--synthetic_scale_max", "0.72",
        "--synthetic_rot_min", "-8",
        "--synthetic_rot_max", "8",
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


def run_command(label: str, cmd: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with STATE.lock:
        STATE.step = label
        STATE.append_log(f"[dashboard] {label}")
        STATE.append_log("$ " + " ".join(cmd))
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
    return STATE.process.wait()


def run_action(action: str, config: dict) -> None:
    with STATE.lock:
        STATE.started_at = datetime.now().isoformat(timespec="seconds")
        STATE.finished_at = None
        STATE.returncode = None
        STATE.action = action
        STATE.step = "starting"
        STATE.config = config
        STATE.logs = []

    code = 0
    try:
        if action == "train":
            source = safe_model_name(config.get("source_model"))
            output = safe_model_name(config.get("output_model"))
            epoch = safe_model_name(config.get("checkpoint_epoch") or "latest")
            n_epochs = safe_int(config.get("n_epochs"), 5, 1, 1000)
            n_epochs_decay = safe_int(config.get("n_epochs_decay"), 0, 0, 1000)
            max_dataset_size = safe_int(config.get("max_dataset_size"), 128, 1, 1000000)
            if "_DVC_TEST" not in output:
                raise ValueError("Output model must include _DVC_TEST to keep this dashboard in the test area.")
            if not model_exists(source):
                raise ValueError(f"Source checkpoint does not exist: {source}")
            bootstrap = [
                sys.executable,
                "src/xraygen/pipeline/bootstrap_pix2pix_checkpoint.py",
                "--source-dir", f"checkpoints/{source}",
                "--dest-dir", f"checkpoints/{output}",
                "--epoch", epoch,
                "--force",
            ]
            code = run_command("bootstrap checkpoint", bootstrap)
            if code == 0:
                code = run_command("continue train generator", train_command(output, n_epochs, n_epochs_decay, max_dataset_size))
        elif action == "evaluate_fid":
            model = safe_model_name(config.get("eval_model"))
            epoch = safe_model_name(config.get("eval_epoch") or "latest")
            phase = str(config.get("phase") or "test")
            if phase not in {"train", "test"}:
                raise ValueError("Phase must be train or test.")
            max_images = safe_int(config.get("max_images"), 50, 1, 100000)
            if not model_exists(model):
                raise ValueError(f"Checkpoint does not exist: {model}")
            code = run_command("evaluate FID", fid_command(model, epoch, phase, max_images))
        else:
            raise ValueError(f"Unknown action: {action}")
    except Exception as exc:
        code = 1
        with STATE.lock:
            STATE.append_log(f"[dashboard] ERROR: {exc}")
    finally:
        with STATE.lock:
            STATE.returncode = code
            STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            STATE.append_log(f"[dashboard] action exited with code {code}")
            STATE.process = None
            STATE.step = None


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


def collect_metrics_for_state() -> dict:
    config = STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    metrics_path = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}/metrics.json"
    metrics = read_json(metrics_path) or {}
    fid = dashboard.extract_fid(metrics) if isinstance(metrics, dict) else None
    return {
        "path": str(metrics_path.relative_to(ROOT)),
        "tiles": [
            {"label": "Selected Model", "value": model, "sub": "checkpoint"},
            {"label": "Phase", "value": phase, "sub": "FID split"},
            {"label": "Epoch", "value": epoch, "sub": "checkpoint epoch"},
            {"label": "FID", "value": dashboard.format_float(fid), "sub": dashboard.format_fid_sub(metrics if isinstance(metrics, dict) else {})},
        ],
        "raw": metrics,
    }


def list_gallery_images(limit: int = 12) -> dict:
    config = STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    root = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}"
    folders = {
        "fake": root / "fake",
        "real": root / "real",
        "debug": root / "debug",
    }
    galleries = {}
    for group, folder in folders.items():
        items = []
        if folder.exists():
            for path in sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)[:limit]:
                items.append({"name": path.name, "url": f"/media?group={quote(group)}&name={quote(path.name)}"})
        galleries[group] = items
    return galleries


def resolve_media_path(group: str, name: str) -> Path | None:
    config = STATE.snapshot().get("config", {})
    model = config.get("eval_model") or config.get("output_model") or DEFAULT_OUTPUT_MODEL
    phase = config.get("phase") or "test"
    epoch = config.get("eval_epoch") or "latest"
    root = ROOT / f"reports/dvc_test/fid_eval_runs/{phase}/{model}/epoch_{epoch}"
    folder = {"fake": root / "fake", "real": root / "real", "debug": root / "debug"}.get(group)
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
  <title>X-ray DVC Test Control Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#111317; --panel:#1b2028; --field:#232a34; --text:#edf2f7; --muted:#9aa7b5; --line:#343d4b; --run:#61a8ff; --ok:#48c78e; --bad:#ff6b6b; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); letter-spacing:0; }
    header { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 24px; border-bottom:1px solid var(--line); background:#151922; position:sticky; top:0; z-index:2; }
    h1 { font-size:20px; margin:0; }
    h2 { font-size:14px; margin:0 0 12px; }
    .sub { color:var(--muted); font-size:13px; margin-top:3px; }
    main { display:grid; grid-template-columns:minmax(320px, 430px) minmax(0, 1fr); gap:18px; padding:18px 24px 24px; }
    .section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }
    .section + .section { margin-top:18px; }
    label { display:block; color:var(--muted); font-size:12px; margin:10px 0 5px; }
    input, select { width:100%; border:1px solid var(--line); border-radius:6px; background:var(--field); color:var(--text); padding:9px 10px; font:inherit; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    button { border:1px solid var(--line); background:var(--field); color:var(--text); border-radius:6px; padding:9px 12px; font:inherit; cursor:pointer; }
    button.primary { background:#245fa8; border-color:#3576c8; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .actions { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
    .metrics { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }
    .tile { background:var(--field); border:1px solid var(--line); border-radius:8px; padding:11px; min-height:76px; min-width:0; }
    .tile-label, .tile-sub { color:var(--muted); font-size:12px; }
    .tile-value { font-size:19px; font-weight:750; margin-top:5px; overflow-wrap:anywhere; }
    .status-line { color:var(--muted); border-top:1px solid var(--line); margin-top:12px; padding-top:10px; font-size:13px; }
    .gallery-tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
    .gallery-tabs button.active { background:#245fa8; border-color:#3576c8; }
    .gallery { display:grid; grid-template-columns:repeat(4, minmax(120px, 1fr)); gap:10px; }
    .thumb { background:var(--field); border:1px solid var(--line); border-radius:8px; overflow:hidden; min-width:0; }
    .thumb img { display:block; width:100%; aspect-ratio:1/1; object-fit:contain; background:#05070a; }
    .thumb-name { color:var(--muted); font-size:11px; padding:7px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    pre { margin:0; padding:12px 14px; height:calc(100vh - 480px); min-height:340px; overflow:auto; background:#07090d; color:#d9e2ee; font:12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; white-space:pre-wrap; }
    @media (max-width: 920px) { header, main { padding-left:14px; padding-right:14px; } main { grid-template-columns:1fr; } .metrics, .gallery { grid-template-columns:repeat(2, minmax(120px, 1fr)); } }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>X-ray DVC Test Control Dashboard</h1>
      <div class="sub" id="subtitle">Idle</div>
    </div>
    <button id="stopBtn">Stop</button>
  </header>
  <main>
    <aside>
      <div class="section">
        <h2>Continue Training</h2>
        <label for="sourceModel">Continue from model</label>
        <select id="sourceModel"></select>
        <label for="checkpointEpoch">Checkpoint epoch</label>
        <select id="checkpointEpoch"></select>
        <label for="outputModel">Train into test model</label>
        <input id="outputModel" value="Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST">
        <div class="row">
          <div><label for="nEpochs">Train epochs</label><input id="nEpochs" type="number" min="1" value="5"></div>
          <div><label for="nEpochsDecay">Decay epochs</label><input id="nEpochsDecay" type="number" min="0" value="0"></div>
        </div>
        <label for="maxDatasetSize">Max dataset size</label>
        <input id="maxDatasetSize" type="number" min="1" value="128">
        <div class="actions"><button class="primary" id="trainBtn">Start Training</button></div>
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
        <h2>Current Metrics</h2>
        <div class="metrics" id="metrics"></div>
        <div class="status-line" id="statusLine">Waiting for run</div>
      </div>
      <div class="section">
        <h2>FID Images</h2>
        <div class="gallery-tabs" id="galleryTabs"></div>
        <div class="gallery" id="gallery"></div>
      </div>
      <div class="section">
        <h2>Log Stream</h2>
        <pre id="logs"></pre>
      </div>
    </section>
  </main>
  <script>
    const sourceModel = document.getElementById('sourceModel');
    const evalModel = document.getElementById('evalModel');
    const checkpointEpoch = document.getElementById('checkpointEpoch');
    const evalEpoch = document.getElementById('evalEpoch');
    const logs = document.getElementById('logs');
    let models = [];
    let activeGallery = 'fake';

    function epochsFor(name) {
      const item = models.find(m => m.name === name);
      return item ? item.epochs : ['latest'];
    }
    function fillEpochSelect(select, name) {
      const current = select.value || 'latest';
      select.innerHTML = epochsFor(name).map(e => `<option ${e === current ? 'selected' : ''}>${e}</option>`).join('');
      if (!select.value) select.value = 'latest';
    }
    async function loadOptions() {
      const data = await fetch('/api/options').then(r => r.json());
      models = data.models || [];
      const names = models.map(m => m.name);
      const options = names.map(name => `<option>${name}</option>`).join('');
      sourceModel.innerHTML = options;
      evalModel.innerHTML = options;
      sourceModel.value = names.includes(data.defaults.source_model) ? data.defaults.source_model : (names[0] || '');
      evalModel.value = names.includes(data.defaults.output_model) ? data.defaults.output_model : sourceModel.value;
      fillEpochSelect(checkpointEpoch, sourceModel.value);
      fillEpochSelect(evalEpoch, evalModel.value);
    }
    sourceModel.onchange = () => fillEpochSelect(checkpointEpoch, sourceModel.value);
    evalModel.onchange = () => fillEpochSelect(evalEpoch, evalModel.value);

    async function startAction(action, config) {
      await fetch('/api/start_action', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({action, config})});
    }
    document.getElementById('trainBtn').onclick = () => startAction('train', {
      source_model: sourceModel.value,
      checkpoint_epoch: checkpointEpoch.value,
      output_model: document.getElementById('outputModel').value,
      n_epochs: document.getElementById('nEpochs').value,
      n_epochs_decay: document.getElementById('nEpochsDecay').value,
      max_dataset_size: document.getElementById('maxDatasetSize').value
    });
    document.getElementById('evalBtn').onclick = () => startAction('evaluate_fid', {
      eval_model: evalModel.value,
      eval_epoch: evalEpoch.value,
      phase: document.getElementById('phase').value,
      max_images: document.getElementById('maxImages').value
    });
    document.getElementById('stopBtn').onclick = () => fetch('/api/stop', {method:'POST'});

    function renderGallery(images) {
      const groups = Object.keys(images || {});
      if (!groups.includes(activeGallery)) activeGallery = groups[0] || 'fake';
      document.getElementById('galleryTabs').innerHTML = groups.map(group => `<button class="${group === activeGallery ? 'active' : ''}" data-gallery="${group}">${group} (${(images[group] || []).length})</button>`).join('');
      document.querySelectorAll('[data-gallery]').forEach(btn => btn.onclick = () => { activeGallery = btn.dataset.gallery; renderGallery(images); });
      const selected = images[activeGallery] || [];
      document.getElementById('gallery').innerHTML = selected.length ? selected.map(item => `<div class="thumb" title="${item.name}"><img src="${item.url}" alt="${item.name}" loading="lazy"><div class="thumb-name">${item.name}</div></div>`).join('') : '<div class="sub">No images yet</div>';
    }
    function render(data) {
      const state = data.state;
      document.getElementById('subtitle').textContent = `${state.status.toUpperCase()}${state.action ? ' · ' + state.action : ''}${state.step ? ' · ' + state.step : ''}`;
      document.getElementById('statusLine').textContent = `Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'} · Metrics: ${data.metrics.path}`;
      document.getElementById('trainBtn').disabled = state.running;
      document.getElementById('evalBtn').disabled = state.running;
      document.getElementById('stopBtn').disabled = !state.running;
      document.getElementById('metrics').innerHTML = data.metrics.tiles.map(t => `<div class="tile"><div class="tile-label">${t.label}</div><div class="tile-value">${t.value}</div><div class="tile-sub">${t.sub}</div></div>`).join('');
      renderGallery(data.images || {});
      const atBottom = logs.scrollTop + logs.clientHeight >= logs.scrollHeight - 24;
      logs.textContent = state.logs.join('\n');
      if (atBottom) logs.scrollTop = logs.scrollHeight;
    }
    loadOptions();
    const source = new EventSource('/events');
    source.onmessage = event => render(JSON.parse(event.data));
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "XrayDvcTestControlDashboard/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
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
        if path == "/api/options":
            self.send_json({"models": list_checkpoint_models(), "defaults": {"source_model": DEFAULT_SOURCE_MODEL, "output_model": DEFAULT_OUTPUT_MODEL}})
            return
        if path == "/api/state":
            self.send_json({"state": STATE.snapshot(), "metrics": collect_metrics_for_state(), "images": list_gallery_images()})
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
        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                while True:
                    payload = json.dumps({"state": STATE.snapshot(), "metrics": collect_metrics_for_state(), "images": list_gallery_images()})
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
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
        if path == "/api/start_action":
            with STATE.lock:
                running = STATE.process is not None and STATE.process.poll() is None
            if running:
                self.send_json({"started": False, "reason": "already running"}, status=409)
                return
            action = str(payload.get("action") or "")
            config = payload.get("config") or {}
            thread = threading.Thread(target=run_action, args=(action, config), daemon=True)
            thread.start()
            self.send_json({"started": True, "action": action})
            return
        if path == "/api/stop":
            self.send_json({"stopped": stop_run()})
            return
        self.send_json({"error": "not found"}, status=404)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a configurable dashboard for the experimental DVC test pipeline.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8767)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[dashboard:test-control] open http://{args.host}:{args.port}")
    print("[dashboard:test-control] choose training/evaluation settings in the browser")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
