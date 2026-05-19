from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse


# ---------------------------------------------------------------------
# Project / test pipeline configuration
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

MAX_LOG_LINES = 1200

TEST_MODEL_NAME = "Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST"

DVC_YAML = ROOT / "dvc_test" / "dvc.yaml"
DVC_CWD = ROOT / "dvc_test"

METRIC_FILES = {
    "real_reference": ROOT / "reports" / "dvc_test" / "real_test_mahal_against_train.json",
    "real_ab_reference": ROOT / "results" / "_dvc_test_gen_real_ab_reference" / "real_ab_input_summary.json",
    "generated_summary": ROOT / "reports" / "dvc_test" / "generated_combo_summary.json",
    "filter_report": ROOT / "reports" / "dvc_test" / "filter_report.json",
    "fid_train": ROOT
    / "reports"
    / "dvc_test"
    / "fid_eval_runs"
    / "train"
    / TEST_MODEL_NAME
    / "epoch_latest"
    / "metrics.json",
    "fid_test": ROOT
    / "reports"
    / "dvc_test"
    / "fid_eval_runs"
    / "test"
    / TEST_MODEL_NAME
    / "epoch_latest"
    / "metrics.json",
    "candidate_eval": ROOT / "reports" / "dvc_test" / "candidate_eval.json",
    "production_registry": ROOT / "models" / "production_test" / "model_registry.json",
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

IMAGE_DIRS = {
    "generated": ROOT / "datasets" / "_dvc_test_generated_combo" / "generated",
    "accepted": ROOT / "datasets" / "_dvc_test_generated_combo" / "accepted",
    "rejected": ROOT / "datasets" / "_dvc_test_generated_combo" / "rejected",
    "real_ab_fake": ROOT / "results" / "_dvc_test_gen_real_ab_reference" / "fake_images",
}


# ---------------------------------------------------------------------
# Dashboard runtime state
# ---------------------------------------------------------------------

class DashboardState:
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
                    stages.append(
                        {
                            "name": match.group(1),
                            "status": "pending",
                            "started_at": None,
                            "finished_at": None,
                        }
                    )

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
            complete = sum(1 for s in self.stages if s["status"] in {"done", "skipped"})

            failed = any(s["status"] == "failed" for s in self.stages) or (
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


STATE = DashboardState()


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def resolve_dvc_command(explicit: str | None) -> str:
    if explicit:
        return explicit

    found = shutil.which("dvc")
    if found:
        return found

    local = ROOT / ".venv" / "bin" / "dvc"
    if local.exists():
        return str(local)

    return "dvc"


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
    phase = metrics.get("phase")
    num_images = metrics.get("num_images")

    parts = []

    if phase:
        parts.append(str(phase))

    if isinstance(num_images, int):
        parts.append(f"{num_images} images")

    if not parts:
        parts.append("lower is better")

    return " | ".join(parts)


def format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return "-"


def format_percent(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return "-"


# ---------------------------------------------------------------------
# Metrics and gallery
# ---------------------------------------------------------------------

def collect_metrics() -> dict:
    raw = {name: read_json(path) for name, path in METRIC_FILES.items()}

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
            {
                "label": "Real AB",
                "value": len(real_ab) if isinstance(real_ab, list) else "-",
                "sub": "reference inputs",
            },
            {
                "label": "Accepted",
                "value": filter_report.get("num_accepted", "-"),
                "sub": "synthetic images",
            },
            {
                "label": "Rejected",
                "value": filter_report.get("num_rejected", "-"),
                "sub": "filtered out",
            },
            {
                "label": "Acceptance",
                "value": format_percent(filter_report.get("acceptance_rate")),
                "sub": "quality gate",
            },
            {
                "label": "FID Train",
                "value": format_float(extract_fid(fid_train)),
                "sub": format_fid_sub(fid_train),
            },
            {
                "label": "FID Test",
                "value": format_float(extract_fid(fid_test)),
                "sub": format_fid_sub(fid_test),
            },
            {
                "label": "Spatial Acc",
                "value": format_percent(summary.get("spatial_accuracy")),
                "sub": "classifier",
            },
            {
                "label": "Threat Acc",
                "value": format_percent(summary.get("threat_accuracy")),
                "sub": "classifier",
            },
            {
                "label": "Decision",
                "value": candidate.get("decision", "-"),
                "sub": "candidate gate",
            },
            {
                "label": "Production",
                "value": registry.get("status", "-"),
                "sub": "promotion",
            },
        ],
        "gates": candidate.get("gates", []),
    }


def list_gallery_images(limit: int = 12) -> dict:
    galleries = {}

    for group, folder in IMAGE_DIRS.items():
        items = []

        if folder.exists():
            paths = sorted(
                p
                for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

            for path in paths[:limit]:
                name = path.name
                items.append(
                    {
                        "name": name,
                        "url": f"/media?group={quote(group)}&name={quote(name)}",
                    }
                )

        galleries[group] = items

    return galleries


def resolve_media_path(group: str, name: str) -> Path | None:
    folder = IMAGE_DIRS.get(group)
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


# ---------------------------------------------------------------------
# DVC runner
# ---------------------------------------------------------------------

def parse_dvc_line(line: str) -> None:
    running_match = re.search(r"Running stage '([^']+)'", line)
    skipped_match = re.search(r"Stage '([^']+)' didn't change, skipping", line)
    unchanged_match = re.search(r"Data and pipelines are up to date", line)

    with STATE.lock:
        if running_match:
            name = running_match.group(1)

            if STATE.current_stage and STATE.current_stage != name:
                STATE.set_stage(STATE.current_stage, "done")

            STATE.current_stage = name
            STATE.set_stage(name, "running")

        elif skipped_match:
            STATE.set_stage(skipped_match.group(1), "skipped")

        elif unchanged_match:
            for stage in STATE.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"

        if line.startswith("ERROR:") and STATE.current_stage:
            STATE.set_stage(STATE.current_stage, "failed")

        STATE.append_log(line)


def run_dvc(dvc_cmd: str, target: str | None = None, force: bool = False) -> None:
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
        with STATE.lock:
            STATE.returncode = 1
            STATE.finished_at = datetime.now().isoformat(timespec="seconds")
            STATE.logs = [f"$ {' '.join(cmd)}", f"[dashboard:test] ERROR: {exc}"]
            STATE.process = None
        return

    with STATE.lock:
        STATE.process = process
        STATE.started_at = datetime.now().isoformat(timespec="seconds")
        STATE.finished_at = None
        STATE.returncode = None
        STATE.current_stage = None
        STATE.logs = [f"$ {' '.join(cmd)}"]
        STATE.stages = STATE.load_stages()

    assert process.stdout is not None

    for line in process.stdout:
        parse_dvc_line(line)

    returncode = process.wait()

    with STATE.lock:
        if STATE.current_stage:
            STATE.set_stage(STATE.current_stage, "done" if returncode == 0 else "failed")

        if returncode == 0:
            for stage in STATE.stages:
                if stage["status"] == "pending":
                    stage["status"] = "skipped"

        STATE.returncode = returncode
        STATE.finished_at = datetime.now().isoformat(timespec="seconds")
        STATE.append_log(f"[dashboard:test] dvc repro exited with code {returncode}")
        STATE.process = None


def stop_dvc() -> bool:
    with STATE.lock:
        proc = STATE.process

    if proc is None or proc.poll() is not None:
        return False

    proc.send_signal(signal.SIGTERM)
    return True


# ---------------------------------------------------------------------
# Frontend HTML
# ---------------------------------------------------------------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>X-ray DVC Test Pipeline</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111317;
      --panel: #1b2028;
      --panel-2: #232a34;
      --text: #edf2f7;
      --muted: #9aa7b5;
      --line: #343d4b;
      --ok: #48c78e;
      --warn: #ffd166;
      --bad: #ff6b6b;
      --run: #61a8ff;
      --hold: #b9a7ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #151922;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 20px; margin: 0; font-weight: 700; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 3px; }
    main {
      display: grid;
      grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px 24px 24px;
    }
    button {
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      padding: 9px 12px;
      font: inherit;
      cursor: pointer;
    }
    button.primary { background: #245fa8; border-color: #3576c8; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    .controls { display: flex; gap: 8px; flex-wrap: wrap; }
    .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .section h2 {
      margin: 0;
      padding: 12px 14px;
      font-size: 14px;
      border-bottom: 1px solid var(--line);
    }
    .progress {
      height: 8px;
      background: #0c0f14;
      overflow: hidden;
    }
    .bar {
      height: 100%;
      width: 0;
      background: var(--run);
      transition: width .25s ease;
    }
    .stage-list { padding: 8px; }
    .stage {
      display: grid;
      grid-template-columns: 14px 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 9px 8px;
      border-radius: 6px;
    }
    .stage + .stage { margin-top: 4px; }
    .stage.running { background: rgba(97, 168, 255, .11); }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--muted);
    }
    .done .dot, .skipped .dot { background: var(--ok); }
    .running .dot { background: var(--run); box-shadow: 0 0 0 4px rgba(97, 168, 255, .14); }
    .failed .dot { background: var(--bad); }
    .stage-name { font-size: 14px; overflow-wrap: anywhere; }
    .badge {
      color: var(--muted);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 7px;
      font-size: 11px;
      text-transform: uppercase;
    }
    .running .badge { color: var(--run); border-color: rgba(97, 168, 255, .45); }
    .done .badge, .skipped .badge { color: var(--ok); border-color: rgba(72, 199, 142, .45); }
    .failed .badge { color: var(--bad); border-color: rgba(255, 107, 107, .55); }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      padding: 12px;
    }
    .tile {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      min-height: 76px;
    }
    .tile-label { color: var(--muted); font-size: 12px; }
    .tile-value { font-size: 22px; font-weight: 750; margin-top: 5px; overflow-wrap: anywhere; }
    .tile-sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .gates { padding: 0 12px 12px; }
    .gate {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      border-top: 1px solid var(--line);
      padding: 9px 0;
      font-size: 13px;
    }
    .pass { color: var(--ok); }
    .fail { color: var(--bad); }
    .gallery-tabs {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 12px 12px 0;
    }
    .gallery-tabs button.active {
      background: #245fa8;
      border-color: #3576c8;
    }
    .gallery {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      padding: 12px;
    }
    .thumb {
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      min-width: 0;
    }
    .thumb img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #05070a;
    }
    .thumb-name {
      color: var(--muted);
      font-size: 11px;
      padding: 7px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    pre {
      margin: 0;
      padding: 12px 14px;
      height: calc(100vh - 360px);
      min-height: 340px;
      overflow: auto;
      background: #07090d;
      color: #d9e2ee;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
    }
    .status-line {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      font-size: 13px;
    }
    @media (max-width: 920px) {
      header, main { padding-left: 14px; padding-right: 14px; }
      main { grid-template-columns: 1fr; }
      .metrics { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .gallery { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      pre { height: 420px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>X-ray DVC Test Pipeline</h1>
      <div class="sub" id="subtitle">Idle</div>
    </div>
    <div class="controls">
      <button class="primary" id="startBtn">Start Test DVC</button>
      <button id="forceBtn">Force Run</button>
      <button id="stopBtn">Stop</button>
    </div>
  </header>
  <main>
    <aside class="section">
      <h2>Stages</h2>
      <div class="progress"><div class="bar" id="bar"></div></div>
      <div class="stage-list" id="stages"></div>
    </aside>
    <section>
      <div class="section">
        <h2>Live Metrics</h2>
        <div class="metrics" id="metrics"></div>
        <div class="gates" id="gates"></div>
      </div>
      <div class="section" style="margin-top:18px">
        <h2>Generated Images</h2>
        <div class="gallery-tabs" id="galleryTabs"></div>
        <div class="gallery" id="gallery"></div>
      </div>
      <div class="section" style="margin-top:18px">
        <h2>Log Stream</h2>
        <div class="status-line" id="statusLine">Waiting for run</div>
        <pre id="logs"></pre>
      </div>
    </section>
  </main>
  <script>
    const startBtn = document.getElementById('startBtn');
    const forceBtn = document.getElementById('forceBtn');
    const stopBtn = document.getElementById('stopBtn');
    const logs = document.getElementById('logs');
    let activeGallery = 'accepted';

    async function start(force=false) {
      await fetch('/api/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({force})
      });
    }

    async function stopRun() {
      await fetch('/api/stop', {method: 'POST'});
    }

    startBtn.onclick = () => start(false);
    forceBtn.onclick = () => start(true);
    stopBtn.onclick = stopRun;

    function galleryLabel(key) {
      return {
        generated: 'Generated',
        accepted: 'Accepted',
        rejected: 'Rejected',
        real_ab_fake: 'Real AB Fake'
      }[key] || key;
    }

    function renderGallery(images) {
      const groups = Object.keys(images || {});
      if (!groups.includes(activeGallery)) activeGallery = groups[0] || 'accepted';

      document.getElementById('galleryTabs').innerHTML = groups.map(group => `
        <button class="${group === activeGallery ? 'active' : ''}" data-gallery="${group}">
          ${galleryLabel(group)} (${(images[group] || []).length})
        </button>
      `).join('');

      document.querySelectorAll('[data-gallery]').forEach(btn => {
        btn.onclick = () => {
          activeGallery = btn.dataset.gallery;
          renderGallery(images);
        };
      });

      const selected = images[activeGallery] || [];

      document.getElementById('gallery').innerHTML = selected.length ? selected.map(item => `
        <div class="thumb" title="${item.name}">
          <img src="${item.url}" alt="${item.name}" loading="lazy">
          <div class="thumb-name">${item.name}</div>
        </div>
      `).join('') : '<div class="sub" style="padding:8px">No images yet</div>';
    }

    function render(data) {
      const state = data.state;
      const metrics = data.metrics;

      document.getElementById('subtitle').textContent =
        `${state.status.toUpperCase()}${state.current_stage ? ' · ' + state.current_stage : ''}`;

      document.getElementById('statusLine').textContent =
        `Started: ${state.started_at || '-'} · Finished: ${state.finished_at || '-'} · Return: ${state.returncode ?? '-'}`;

      document.getElementById('bar').style.width =
        `${Math.round((state.progress || 0) * 100)}%`;

      startBtn.disabled = state.running;
      forceBtn.disabled = state.running;
      stopBtn.disabled = !state.running;

      document.getElementById('stages').innerHTML = state.stages.map(s => `
        <div class="stage ${s.status}">
          <span class="dot"></span>
          <span class="stage-name">${s.name}</span>
          <span class="badge">${s.status}</span>
        </div>
      `).join('');

      document.getElementById('metrics').innerHTML = metrics.tiles.map(t => `
        <div class="tile">
          <div class="tile-label">${t.label}</div>
          <div class="tile-value">${t.value}</div>
          <div class="tile-sub">${t.sub}</div>
        </div>
      `).join('');

      document.getElementById('gates').innerHTML = (metrics.gates || []).map(g => `
        <div class="gate">
          <span>${g.name}: ${g.value} ${g.operator} ${g.threshold}</span>
          <strong class="${g.passed ? 'pass' : 'fail'}">${g.passed ? 'PASS' : 'FAIL'}</strong>
        </div>
      `).join('');

      renderGallery(data.images || {});

      const atBottom = logs.scrollTop + logs.clientHeight >= logs.scrollHeight - 24;
      logs.textContent = state.logs.join('\n');

      if (atBottom) {
        logs.scrollTop = logs.scrollHeight;
      }
    }

    const source = new EventSource('/events');
    source.onmessage = event => render(JSON.parse(event.data));
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "XrayDvcTestDashboardStandalone/1.0"

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

        if path == "/api/state":
            self.send_json(
                {
                    "state": STATE.snapshot(),
                    "metrics": collect_metrics(),
                    "images": list_gallery_images(),
                }
            )
            return

        if path == "/api/metrics":
            self.send_json(collect_metrics())
            return

        if path == "/api/images":
            self.send_json(list_gallery_images())
            return

        if path == "/media":
            query = parse_qs(urlparse(self.path).query)
            group = (query.get("group") or [""])[0]
            name = (query.get("name") or [""])[0]

            media_path = resolve_media_path(group, name)

            if media_path is None:
                self.send_json({"error": "image not found"}, status=404)
                return

            body = media_path.read_bytes()
            content_type = (
                mimetypes.guess_type(str(media_path))[0]
                or "application/octet-stream"
            )

            self.send_response(200)
            self.send_header("Content-Type", content_type)
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
                    payload = json.dumps(
                        {
                            "state": STATE.snapshot(),
                            "metrics": collect_metrics(),
                            "images": list_gallery_images(),
                        }
                    )
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

        if path == "/api/start":
            with STATE.lock:
                running = STATE.process is not None and STATE.process.poll() is None

            if running:
                self.send_json(
                    {"started": False, "reason": "already running"},
                    status=409,
                )
                return

            dvc_cmd = resolve_dvc_command(payload.get("dvc_cmd"))
            target = payload.get("target") or None
            force = bool(payload.get("force", False))

            thread = threading.Thread(
                target=run_dvc,
                args=(dvc_cmd, target, force),
                daemon=True,
            )
            thread.start()

            self.send_json(
                {
                    "started": True,
                    "dvc_cmd": dvc_cmd,
                    "target": target,
                    "force": force,
                }
            )
            return

        if path == "/api/stop":
            self.send_json({"stopped": stop_dvc()})
            return

        self.send_json({"error": "not found"}, status=404)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a standalone live dashboard for the experimental DVC test pipeline."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)

    args = parser.parse_args()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"[dashboard:test] open http://{args.host}:{args.port}")
    print(f"[dashboard:test] project root: {ROOT}")
    print(f"[dashboard:test] dvc cwd: {DVC_CWD}")
    print(f"[dashboard:test] dvc yaml: {DVC_YAML}")
    print("[dashboard:test] click Start Test DVC to run the test pipeline")

    httpd.serve_forever()


if __name__ == "__main__":
    main()