#!/usr/bin/env python3
"""
StyleGAN2-ADA stats.jsonl GUI viewer (Tkinter + Matplotlib)

Shows 4 live-updating plots:
- Loss/G/loss
- Loss/D/loss
- Progress/augment
- Timing/sec_per_kimg

Usage:
  python stats_gui.py --stats models/generator/stylegan/runs/00000-.../stats.jsonl

Optional:
  python stats_gui.py --stats .../stats.jsonl --refresh 2.0
  python stats_gui.py --stats .../stats.jsonl --no-tail   # parse whole file every refresh (slower)
"""

import json
import time
import argparse
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


KEYS = [
    ("Loss/G/loss", "Loss/G/loss"),
    ("Loss/D/loss", "Loss/D/loss"),
    ("Progress/augment", "Progress/augment"),
    ("Timing/sec_per_kimg", "Timing/sec_per_kimg"),
]


def safe_get(d: dict, key: str):
    """Return scalar float if possible, else None."""
    v = d.get(key, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


class StatsReader:
    """
    Efficient reader that can "tail" a growing stats.jsonl file by tracking byte offset.
    """
    def __init__(self, path: Path, tail_mode: bool = True):
        self.path = Path(path)
        self.tail_mode = tail_mode
        self.offset = 0

        # Storage
        self.ticks = []
        self.data = {k: [] for _, k in KEYS}
        self.timestamps = []  # unix seconds

    def reset(self):
        self.offset = 0
        self.ticks.clear()
        self.timestamps.clear()
        for k in self.data:
            self.data[k].clear()

    def read_new(self):
        if not self.path.exists():
            raise FileNotFoundError(f"stats file not found: {self.path}")

        mode = "rb"  # read bytes so offsets are safe
        with open(self.path, mode) as f:
            if self.tail_mode:
                f.seek(self.offset)
            else:
                f.seek(0)

            chunk = f.read()
            if self.tail_mode:
                self.offset = f.tell()

        if not chunk:
            return 0

        # Decode lines safely
        text = chunk.decode("utf-8", errors="replace")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        added = 0

        # If not tail_mode, re-parse whole file and replace arrays
        if not self.tail_mode:
            self.reset()

        for ln in lines:
            try:
                obj = json.loads(ln)
            except Exception:
                continue

            tick = safe_get(obj, "Progress/tick")
            ts = safe_get(obj, "timestamp")
            if tick is None:
                continue

            # Append tick & metrics
            self.ticks.append(tick)
            self.timestamps.append(ts if ts is not None else time.time())
            for _, k in KEYS:
                val = safe_get(obj, k)
                self.data[k].append(val)
            added += 1

        return added


class StatsGUI:
    def __init__(self, root: tk.Tk, stats_path: Path | None, refresh_s: float = 2.0, tail_mode: bool = True):
        self.root = root
        self.root.title("StyleGAN2-ADA stats.jsonl Viewer")
        self.refresh_s = refresh_s

        self.stats_path = stats_path
        self.reader = StatsReader(stats_path, tail_mode=tail_mode) if stats_path else None

        self._build_ui()
        self._build_plots()

        self.running = False
        self.last_update_label = None

        if self.stats_path:
            self._set_path(self.stats_path)

    def _build_ui(self):
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(fill="x")

        self.path_var = tk.StringVar(value=str(self.stats_path) if self.stats_path else "")
        tk.Label(top, text="stats.jsonl:").pack(side="left")
        self.path_entry = tk.Entry(top, textvariable=self.path_var, width=80)
        self.path_entry.pack(side="left", padx=6)

        tk.Button(top, text="Browse", command=self._browse).pack(side="left", padx=4)
        tk.Button(top, text="Load", command=self._load_clicked).pack(side="left", padx=4)

        self.run_btn = tk.Button(top, text="Start", command=self._toggle_run)
        self.run_btn.pack(side="left", padx=8)

        self.tail_var = tk.BooleanVar(value=True)
        self.tail_chk = tk.Checkbutton(top, text="Tail mode (fast)", variable=self.tail_var, command=self._toggle_tail_mode)
        self.tail_chk.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(self.root, textvariable=self.status_var, anchor="w", padx=10).pack(fill="x")

    def _build_plots(self):
        self.fig = plt.Figure(figsize=(11, 7), dpi=100)
        self.axes = []
        for i in range(4):
            ax = self.fig.add_subplot(2, 2, i + 1)
            self.axes.append(ax)

        self.lines = {}
        for ax, (title, key) in zip(self.axes, KEYS):
            ax.set_title(title)
            ax.set_xlabel("tick")
            ax.grid(True, alpha=0.25)
            (line,) = ax.plot([], [], marker=".", linestyle="-", linewidth=1)
            self.lines[key] = line

        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_path(self, p: Path):
        self.stats_path = Path(p)
        self.path_var.set(str(self.stats_path))
        self.reader = StatsReader(self.stats_path, tail_mode=self.tail_var.get())
        self.reader.reset()
        self._update_once(force_full_redraw=True)

    def _browse(self):
        p = filedialog.askopenfilename(title="Select stats.jsonl", filetypes=[("jsonl", "*.jsonl"), ("All files", "*.*")])
        if p:
            self._set_path(Path(p))

    def _load_clicked(self):
        p = self.path_var.get().strip()
        if not p:
            messagebox.showerror("Missing path", "Please provide a path to stats.jsonl")
            return
        self._set_path(Path(p))

    def _toggle_tail_mode(self):
        if self.reader is not None:
            self.reader.tail_mode = self.tail_var.get()
            # If switching off tail mode, it's better to re-parse whole file
            if not self.reader.tail_mode:
                self.reader.reset()
            self._update_once(force_full_redraw=True)

    def _toggle_run(self):
        self.running = not self.running
        self.run_btn.configure(text="Stop" if self.running else "Start")
        if self.running:
            self._schedule_next()

    def _schedule_next(self):
        if not self.running:
            return
        self._update_once()
        ms = int(max(0.5, self.refresh_s) * 1000)
        self.root.after(ms, self._schedule_next)

    def _update_once(self, force_full_redraw: bool = False):
        if self.reader is None:
            return

        try:
            added = self.reader.read_new()
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            return

        ticks = self.reader.ticks
        if not ticks:
            self.status_var.set("No data yet (waiting for stats.jsonl to get entries)...")
            return

        # Update plots
        for (_, key), ax in zip(KEYS, self.axes):
            ys = self.reader.data[key]
            xs = ticks[: len(ys)]
            # Filter Nones so matplotlib doesn't crash
            xf, yf = [], []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                xf.append(x)
                yf.append(y)

            self.lines[key].set_data(xf, yf)
            if force_full_redraw or added > 0:
                if len(xf) >= 2:
                    ax.set_xlim(min(xf), max(xf))
                # autoscale y
                if len(yf) >= 2:
                    ymin, ymax = min(yf), max(yf)
                    pad = (ymax - ymin) * 0.10 if ymax > ymin else 1.0
                    ax.set_ylim(ymin - pad, ymax + pad)

        # Status line
        last_tick = ticks[-1]
        last_kimg = self._last_non_none("Progress/kimg")
        last_aug = self._last_non_none("Progress/augment")
        spk = self._last_non_none("Timing/sec_per_kimg")
        g_loss = self._last_non_none("Loss/G/loss")
        d_loss = self._last_non_none("Loss/D/loss")

        parts = [f"tick={last_tick:.0f}"]
        if last_kimg is not None:
            parts.append(f"kimg={last_kimg:.3f}")
        if spk is not None:
            parts.append(f"sec/kimg={spk:.2f}")
        if last_aug is not None:
            parts.append(f"augment={last_aug:.4f}")
        if g_loss is not None:
            parts.append(f"G={g_loss:.3f}")
        if d_loss is not None:
            parts.append(f"D={d_loss:.3f}")
        if added > 0:
            parts.append(f"(+{added} new lines)")

        self.status_var.set(" | ".join(parts))
        self.canvas.draw_idle()

    def _last_non_none(self, key: str):
        if self.reader is None:
            return None
        ys = self.reader.data.get(key, [])
        for v in reversed(ys):
            if v is not None:
                return v
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=str, default="", help="Path to run_dir/stats.jsonl")
    ap.add_argument("--refresh", type=float, default=2.0, help="GUI refresh seconds (default 2.0)")
    ap.add_argument("--no-tail", action="store_true", help="Disable tail mode (re-parse whole file each refresh)")
    args = ap.parse_args()

    stats_path = Path(args.stats) if args.stats else None
    if stats_path and stats_path.is_dir():
        stats_path = stats_path / "stats.jsonl"

    root = tk.Tk()
    gui = StatsGUI(
        root=root,
        stats_path=stats_path,
        refresh_s=args.refresh,
        tail_mode=(not args.no_tail),
    )
    root.mainloop()


if __name__ == "__main__":
    main()



"""
python stats_gui.py \
  --stats models/generator/stylegan/runs/00001-myset_256-mirror-paper256-kimg200-batch8-ada-color/stats.jsonl \
  --refresh 1.5
"""