#!/usr/bin/env python3
"""Mahalanobis Scene Viewer for generation runs.

Reads a generation summary JSON that contains:
- mahalanobis.per_image.scene_xxxx.mahalanobis
- scenes[] metadata
and shows a graph + matching generated image.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageOps, ImageTk, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class SceneRecord:
    stem: str
    index: int
    score: float
    seed: Optional[int]
    scene_mode: str
    placed_count: Optional[int]
    selected_count: Optional[int]


def load_generation_summary(path: Path) -> tuple[List[SceneRecord], dict]:
    data = json.loads(path.read_text())
    mahal = data.get("mahalanobis", {})
    per_image = mahal.get("per_image", {})
    scenes_meta = {s.get("stem"): s for s in data.get("scenes", [])}

    records: List[SceneRecord] = []
    for stem, item in per_image.items():
        if item.get("status") != "ok" or item.get("mahalanobis") is None:
            continue
        m = re.search(r"(\d+)$", stem)
        idx = int(m.group(1)) if m else len(records)
        meta = scenes_meta.get(stem, {})
        records.append(SceneRecord(
            stem=stem,
            index=idx,
            score=float(item["mahalanobis"]),
            seed=meta.get("seed"),
            scene_mode=meta.get("scene_mode", ""),
            placed_count=meta.get("placed_count"),
            selected_count=meta.get("selected_count"),
        ))

    records.sort(key=lambda r: r.index)
    return records, data


def find_best_scene_image(project_root: Path, record: SceneRecord) -> Optional[Path]:
    patterns = [
        f"**/{record.stem}.png",
        f"**/{record.stem}.jpg",
        f"**/{record.stem}.jpeg",
        f"**/{record.stem}.webp",
        f"**/*{record.stem}*",
    ]
    candidates: List[Path] = []
    for pattern in patterns:
        for p in project_root.glob(pattern):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                candidates.append(p)
    if not candidates:
        return None
    candidates = sorted(set(candidates), key=lambda p: (0 if record.stem in p.name else 1, len(p.parts)))
    return candidates[0]


class App:
    def __init__(self, root: tk.Tk, records: List[SceneRecord], summary: dict, summary_path: Path, project_root: Optional[Path]):
        self.root = root
        self.records = records
        self.summary = summary
        self.summary_path = summary_path
        self.project_root = project_root
        self.current_record: Optional[SceneRecord] = None
        self.current_pil = None
        self.current_img_refs = []

        self.root.title("Mahalanobis Scene Viewer")
        self.root.geometry("1600x980")

        self.project_root_var = tk.StringVar(value=str(project_root) if project_root else "")
        self.status_var = tk.StringVar(value=f"Loaded: {summary_path}")
        self.image_title_var = tk.StringVar(value="No image loaded")

        vals = np.array([r.score for r in records], dtype=float)
        self.stats = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "q10": float(np.percentile(vals, 10)),
            "q25": float(np.percentile(vals, 25)),
            "q50": float(np.percentile(vals, 50)),
            "q75": float(np.percentile(vals, 75)),
            "q90": float(np.percentile(vals, 90)),
            "q95": float(np.percentile(vals, 95)),
        }

        self.build_ui()
        self.populate_list()
        self.draw_plot()
        self.select_best()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Project root:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.project_root_var, width=70).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(top, text="Browse", command=self.choose_project_root).pack(side=tk.LEFT)
        ttk.Button(top, text="Best score", command=self.select_best).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(top, text="Refresh image", command=self.update_details).pack(side=tk.LEFT, padx=(6, 0))

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=4)

        ttk.Label(left, text="Scenes").pack(anchor="w")
        self.scene_list = tk.Listbox(left, exportselection=False, width=22, height=26)
        self.scene_list.pack(fill=tk.Y, expand=False)
        self.scene_list.bind("<<ListboxSelect>>", lambda e: self.update_details())
        self.info_text = tk.Text(left, width=42, height=18, wrap="word")
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pane.pack(fill=tk.BOTH, expand=True)
        graph_frame, image_frame = ttk.Frame(right_pane), ttk.Frame(right_pane)
        right_pane.add(graph_frame, weight=3)
        right_pane.add(image_frame, weight=4)

        self.fig = Figure(figsize=(12.5, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        ttk.Label(image_frame, textvariable=self.image_title_var, anchor="w", font=("TkDefaultFont", 11, "bold")).pack(fill=tk.X, pady=(8, 2))
        self.image_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", lambda e: self.redraw_image())

        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def populate_list(self):
        self.scene_list.delete(0, tk.END)
        for r in self.records:
            self.scene_list.insert(tk.END, f"{r.stem} | {r.score:.3f}")

    def choose_project_root(self):
        selected = filedialog.askdirectory(title="Choose project root")
        if selected:
            self.project_root_var.set(selected)
            self.project_root = Path(selected)
            self.update_details()

    def select_best(self):
        if not self.records:
            return
        idx = min(range(len(self.records)), key=lambda i: self.records[i].score)
        self.scene_list.selection_clear(0, tk.END)
        self.scene_list.selection_set(idx)
        self.scene_list.activate(idx)
        self.scene_list.see(idx)
        self.update_details()

    def get_selected(self) -> Optional[SceneRecord]:
        sel = self.scene_list.curselection()
        if not sel:
            return None
        idx = sel[0]
        if idx >= len(self.records):
            return None
        return self.records[idx]

    def draw_plot(self):
        self.ax.clear()
        xs = list(range(len(self.records)))
        ys = [r.score for r in self.records]
        self.ax.plot(xs, ys, marker="o")
        self.ax.set_title("Mahalanobis per generated scene")
        self.ax.set_xlabel("Scene index")
        self.ax.set_ylabel("Mahalanobis distance")
        self.ax.grid(True, alpha=0.3)

        best = min(self.records, key=lambda r: r.score)
        best_idx = self.records.index(best)
        self.ax.scatter([best_idx], [best.score], s=80)

        s = self.stats
        summary_box = (
            f"Current run stats\n"
            f"mean: {s['mean']:.3f}\n"
            f"std: {s['std']:.3f}\n"
            f"min: {s['min']:.3f}\n"
            f"max: {s['max']:.3f}"
        )
        self.ax.text(1.02, 0.98, summary_box, transform=self.ax.transAxes, va="top", ha="left", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9), clip_on=False)

        guide_box = (
            "Empirical guide for THIS setup\n"
            f"very good: <= {s['q10']:.2f}\n"
            f"good/typical: {s['q10']:.2f} - {s['mean']+s['std']:.2f}\n"
            f"borderline: {s['mean']+s['std']:.2f} - {s['mean']+2*s['std']:.2f}\n"
            f"outlier-ish: > {s['mean']+2*s['std']:.2f}\n"
            "Lower is better.\n"
            "Not a universal scale."
        )
        self.ax.text(1.02, 0.68, guide_box, transform=self.ax.transAxes, va="top", ha="left", fontsize=8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9), clip_on=False)
        self.fig.tight_layout(rect=[0, 0, 0.82, 1])
        self.canvas.draw_idle()

    def on_plot_click(self, event):
        if event.xdata is None:
            return
        idx = min(range(len(self.records)), key=lambda i: abs(i - event.xdata))
        self.scene_list.selection_clear(0, tk.END)
        self.scene_list.selection_set(idx)
        self.scene_list.activate(idx)
        self.scene_list.see(idx)
        self.update_details()

    def update_details(self):
        record = self.get_selected()
        if record is None:
            return
        self.current_record = record
        lines = [
            f"Scene: {record.stem}",
            f"Mahalanobis: {record.score:.6f}",
            f"Seed: {record.seed}",
            f"Mode: {record.scene_mode}",
            f"Placed count: {record.placed_count}",
            f"Selected count: {record.selected_count}",
        ]
        project_root = Path(self.project_root_var.get()).expanduser() if self.project_root_var.get().strip() else None
        image_path = find_best_scene_image(project_root, record) if project_root and project_root.exists() else None
        if image_path:
            self.show_image(image_path)
            lines.append(f"Image: {image_path}")
            self.status_var.set(f"Loaded image: {image_path}")
        else:
            self.current_pil = None
            self.image_title_var.set("No matching image found")
            self.image_canvas.delete("all")
            self.image_canvas.create_text(20, 20, anchor="nw", text="No matching image found", fill="white")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", "\n".join(lines))

    def show_image(self, image_path: Path):
        img = Image.open(image_path).convert("RGB")
        self.current_pil = img
        title = f"{image_path.name} | Mahalanobis: {self.current_record.score:.4f}" if self.current_record else image_path.name
        self.image_title_var.set(title)
        self.redraw_image()

    def redraw_image(self):
        if self.current_pil is None:
            return
        self.image_canvas.update_idletasks()
        cw = max(self.image_canvas.winfo_width(), 300)
        ch = max(self.image_canvas.winfo_height(), 220)
        disp = ImageOps.contain(self.current_pil, (cw - 20, ch - 20))
        tk_img = ImageTk.PhotoImage(disp)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
        if self.current_record is not None:
            self.image_canvas.create_text(cw - 12, 12, anchor="ne", fill="white", font=("TkDefaultFont", 10, "bold"),
                                          text=f"Mahalanobis: {self.current_record.score:.4f}")
        self.current_img_refs = [tk_img]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True, help="Generation summary JSON with mahalanobis field")
    ap.add_argument("--project-root", default="", help="Root directory to search matching scene images")
    return ap.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise SystemExit(f"Summary JSON not found: {summary_path}")
    records, summary = load_generation_summary(summary_path)
    if not records:
        raise SystemExit("No valid Mahalanobis scene records found.")
    root = tk.Tk()
    App(root, records, summary, summary_path, Path(args.project_root).expanduser().resolve() if args.project_root else None)
    root.mainloop()


if __name__ == "__main__":
    main()


"""
python Codes_Notebooks/Pix2Pix/mahal_scene_viewer.py \
  --summary-json "results/_gen_stage23_combo_tray/generated/generated_combo_summary.json" \
  --project-root "/home/ssy/Desktop/xray-gen-ai_Project"

"""