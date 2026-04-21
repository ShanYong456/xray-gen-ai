#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import ttk

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
    realism_score: float
    novelty_score: float
    status: str


def load_summary_json(path: Path):
    data = json.loads(path.read_text())

    realism = data.get("realism_mahalanobis", {})
    novelty = data.get("real_nn_novelty", {})
    realism_per = realism.get("per_image", {})
    novelty_per = novelty.get("per_image", {})

    records: List[SceneRecord] = []
    for stem, rinfo in realism_per.items():
        ninfo = novelty_per.get(stem, {})
        rscore = rinfo.get("mahalanobis")
        nscore = ninfo.get("nearest_real_distance")
        rstatus = rinfo.get("status", "")
        nstatus = ninfo.get("status", "")

        if rscore is None or nscore is None or rstatus != "ok" or nstatus != "ok":
            continue

        m = re.search(r"(\d+)$", stem)
        idx = int(m.group(1)) if m else len(records)

        records.append(SceneRecord(
            stem=stem,
            index=idx,
            realism_score=float(rscore),
            novelty_score=float(nscore),
            status="ok",
        ))

    records.sort(key=lambda r: r.index)

    if not records:
        raise SystemExit(
            "No valid generated image records found in summary JSON. "
            "Make sure the summary contains realism_mahalanobis.per_image and real_nn_novelty.per_image."
        )

    return records, data


def find_best_scene_image(images_dir: Path, record: SceneRecord) -> Optional[Path]:
    if not images_dir.exists():
        return None

    preferred = [
        images_dir / f"{record.stem}_combo_smooth2x.png",
        images_dir / f"{record.stem}_shampoo_smooth2x.png",
        images_dir / f"{record.stem}_blade_smooth2x.png",
        images_dir / f"{record.stem}_tray_smooth2x.png",
        images_dir / f"{record.stem}.png",
        images_dir / f"{record.stem}.jpg",
        images_dir / f"{record.stem}.jpeg",
        images_dir / f"{record.stem}.webp",
    ]
    for p in preferred:
        if p.exists() and p.is_file():
            return p

    candidates = sorted([
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and record.stem in p.name
    ])
    return candidates[0] if candidates else None


class App:
    def __init__(self, root: tk.Tk, records: List[SceneRecord], summary: dict, summary_path: Path, images_dir: Path):
        self.root = root
        self.records = records
        self.summary = summary
        self.summary_path = summary_path
        self.images_dir = images_dir

        self.current_record: Optional[SceneRecord] = None
        self.current_pil = None
        self.current_img_refs = []

        self.root.title("Generated Image Realism + Novelty Viewer")
        self.root.geometry("1680x1000")

        self.status_var = tk.StringVar(value=f"Loaded summary: {summary_path}")
        self.image_title_var = tk.StringVar(value="No image loaded")

        vals = np.array([r.realism_score for r in records], dtype=float)
        self.stats = {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }

        self.build_ui()
        self.populate_list()
        self.draw_plot()
        self.select_best()

    def build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text=f"Generated images dir: {self.images_dir}").pack(side=tk.LEFT)
        ttk.Button(top, text="Best realism", command=self.select_best).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(top, text="Refresh image", command=self.update_details).pack(side=tk.LEFT, padx=(6, 0))

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=4)

        ttk.Label(left, text="Generated scenes").pack(anchor="w")
        self.scene_list = tk.Listbox(left, exportselection=False, width=46, height=28)
        self.scene_list.pack(fill=tk.Y, expand=False)
        self.scene_list.bind("<<ListboxSelect>>", lambda e: self.update_details())

        self.info_text = tk.Text(left, width=52, height=22, wrap="word")
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pane.pack(fill=tk.BOTH, expand=True)
        graph_frame, image_frame = ttk.Frame(right_pane), ttk.Frame(right_pane)
        right_pane.add(graph_frame, weight=3)
        right_pane.add(image_frame, weight=4)

        self.fig = Figure(figsize=(13, 4.8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        ttk.Label(
            image_frame,
            textvariable=self.image_title_var,
            anchor="w",
            font=("TkDefaultFont", 11, "bold")
        ).pack(fill=tk.X, pady=(8, 2))

        self.image_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", lambda e: self.redraw_image())

        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def populate_list(self):
        self.scene_list.delete(0, tk.END)
        for r in self.records:
            self.scene_list.insert(
                tk.END,
                f"{r.stem} | R:{r.realism_score:.3f} ↓ | N:{r.novelty_score:.3f} ↑"
            )

    def select_best(self):
        if not self.records:
            return
        idx = min(range(len(self.records)), key=lambda i: self.records[i].realism_score)
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
        ys = [r.realism_score for r in self.records]

        self.ax.plot(xs, ys, marker="o")
        self.ax.set_title("Generated image realism_mahalanobis")
        self.ax.set_xlabel("Scene index")
        self.ax.set_ylabel("realism_mahalanobis (lower is better)")
        self.ax.grid(True, alpha=0.3)

        best = min(self.records, key=lambda r: r.realism_score)
        best_idx = self.records.index(best)
        self.ax.scatter([best_idx], [best.realism_score], s=80)

        info_box = (
            "Meaning\n"
            "R = realism_mahalanobis\n"
            "lower is better\n"
            "\n"
            "N = real_nn_novelty\n"
            "higher = more different\n"
            "from closest real image"
        )
        self.ax.text(
            1.02, 0.98, info_box,
            transform=self.ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            clip_on=False
        )

        stats_box = (
            "Realism stats\n"
            f"mean: {self.stats['mean']:.3f}\n"
            f"std: {self.stats['std']:.3f}\n"
            f"min: {self.stats['min']:.3f}\n"
            f"max: {self.stats['max']:.3f}"
        )
        self.ax.text(
            1.02, 0.64, stats_box,
            transform=self.ax.transAxes,
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9),
            clip_on=False
        )

        self.fig.tight_layout(rect=[0, 0, 0.80, 1])
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
        image_path = find_best_scene_image(self.images_dir, record)

        lines = [
            f"Scene: {record.stem}",
            f"realism_mahalanobis: {record.realism_score:.6f} (lower is better)",
            f"real_nn_novelty: {record.novelty_score:.6f} (higher = more different from closest real image)",
            f"Status: {record.status}",
        ]

        if image_path:
            self.show_image(image_path)
            lines.append("")
            lines.append(f"Image: {image_path}")
            self.status_var.set(f"Loaded image: {image_path}")
        else:
            self.current_pil = None
            self.image_title_var.set("No matching image found")
            self.image_canvas.delete("all")
            self.image_canvas.create_text(20, 20, anchor="nw", text="No matching image found", fill="white")
            self.status_var.set("No matching image found")

        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", "\n".join(lines))

    def show_image(self, image_path: Path):
        img = Image.open(image_path).convert("RGB")
        self.current_pil = img
        title = (
            f"{image_path.name} | "
            f"R={self.current_record.realism_score:.4f} ↓ | "
            f"N={self.current_record.novelty_score:.4f} ↑"
        )
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
            self.image_canvas.create_text(
                cw - 12, 12,
                anchor="ne",
                fill="white",
                font=("TkDefaultFont", 10, "bold"),
                text=f"R:{self.current_record.realism_score:.3f} ↓ | N:{self.current_record.novelty_score:.3f} ↑"
            )
        self.current_img_refs = [tk_img]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-json", required=True, help="generated summary JSON with realism_mahalanobis and real_nn_novelty")
    ap.add_argument("--generated-images-dir", required=True, help="Folder containing generated PNG images")
    return ap.parse_args()


def main():
    args = parse_args()
    summary_path = Path(args.summary_json).expanduser().resolve()
    if not summary_path.exists():
        raise SystemExit(f"Summary JSON not found: {summary_path}")

    images_dir = Path(args.generated_images_dir).expanduser().resolve()
    if not images_dir.exists():
        raise SystemExit(f"Generated images dir not found: {images_dir}")

    records, summary = load_summary_json(summary_path)

    root = tk.Tk()
    App(root, records, summary, summary_path, images_dir)
    root.mainloop()


if __name__ == "__main__":
    main()

"""
python Codes_Notebooks/Pix2Pix/mahal_scene_viewer.py \
  --summary-json results/_gen_stage23_combo_tray/generated/generated_combo_summary.json \
  --generated-images-dir results/_gen_stage23_combo_tray/generated
"""