#!/usr/bin/env python3
"""
FID Epoch Viewer
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import tkinter as tk
from tkinter import filedialog, ttk

from PIL import Image, ImageOps, ImageTk, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_IMAGE_GLOBS = [
    "results/**/epoch_{epoch}/**/*",
    "results/**/*epoch*{epoch}*",
    "checkpoints/**/web/images/*{epoch}*",
    "fid_eval_runs/**/epoch_{epoch}/**/*",
    "fid_eval_runs/**/**/*{epoch}*",
    "**/*epoch_{epoch}*",
    "**/*_{epoch}*",
]
FID_GUIDE_TEXT = (
    "FID guide (rough only)\n"
    "< 10   : excellent\n"
    "10-30  : strong\n"
    "30-50  : moderate\n"
    "> 50   : weak\n"
    "Depends on dataset/setup."
)

@dataclass
class Record:
    run_name: str
    epoch: int
    fid: float
    phase: str
    num_images: int
    metrics_json: str
    raw: dict


def extract_stage_num(run_name: str) -> int:
    m = re.search(r"Stage(\d+)", run_name, re.IGNORECASE)
    return int(m.group(1)) if m else 10**9


def infer_run_name(metrics_json: str) -> str:
    p = Path(metrics_json)
    parts = p.parts
    if "fid_eval_runs" in parts:
        idx = parts.index("fid_eval_runs")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    for i, part in enumerate(parts):
        if re.fullmatch(r"epoch_\d+", part) and i > 0:
            return parts[i - 1]
    return p.parent.name or "unknown_run"


def load_records(metrics_file: Path) -> List[Record]:
    records: List[Record] = []
    with metrics_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj or "epoch" not in obj or "fid" not in obj:
                continue
            metrics_json = str(obj.get("metrics_json", ""))
            try:
                records.append(Record(
                    run_name=infer_run_name(metrics_json),
                    epoch=int(obj["epoch"]),
                    fid=float(obj["fid"]),
                    phase=str(obj.get("phase", "")),
                    num_images=int(obj.get("num_images", 0)),
                    metrics_json=metrics_json,
                    raw=obj,
                ))
            except (TypeError, ValueError):
                continue
    return records


def deduplicate_records(records: Sequence[Record]) -> Dict[str, List[Record]]:
    grouped: Dict[str, Dict[int, Record]] = {}
    for r in records:
        grouped.setdefault(r.run_name, {})
        old = grouped[r.run_name].get(r.epoch)
        if old is None or r.fid < old.fid:
            grouped[r.run_name][r.epoch] = r
    return {k: sorted(v.values(), key=lambda x: x.epoch) for k, v in grouped.items()}


def safe_rel(p: Path, base: Optional[Path]) -> str:
    if base is None:
        return str(p)
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)


def score_candidate(path: Path, epoch: int, run_name: str) -> Tuple[int, int, int, int]:
    text = str(path).lower()
    return (
        0 if run_name.lower() in text else 1,
        0 if f"epoch_{epoch}" in text else 1,
        0 if re.search(rf"(?<!\d){epoch}(?!\d)", text) else 1,
        len(path.parts),
    )


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def find_best_image(project_root: Path, record: Record) -> Optional[Path]:
    run_name, epoch = record.run_name, record.epoch
    mj = project_root / record.metrics_json if record.metrics_json else None
    candidates: List[Path] = []
    if mj is not None:
        for base in [mj.parent, mj.parent.parent if mj.parent != mj.parent.parent else mj.parent]:
            if base.exists():
                for p in base.rglob("*"):
                    if is_image(p) and (f"epoch_{epoch}" in str(p) or re.search(rf"(?<!\d){epoch}(?!\d)", str(p))):
                        candidates.append(p)
    if not candidates:
        for pattern in DEFAULT_IMAGE_GLOBS:
            for p in project_root.glob(pattern.format(epoch=epoch)):
                if is_image(p):
                    candidates.append(p)
    if not candidates:
        for p in project_root.rglob("*"):
            if is_image(p) and run_name.lower() in str(p).lower():
                candidates.append(p)
    return sorted(set(candidates), key=lambda p: score_candidate(p, epoch, run_name))[0] if candidates else None


class App:
    def __init__(self, root: tk.Tk, records_by_run: Dict[str, List[Record]], metrics_file: Path, project_root: Optional[Path]):
        self.root = root
        self.records_by_run = records_by_run
        self.metrics_file = metrics_file
        self.project_root = project_root
        self.current_image_tk = []
        self.current_display_panels = None
        self.current_record: Optional[Record] = None
        self.current_image_path: Optional[Path] = None

        self.root.title("FID Epoch Viewer")
        self.root.geometry("1600x980")

        self.run_names = ["__ALL__"] + sorted(records_by_run.keys(), key=extract_stage_num)
        self.selected_run = tk.StringVar(value="__ALL__")
        self.status_var = tk.StringVar(value=f"Loaded metrics: {metrics_file}")
        self.project_root_var = tk.StringVar(value=str(project_root) if project_root else "")

        self.build_ui()
        if self.run_names:
            self.update_epoch_list()
            self.draw_plot()
            self.select_best_epoch()

    def build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="View:").pack(side=tk.LEFT)
        run_box = ttk.Combobox(top, textvariable=self.selected_run, values=self.run_names, state="readonly", width=60)
        run_box.pack(side=tk.LEFT, padx=(6, 12))
        run_box.bind("<<ComboboxSelected>>", lambda e: self.on_run_changed())
        ttk.Label(top, text="Project root:").pack(side=tk.LEFT)
        ttk.Entry(top, textvariable=self.project_root_var, width=55).pack(side=tk.LEFT, padx=(6, 6))
        ttk.Button(top, text="Browse", command=self.choose_project_root).pack(side=tk.LEFT)
        ttk.Button(top, text="Best epoch", command=self.select_best_epoch).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(top, text="Refresh image", command=self.update_selected_epoch_details).pack(side=tk.LEFT, padx=(6, 0))

        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=4)

        ttk.Label(left, text="Epochs").pack(anchor="w")
        self.epoch_list = tk.Listbox(left, exportselection=False, width=22, height=24)
        self.epoch_list.pack(fill=tk.Y, expand=False)
        self.epoch_list.bind("<<ListboxSelect>>", lambda e: self.update_selected_epoch_details())
        self.info_text = tk.Text(left, width=42, height=18, wrap="word")
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        self.right_pane.pack(fill=tk.BOTH, expand=True)
        graph_frame, image_frame = ttk.Frame(self.right_pane), ttk.Frame(self.right_pane)
        self.right_pane.add(graph_frame, weight=3)
        self.right_pane.add(image_frame, weight=4)

        self.fig = Figure(figsize=(12.5, 4.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.image_title_var = tk.StringVar(value="No image loaded")
        ttk.Label(image_frame, textvariable=self.image_title_var, anchor="w", font=("TkDefaultFont", 11, "bold")).pack(fill=tk.X, pady=(8, 2))
        self.image_canvas = tk.Canvas(image_frame, bg="black", highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        self.image_canvas.bind("<Configure>", lambda e: self.redraw_current_image())

        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    def choose_project_root(self) -> None:
        selected = filedialog.askdirectory(title="Choose project root")
        if selected:
            self.project_root_var.set(selected)
            self.project_root = Path(selected)
            self.update_selected_epoch_details()

    def on_run_changed(self) -> None:
        self.update_epoch_list()
        self.draw_plot()
        self.select_best_epoch()

    def get_current_records(self) -> List[Record]:
        selected = self.selected_run.get()
        if selected == "__ALL__":
            all_records: List[Record] = []
            for run_records in self.records_by_run.values():
                all_records.extend(run_records)
            return sorted(all_records, key=lambda x: (extract_stage_num(x.run_name), x.epoch, x.run_name))
        return self.records_by_run.get(selected, [])

    def update_epoch_list(self) -> None:
        self.epoch_list.delete(0, tk.END)
        for r in self.get_current_records():
            self.epoch_list.insert(tk.END, f"S{extract_stage_num(r.run_name)} | epoch {r.epoch}" if self.selected_run.get()=="__ALL__" else f"epoch {r.epoch}")

    def select_best_epoch(self) -> None:
        records = self.get_current_records()
        if not records:
            return
        best_idx = min(range(len(records)), key=lambda i: records[i].fid)
        self.epoch_list.selection_clear(0, tk.END)
        self.epoch_list.selection_set(best_idx)
        self.epoch_list.activate(best_idx)
        self.epoch_list.see(best_idx)
        self.update_selected_epoch_details()

    def get_selected_record(self) -> Optional[Record]:
        sel = self.epoch_list.curselection()
        records = self.get_current_records()
        if not sel or not records:
            return None
        idx = sel[0]
        return records[idx] if idx < len(records) else None

    def draw_plot(self) -> None:
        records = self.get_current_records()
        self.ax.clear()
        if not records:
            self.ax.set_title("No data")
            self.canvas.draw_idle()
            return
        selected = self.selected_run.get()
        if selected == "__ALL__":
            records = sorted(records, key=lambda x: (extract_stage_num(x.run_name), x.epoch, x.run_name))
            xs = list(range(len(records)))
            self.ax.plot(xs, [r.fid for r in records], marker="o")
            self.ax.set_title("All runs (continuous)")
            self.ax.set_xlabel("Stage / Epoch")
            self.ax.set_xticks(xs)
            self.ax.set_xticklabels([f"S{extract_stage_num(r.run_name)}-E{r.epoch}" for r in records], rotation=60, ha="right", fontsize=8)
            prev_stage = None
            for i, r in enumerate(records):
                stage = extract_stage_num(r.run_name)
                if prev_stage is not None and stage != prev_stage:
                    self.ax.axvline(i - 0.5, linestyle="--", alpha=0.3)
                prev_stage = stage
            best = min(records, key=lambda r: r.fid)
        else:
            self.ax.plot([r.epoch for r in records], [r.fid for r in records], marker="o")
            self.ax.set_title(selected)
            self.ax.set_xlabel("Epoch")
            best = min(records, key=lambda r: r.fid)
        self.ax.set_ylabel("FID")
        self.ax.grid(True, alpha=0.3)
        self.ax.text(1.02, 0.98, f"Best FID\nStage: {extract_stage_num(best.run_name)}\nEpoch: {best.epoch}\nFID: {best.fid:.4f}", transform=self.ax.transAxes, va="top", ha="left", fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9), clip_on=False)
        self.ax.text(1.02, 0.72, FID_GUIDE_TEXT, transform=self.ax.transAxes, va="top", ha="left", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9), clip_on=False)
        self.fig.tight_layout(rect=[0, 0, 0.82, 1])
        self.canvas.draw_idle()

    def on_plot_click(self, event) -> None:
        if event.xdata is None:
            return
        records = self.get_current_records()
        if not records:
            return
        if self.selected_run.get() == "__ALL__":
            records = sorted(records, key=lambda x: (extract_stage_num(x.run_name), x.epoch, x.run_name))
            idx = min(range(len(records)), key=lambda i: abs(i - event.xdata))
        else:
            idx = min(range(len(records)), key=lambda i: abs(records[i].epoch - event.xdata))
        self.epoch_list.selection_clear(0, tk.END)
        self.epoch_list.selection_set(idx)
        self.epoch_list.activate(idx)
        self.epoch_list.see(idx)
        self.update_selected_epoch_details()

    def update_selected_epoch_details(self) -> None:
        record = self.get_selected_record()
        if record is None:
            return
        lines = [f"Run: {record.run_name}", f"Stage: {extract_stage_num(record.run_name)}", f"Epoch: {record.epoch}", f"FID: {record.fid:.6f}", f"Phase: {record.phase}", f"Images: {record.num_images}", f"metrics.json: {record.metrics_json}"]
        project_root = Path(self.project_root_var.get()).expanduser() if self.project_root_var.get().strip() else None
        image_path = find_best_image(project_root, record) if project_root and project_root.exists() else None
        if image_path:
            self.current_record = record
            self.current_image_path = image_path
            lines.append(f"Matched image: {safe_rel(image_path, project_root)}")
            self.show_image(image_path, record)
            self.status_var.set(f"Loaded image for epoch {record.epoch}: {image_path}")
        else:
            self.current_record = None
            self.current_image_path = None
            self.image_title_var.set("No matching image found")
            self.current_image_tk = []
            self.current_display_panels = None
            self.image_canvas.delete("all")
            self.image_canvas.create_text(20, 20, anchor="nw", text="No matching image found", fill="white")
            lines.append("Matched image: not found" if project_root else "Matched image: project root not set")
            self.status_var.set(f"No image found for epoch {record.epoch}." if project_root else "Set project root to search for images.")
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert("1.0", "\n".join(lines))

    def find_companion_image(self, image_path: Path, keywords: list[str]) -> Optional[Path]:
        folder = image_path.parent
        stem = image_path.stem.lower()
        candidates = []
        for p in folder.iterdir():
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS or p == image_path:
                continue
            name = p.name.lower()
            if any(k in name for k in keywords):
                shared = 1 if any(tok in name for tok in re.findall(r"[a-z]+|\d+", stem)) else 0
                candidates.append((0 if shared else 1, len(name), p))
        return sorted(candidates)[0][2] if candidates else None

    def get_display_panels(self, image_path: Path, record: Record, img: Image.Image) -> list[tuple[str, Image.Image]]:
        path_text = str(image_path).lower()
        if ("debug" in path_text or image_path.stem.lower().startswith("debug_")) and img.width >= img.height * 2.4:
            third = img.width // 3
            return [
                ("Input", img.crop((0, 0, third, img.height))),
                ("Fake image", img.crop((third, 0, 2 * third, img.height))),
                ("Real image", img.crop((2 * third, 0, img.width, img.height))),
            ]
        fake_path = real_path = None
        if any(k in path_text for k in ["fake", "generated", "synth", "synthetic"]):
            fake_path = image_path
            real_path = self.find_companion_image(image_path, ["real", "real_b", "ground", "target"])
        elif any(k in path_text for k in ["real", "ground_truth", "groundtruth", "target"]):
            real_path = image_path
            fake_path = self.find_companion_image(image_path, ["fake", "generated", "synth", "synthetic"])
        panels = []
        if fake_path:
            try: panels.append(("Fake image", Image.open(fake_path).convert("RGB")))
            except Exception: pass
        if real_path:
            try: panels.append(("Real image", Image.open(real_path).convert("RGB")))
            except Exception: pass
        return panels if panels else [(self.classify_image_type(image_path, record), img)]

    def classify_image_type(self, image_path: Path, record: Record) -> str:
        path_text = str(image_path).lower()
        if any(k in path_text for k in ["fake", "generated", "synth", "synthetic"]):
            return "Fake image"
        if any(k in path_text for k in ["real", "ground_truth", "groundtruth", "target"]):
            return "Real image"
        if any(k in path_text for k in ["results/", "/results", "web/images"]):
            return "Fake image (likely)"
        return f"Image type unknown | phase={record.phase}" if record.phase else "Image type unknown"

    def show_image(self, image_path: Path, record: Record) -> None:
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            self.image_canvas.delete("all")
            self.image_canvas.create_text(20, 20, anchor="nw", text=f"Failed to open image:\n{e}", fill="white")
            self.current_image_tk = []
            self.current_display_panels = None
            return
        self.current_record = record
        self.current_display_panels = self.get_display_panels(image_path, record, img)
        self.image_title_var.set(" | ".join(label for label, _ in self.current_display_panels) + f" | {image_path.name}")
        self.redraw_current_image()

    def redraw_current_image(self) -> None:
        if not self.current_display_panels:
            return
        self.image_canvas.update_idletasks()
        canvas_w = max(self.image_canvas.winfo_width(), 300)
        canvas_h = max(self.image_canvas.winfo_height(), 220)
        gap = 20
        top_pad = 32
        panels = self.current_display_panels
        slot_w = max((canvas_w - 20 - gap * (len(panels)-1)) // len(panels), 60)
        slot_h = max(canvas_h - top_pad - 20, 60)
        self.image_canvas.delete("all")
        refs = []
        if self.current_record is not None:
            fid_box = f"FID: {self.current_record.fid:.4f}"
            self.image_canvas.create_text(
                canvas_w - 12, 12, text=fid_box, anchor="ne", fill="white",
                font=("TkDefaultFont", 10, "bold")
            )
        x = 10
        for label, pil_img in panels:
            disp = ImageOps.contain(pil_img, (slot_w, slot_h))
            tk_img = ImageTk.PhotoImage(disp)
            refs.append(tk_img)
            self.image_canvas.create_text(x + slot_w // 2, 14, text=label, fill="white", font=("TkDefaultFont", 10, "bold"))
            self.image_canvas.create_image(x + slot_w // 2, top_pad + slot_h // 2, image=tk_img, anchor="center")
            x += slot_w + gap
        self.current_image_tk = refs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FID vs epoch and show matching images.")
    parser.add_argument("--metrics-file", required=True)
    parser.add_argument("--project-root", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_file = Path(args.metrics_file).expanduser().resolve()
    if not metrics_file.exists():
        raise SystemExit(f"Metrics file not found: {metrics_file}")
    records = load_records(metrics_file)
    if not records:
        raise SystemExit("No valid records found in the metrics file.")
    root = tk.Tk()
    App(root=root, records_by_run=deduplicate_records(records), metrics_file=metrics_file, project_root=Path(args.project_root).expanduser().resolve() if args.project_root else None)
    root.mainloop()

if __name__ == "__main__":
    main()

"""
python Codes_Notebooks/Pix2Pix/fid_epoch_viewer.py \
  --metrics-file "checkpoints/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage23_COMPLETESyn/fid_history.jsonl" \
  --project-root "/home/ssy/Desktop/xray-gen-ai_Project"


"""