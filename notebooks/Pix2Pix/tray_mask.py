#!/usr/bin/env python3
from pathlib import Path
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
EMPTY_DIR = Path("data/interim/GAN/Empty")
OUT_PATH = EMPTY_DIR / "tray_mask.png"
SIZE = 1024
MAX_N = 50

# Preprocess params (initial auto mask)
CLOSE_K = 41  # fill holes/connect tray
USE_INVERT = False  # set True if your tray is darker than background

# GUI params
WINDOW = "Tray Mask Editor"
BRUSH_INIT = 15
OVERLAY_ALPHA = 0.35  # how strong the mask overlay is

# =========================
# Build initial mask
# =========================
def build_initial_mask() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      med: uint8 grayscale median image (H,W)
      m:   uint8 binary mask (H,W) values {0,255}
    """
    paths = sorted([p for p in EMPTY_DIR.glob("*.png")])[:MAX_N]
    if not paths:
        raise RuntimeError(f"No PNGs found in {EMPTY_DIR}")

    stack = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        if im.shape != (SIZE, SIZE):
            im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        stack.append(im)

    if not stack:
        raise RuntimeError("Could not read any images.")

    stack = np.stack(stack, axis=0)  # (N,H,W)
    med = np.median(stack, axis=0).astype(np.uint8)

    # Otsu threshold
    if not USE_INVERT:
        _, m = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, m = cv2.threshold(med, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Close holes
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # Force clean binary
    m = np.where(m > 127, 255, 0).astype(np.uint8)
    return med, m

# =========================
# GUI helpers
# =========================
def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def render_overlay(base_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    base_gray: (H,W) uint8
    mask:      (H,W) uint8 {0,255}
    Returns BGR overlay visualization.
    """
    base = to_bgr(base_gray)

    # Red overlay where mask is 255
    overlay = base.copy()
    red = np.zeros_like(base)
    red[:, :, 2] = 255  # BGR red channel

    m_bool = mask > 127
    overlay[m_bool] = cv2.addWeighted(base[m_bool], 1.0 - OVERLAY_ALPHA, red[m_bool], OVERLAY_ALPHA, 0)

    # Draw contour for clarity
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 255), 2)  # yellow outline

    # HUD
    return overlay

class EditorState:
    def __init__(self, base_gray: np.ndarray, mask_init: np.ndarray):
        self.base = base_gray
        self.mask_init = mask_init.copy()
        self.mask = mask_init.copy()
        self.brush = BRUSH_INIT
        self.drawing = False
        self.mode = None  # "add" or "erase"

    def reset(self):
        self.mask = self.mask_init.copy()

def mouse_cb(event, x, y, flags, state: EditorState):
    # Mouse wheel changes brush
    if event == cv2.EVENT_MOUSEWHEEL:
        # flags > 0 = forward, < 0 = backward
        if flags > 0:
            state.brush = min(200, state.brush + 2)
        else:
            state.brush = max(1, state.brush - 2)
        return

    # Start drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        state.drawing = True
        state.mode = "add"
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.drawing = True
        state.mode = "erase"

    # Stop drawing
    elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
        state.drawing = False
        state.mode = None

    # Draw while moving
    elif event == cv2.EVENT_MOUSEMOVE and state.drawing:
        if state.mode == "add":
            cv2.circle(state.mask, (x, y), state.brush, 255, -1)
        elif state.mode == "erase":
            cv2.circle(state.mask, (x, y), state.brush, 0, -1)

def main():
    med, mask0 = build_initial_mask()

    state = EditorState(med, mask0)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1100, 900)
    cv2.setMouseCallback(WINDOW, mouse_cb, state)

    print("\n=== Tray Mask GUI ===")
    print("Left-drag   : ADD tray (white)")
    print("Right-drag  : ERASE tray (black)")
    print("Mouse wheel : brush size")
    print("Keys: [s]=save  [r]=reset  [q]=quit")
    print("Saving to:", OUT_PATH)
    print("=====================\n")

    while True:
        vis = render_overlay(state.base, state.mask)

        # show brush size on screen
        cv2.putText(vis, f"brush={state.brush}px | s=save r=reset q=quit",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            state.reset()
            print("[reset] back to initial auto mask")
        elif key == ord('s'):
            OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            # Force binary clean
            out = np.where(state.mask > 127, 255, 0).astype(np.uint8)
            cv2.imwrite(str(OUT_PATH), out)
            print("[saved]", OUT_PATH)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()