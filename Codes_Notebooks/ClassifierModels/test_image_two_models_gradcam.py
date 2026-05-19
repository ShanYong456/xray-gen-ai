import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "../../results/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage23_COMPLETESyn/test_latest/images_fake/deb1f5bb-2026-03-03_15-40-15-784_te_000095_fake_B.png"

# Two separate model checkpoints
SPATIAL_MODEL_PATH = "../../models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/spatial/checkpoints/train_best.pt"
THREAT_MODEL_PATH  = "../../models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/threat/checkpoints/train_best.pt"

OUT_DIR = Path("../../reports/single_image_test/two_models_gradcam")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 1024
GRAY_MEAN = (0.5,)
GRAY_STD = (0.25,)

SPATIAL_CLASSES = ["isolated", "overlap"]
THREAT_CLASSES = ["non_contraband", "contraband"]

# ============================================================
# TRANSFORM
# Must match your training transform
# ============================================================
transform = T.Compose([
    T.Grayscale(num_output_channels=1),
    T.Resize(int(IMAGE_SIZE * 1.10)),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(GRAY_MEAN, GRAY_STD),
])

# ============================================================
# SINGLE-TASK MODEL
# Same feature extractor as your previous CNN, but one binary head only
# ============================================================
class SimpleCNN_SingleTask(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ============================================================
# IMAGE HELPERS
# ============================================================
def make_black_border_white(pil_img, threshold=15):
    img = pil_img.convert("L")
    arr = np.array(img)
    border_mask = arr <= threshold
    arr[border_mask] = 255
    return Image.fromarray(arr).convert("L")


def crop_white_border(pil_img, threshold=252, margin=0):
    img = pil_img.convert("L")
    arr = np.array(img)
    mask = arr < threshold

    if not mask.any():
        return pil_img

    ys, xs = np.where(mask)
    left = max(xs.min() - margin, 0)
    right = min(xs.max() + margin, pil_img.size[0] - 1)
    top = max(ys.min() - margin, 0)
    bottom = min(ys.max() + margin, pil_img.size[1] - 1)

    return pil_img.crop((left, top, right + 1, bottom + 1))


def prepare_image(image_path):
    img_raw = Image.open(image_path)
    print("Original image mode:", img_raw.mode)
    print("Original image size:", img_raw.size)

    img = make_black_border_white(img_raw, threshold=15)
    img = crop_white_border(img, threshold=252, margin=0)

    print("Processed display size:", img.size)
    return img


def tensor_to_display_image(x_tensor):
    """
    Convert normalized 1x1xHxW tensor back to uint8 RGB for Grad-CAM overlay.
    """
    x = x_tensor.detach().cpu().squeeze(0).squeeze(0).numpy()
    x = (x * GRAY_STD[0]) + GRAY_MEAN[0]
    x = np.clip(x, 0, 1)
    img_u8 = (x * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    return img_rgb

# ============================================================
# CHECKPOINT LOADING
# ============================================================
def clean_state_dict(state):
    if isinstance(state, dict):
        for key in ["model_state", "model_state_dict", "state_dict"]:
            if key in state:
                state = state[key]
                break

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    return state


def load_single_task_model(model_path):
    model = SimpleCNN_SingleTask(in_channels=1, num_classes=2).to(DEVICE)

    try:
        state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=DEVICE)

    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()

    print("Loaded model:", model_path)
    return model

# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam, logits.detach(), probs.detach()


def make_gradcam_overlay(display_rgb, cam, alpha=0.45):
    heatmap = (cam * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def run_model_with_gradcam(model, x, class_names, task_name, save_dir):
    # Last conv block before GAP. This works with the model defined above.
    target_layer = model.features[-4]  # final Conv2d(256, 512, ...)
    gradcam = GradCAM(model, target_layer)

    cam, logits, probs_t = gradcam(x, class_idx=None)
    gradcam.remove_hooks()

    probs = probs_t.squeeze(0).cpu().numpy()
    pred_id = int(np.argmax(probs))
    pred_name = class_names[pred_id]

    display_rgb = tensor_to_display_image(x)
    overlay = make_gradcam_overlay(display_rgb, cam)

    save_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = save_dir / f"{task_name}_gradcam_{pred_name}.png"
    raw_path = save_dir / f"{task_name}_input.png"

    Image.fromarray(display_rgb).save(raw_path)
    Image.fromarray(overlay).save(overlay_path)

    print(f"\n===== {task_name.upper()} MODEL =====")
    print(f"Prediction: {pred_name} (class {pred_id})")
    for i, cname in enumerate(class_names):
        print(f"  {cname}: {probs[i]:.4f}")
    print("Grad-CAM saved to:", overlay_path)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(display_rgb, cmap="gray")
    plt.title(f"Input - {task_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM: {task_name} → {pred_name}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    return {
        "task": task_name,
        "prediction": pred_name,
        "class_id": pred_id,
        "probabilities": probs,
        "gradcam_path": str(overlay_path),
    }

# ============================================================
# RUN SINGLE IMAGE TEST
# ============================================================
spatial_model = load_single_task_model(SPATIAL_MODEL_PATH)
threat_model = load_single_task_model(THREAT_MODEL_PATH)

img = prepare_image(IMAGE_PATH)
x = transform(img).unsqueeze(0).to(DEVICE)

print("Input tensor shape:", tuple(x.shape))
print("Input tensor min/max:", x.min().item(), x.max().item())

# Display processed input
plt.figure(figsize=(7, 5))
plt.imshow(img, cmap="gray", vmin=0, vmax=255)
plt.axis("off")
plt.title("Processed image input")
plt.show()

spatial_result = run_model_with_gradcam(
    spatial_model,
    x,
    SPATIAL_CLASSES,
    task_name="spatial_overlap_isolated",
    save_dir=OUT_DIR,
)

threat_result = run_model_with_gradcam(
    threat_model,
    x,
    THREAT_CLASSES,
    task_name="threat_contraband_noncontraband",
    save_dir=OUT_DIR,
)

print("\n===== FINAL TWO-MODEL RESULT =====")
print("Image:", IMAGE_PATH)
print("Spatial prediction:", spatial_result["prediction"])
print("Threat prediction:", threat_result["prediction"])
print("Outputs saved in:", OUT_DIR)
