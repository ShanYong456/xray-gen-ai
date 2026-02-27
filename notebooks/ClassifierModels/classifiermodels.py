import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --------------------------------
# IMPORT YOUR MODEL DEFINITIONS
# (edit these imports to match your repo)
# --------------------------------
from notebooks.Stage1 import SimpleCNN_GAP
# from notebooks.Stage2 import YourStage2Model
# from notebooks.Stage3 import YourStage3Model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_PATH = "sample.png"
IMAGE_SIZE = 256

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --------------------------------
# EDIT THIS LIST ONLY
# --------------------------------
MODEL_SPECS = [
    {
        "name": "Stage1",
        "ckpt": "../../models/classifier/Stage1/Stage1.pt",
        "build": lambda: SimpleCNN_GAP(num_classes=2),
        "class_names": {0: "stage0", 1: "stage1"},
    },
    {
        "name": "Stage2",
        "ckpt": "../../models/classifier/Stage2/Stage2.pt",
        "build": lambda: SimpleCNN_GAP(num_classes=2),  # change if different
        "class_names": {0: "stage0", 1: "stage1"},
    },
    {
        "name": "Stage3",
        "ckpt": "../../models/classifier/Stage3/Stage3.pt",
        "build": lambda: SimpleCNN_GAP(num_classes=2),  # change if different
        "class_names": {0: "stage0", 1: "stage1"},
    },
]


def load_model(build_fn, ckpt_path: str):
    model = build_fn()
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    # supports {"model_state_dict": ...} or raw state_dict
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # strip "module." if saved with DataParallel
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def predict_one(model, x, class_names: dict[int, str]):
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    pred_id = int(pred.item())
    conf = float(conf.item())
    pred_name = class_names.get(pred_id, str(pred_id))
    return pred_id, pred_name, conf


def main():
    img = Image.open(IMAGE_PATH).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    print("\n===== PREDICTIONS (1 image → 3 models) =====")
    for spec in MODEL_SPECS:
        model = load_model(spec["build"], spec["ckpt"])
        pred_id, pred_name, conf = predict_one(model, x, spec["class_names"])

        print(f"{spec['name']}: {pred_name} (class {pred_id})  conf={conf:.4f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
