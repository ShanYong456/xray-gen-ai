import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

from xraygen.data.dataset import XRayDataset
from xraygen.models.classifier import XRayClassifier

import json

#LOAD THE EXPORTED DATASET

with open("/home/ssy/Desktop/data_preprocessing/exports/xray_ls_cls_plus_coco_full/metadata.json") as f:
    manifest = json.load(f)

labels = manifest["labels"]
samples = manifest["samples"]

print("Num labels:", len(labels))
print("Num samples:", len(samples))


def train_classifier(
        csv_path="data/raw/metadata_sample.csv",
        img_root="data/raw",
        num_epochs=5,
        batch_size=4,
        lr=0.001,
        model_out_path="models/classifier/model.pt",
        val_split=0.2,
):
    
    # Device Setup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Transforms and Dataset

    transform = T.Compose([
        T.Resize((256, 256)), # make sure this matches your model's expectation
        T.ToTensor(),  # grayscale -> [1, H, W], values in [0,1]
    ])

    full_dataset = XRayDataset(csv_path, img_root, transform=transform)
    dataset_size = len(full_dataset)
    print(f"Total samples: {dataset_size}")

    if dataset_size == 0:
        raise ValueError("Dataset is empty! Check your csv and image paths.")
    

    # Train/val split

    val_size = max(1, int(val_split * dataset_size)) if dataset_size > 1 else 0
    train_size = dataset_size - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_dataset, val_dataset = full_dataset, None
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset is not None:
        print(f"Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    
    # Model, Loss, Optimizer

    model = XRayClassifier(num_classes=2).to(device) # 0 = contraband, 1 = contraband
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total 
        train_acc = correct / total

        # validation
        if val_loader is not None:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * images.size(0)
                    _, preds = outputs.max(1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total

            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} "
                f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )
        else:
            print(
                f"Epoch [{epoch}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f}"
            )
    
    # Save model

    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved classifier to: {model_out_path}")


def main():
    train_classifier()


if __name__ == "__main__":
    main()








