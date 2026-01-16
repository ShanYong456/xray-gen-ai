from torch.utils.data import Dataset
from PIL import Image
import pandas as pd 
import os

class XRayDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["filename"])
        img = Image.open(img_path).convert("L") #greyscale
        label = int(row["has_contraband"])
        if self.transform:
            img = self.transform(img)
        return img, label