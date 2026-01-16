# src/xraygen/data/build_fiftyone_dataset.py

import os
import fiftyone as fo
import pandas as pd


def build_fiftyone_dataset(
    name: str = "xray-dataset",
    csv_path: str = "data/raw/metadata_sample.csv",
    img_root: str = "data/raw",
    overwrite: bool = False,
) -> fo.Dataset:
    """
    Build (or load) a FiftyOne dataset from a CSV with columns:
        - filename
        - has_contraband  (0 or 1)

    Args:
        name: name of the FiftyOne dataset
        csv_path: path to CSV file
        img_root: root folder where images live
        overwrite: if True, delete existing dataset with same name

    Returns:
        The FiftyOne Dataset object.
    """
    # -------------------------------
    # 1. Handle existing dataset
    # -------------------------------
    if name in fo.list_datasets():
        if overwrite:
            print(f"[FiftyOne] Deleting existing dataset '{name}'")
            fo.delete_dataset(name)
        else:
            print(f"[FiftyOne] Dataset '{name}' already exists, loading it")
            return fo.load_dataset(name)

    # -------------------------------
    # 2. Read CSV
    # -------------------------------
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"filename", "has_contraband"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    print(f"[FiftyOne] Building dataset '{name}' from {len(df)} rows")

    # -------------------------------
    # 3. Create dataset
    # -------------------------------
    dataset = fo.Dataset(name)
    dataset.default_classes = ["no_contraband", "contraband"]

    # -------------------------------
    # 4. Add samples
    # -------------------------------
    samples = []

    for _, row in df.iterrows():
        filename = row["filename"]
        has_contraband = int(row["has_contraband"])

        filepath = os.path.join(img_root, filename)

        if not os.path.exists(filepath):
            print(f"[WARN] Image file not found: {filepath}, skipping")
            continue

        sample = fo.Sample(filepath=filepath)

        label_str = "contraband" if has_contraband == 1 else "no_contraband"
        sample["has_contraband"] = fo.Classification(label=label_str)

        samples.append(sample)

    dataset.add_samples(samples)
    print(f"[FiftyOne] Added {len(samples)} samples to dataset '{name}'")

    return dataset
