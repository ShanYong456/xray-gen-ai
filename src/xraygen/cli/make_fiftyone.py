# src/xraygen/cli/make_fiftyone.py

import argparse

import fiftyone as fo

from xraygen.data.build_fiftyone_dataset import build_fiftyone_dataset


def main():
    parser = argparse.ArgumentParser(description="Build and launch FiftyOne dataset viewer")
    parser.add_argument(
        "--name",
        type=str,
        default="xray-demo",
        help="Name of the FiftyOne dataset",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/metadata_sample.csv",
        help="Path to CSV with filename + has_contraband",
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default="data/raw",
        help="Root directory where image files are stored",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset with same name",
    )

    args = parser.parse_args()

    dataset = build_fiftyone_dataset(
        name=args.name,
        csv_path=args.csv,
        img_root=args.img_root,
        overwrite=args.overwrite,
    )

    # Launch the FiftyOne app
    session = fo.launch_app(dataset)
    print("[FiftyOne] App launched. Close the window or hit Ctrl+C to stop.")
    session.wait()


if __name__ == "__main__":
    main()
