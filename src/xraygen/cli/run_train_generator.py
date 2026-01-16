import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xraygen.train.train_generator import train_gan


def main():
    parser = argparse.ArgumentParser(description="Train X-ray GAN generator")
    parser.add_argument("--csv", type=str, default="data/raw/metadata_sample.csv")
    parser.add_argument("--img_root", type=str, default="data/raw")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out_dir", type=str, default="models/generator")

    args = parser.parse_args()

    train_gan(
        csv_path = args.csv,
        img_root = args.img_root,
        nz = args.nz,
        batch_size = args.batch_size,
        num_epochs = args.epochs,
        lr = args.lr,
        out_dir = args.out_dir,
    )

if __name__ == "__main__":
    main()