import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from xraygen.train.train_classifier import train_classifier as main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/classifier.yaml")
    args = parser.parse_args()
    main()