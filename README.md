# X-Ray Synthetic Image Generator (Internship Project)

This project is a 20-week internship pipeline to **design, train, and evaluate** an AI system that generates synthetic security X-ray images and explains model decisions.

The core goals are:

- Generate realistic X-ray images of baggage/trays (GAN-based generator)
- Train a CNN classifier to detect contraband-like items
- Use **Grad-CAM** to visualize where the classifier is "looking"
- Compare **real vs. generated** images using both metrics and explainability

---

## Project Structure

```text
xray-gen-ai/
├── src/
│   └── xraygen/
│       ├── data/
│       │   ├── dataset.py              # XRayDataset (CSV + images)
│       │   └── build_fiftyone_dataset.py
│       ├── models/
│       │   ├── classifier.py           # CNN classifier
│       │   └── generator.py            # DCGAN-style generator + discriminator
│       ├── train/
│       │   ├── train_classifier.py     # Train classifier
│       │   └── train_generator.py      # Train GAN
│       ├── explain/
│       │   └── gradcam.py              # Grad-CAM implementation
│       ├── eval/
│       │   └── eval_generator.py       # Step 2.0: generator + classifier + Grad-CAM
│       └── cli/
│           ├── make_fiftyone.py        # Build & launch FiftyOne dataset
│           └── run_train_generator.py  # CLI wrapper for GAN training
├── data/
│   ├── raw/        # Original X-ray images + metadata CSV (DVC / local only)
│   ├── interim/
│   └── processed/
├── models/
│   ├── classifier/ # Trained classifier weights (ignored by git, tracked by DVC)
│   └── generator/  # Trained generator weights (ignored by git, tracked by DVC)
├── notebooks/
│   ├── 01_eda.ipynb                    # EDA + dataset sanity check
│   └── 02_label_check_gradcam.ipynb    # Grad-CAM on real X-rays
├── reports/
│   ├── generated/                      # Grids of generated samples
│   ├── gradcam_real/                   # Grad-CAM on real X-rays
│   └── gradcam_generated/              # Grad-CAM on generated X-rays
├── dvc.yaml                            # DVC pipeline definition
├── .gitignore
└── README.md
