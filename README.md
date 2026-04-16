# X-Ray Synthetic Image Generator

**Lightweight toolkit for generating and analysing synthetic X‑ray baggage imagery.**

Originally developed as a 20‑week internship project, this repository now contains utilities, notebooks and experiments for
training GANs, classifiers and performing explainability on security X‑ray datasets.

## Goals

- produce realistic synthetic X‑ray images using a variety of GAN architectures
- train a convolutional classifier to detect contraband objects
- visualise model decisions with Grad‑CAM and similar techniques
- compare real and generated data with metrics, visual reports and interactive tools

---

## Current Repository Layout

```
/home/ssy/Desktop/xray-gen-ai_Project/
├── xray-gen-ai/                      # Python package / helper scripts
│   ├── dataset_processing.py         # FiftyOne dataset builders & utilities
│   ├── verify_setup.py               # environment/GPUs/simulator health check
│   ├── pyproject.toml                # package metadata
│   └── README.md                     # (currently empty)
├── data/                             # managed by DVC
│   ├── raw/                          # source images + metadata CSVs
│   ├── interim/                      # preprocessing outputs, augmentation
│   └── processed/                    # final train/test splits
├── models/                           # model weights & checkpoints
│   ├── classifier/                   # classification network weights (DVC)
│   └── generator/                    # generator/discriminator checkpoints (DVC)
├── Codes_Notebooks/                  # exploratory and training notebooks
│   ├── CDGan/                        # custom DCGAN experiments
│   ├── ClassifierModels/             # classifier demos & evaluation
│   ├── Pix2Pix/                      # Pix2Pix dataset/prep & generation
│   ├── StyleGan/                     # StyleGAN2‑ADA helpers and training
│   ├── Stage0/ … Stage3/             # pipeline stage notebooks
│   └── …                             # other experimental notebooks
├── reports/                          # outputs for papers/presentations
│   ├── generated/                    # generated image grids
│   ├── gradcam_real/                 # heatmaps on real X‑rays
│   └── gradcam_generated/            # heatmaps on synthetic X‑rays
├── external/                         # third‑party codebases (pix2pix, stylegan2)
│   ├── pix2pix/
│   └── stylegan2-ada-pytorch/
├── checkpoints/                      # assorted GAN checkpoints (not Git-tracked)
├── results/                          # miscellaneous logs and evaluation results
├── datasets/                         # auxiliary dataset exports, e.g. _gen_real
├── dvc.yaml                          # DVC pipeline definition
├── dvc.lock                          # DVC lockfile
├── requirements.txt                  # Python dependencies
├── .gitignore                        # ignores large files & caches
└── README.md                         # this document
```

> ℹmany subdirectories (data, models, checkpoints) are tracked via DVC; see `dvc.yaml` for pipeline

### Key scripts in `xray-gen-ai/`

- `dataset_processing.py` – build FiftyOne datasets from Label Studio exports (classification + COCO detection)
- `verify_setup.py` – sanity check for Python version, CUDA availability and gVirtualXray installation

The rest of the work happens primarily inside the Jupyter notebooks under `Codes_Notebooks/`.

---

## Getting Started

### Setup environment

```bash
python -m venv .venv             # create venv
source .venv/bin/activate        # activate (Linux/Mac)
pip install -r requirements.txt  # install packages
```

---

## Workflow Overview

1. prepare/annotate data → `data/raw/` → `Codes_Notebooks/` (e.g. Pix2Pix prep) → `data/interim/`/`processed/`
2. train models via notebook scripts (`train_gan.py`, `train_stylegan.py`, `modeltraining.py`, etc.)
3. evaluate with classifier networks and Grad‑CAM utilities
4. save analysis reports under `reports/` and checkpoints under `models/`/`checkpoints/`

---

## Dependencies

See `requirements.txt` for a full list. Core libs include:
- PyTorch & torchvision
- FiftyOne (dataset inspection)
- TorchAMP, numpy, pandas, matplotlib, seaborn

External repos supply GAN training code (see `external/`).

---


##  Quick Start

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (for GPU training, optional but recommended)
- Git & DVC

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd xray-gen-ai
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull data from DVC:**
   ```bash
   dvc pull
   ```


---

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Core dependencies include:
- **PyTorch** – Deep learning framework
- **Torchvision** – Computer vision utilities
- **FiftyOne** – Interactive dataset exploration
- **Scikit-learn** – Metrics and utilities
- **Pandas** – Data handling
- **Matplotlib & Seaborn** – Visualization

---

