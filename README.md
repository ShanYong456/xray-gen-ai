# X-Ray Synthetic Image Generator

This repository contains a research and automation pipeline for generating, filtering, evaluating, and inspecting synthetic X-ray baggage imagery. The project addresses a common limitation in security X-ray machine learning work: real annotated baggage data is difficult to collect, expensive to label, and often too limited for robust classifier development.

The current implementation focuses on tray scenes containing shampoo and blade objects. It uses a Pix2Pix-style image-to-image generation workflow to create X-ray-like outputs from structured object, tray, and mask inputs. Generated images are then evaluated through realism scoring, novelty checks, FID metrics, downstream CNN classifier validation, and Grad-CAM visual inspection.

The repository includes both current pipeline code and earlier experimental work. The maintained automation layer is centered on `src/xraygen/pipeline`, `dvc.yaml`, and `dvc_test/dvc.yaml`. The notebooks and older experiment folders remain available for reference, comparison, and future development.

## Project Objectives

The project is designed to support a complete synthetic-data workflow:

1. Prepare real X-ray tray and object data.
2. Train or reuse a generator checkpoint.
3. Generate synthetic baggage scenes.
4. Filter unrealistic, low-quality, or low-novelty samples.
5. Evaluate generator outputs with image-quality metrics.
6. Validate downstream classifier performance.
7. Inspect classifier behavior with Grad-CAM.
8. Promote generator and classifier artifacts only when quality gates pass.

The emphasis is not only on producing synthetic images, but on validating whether those images are useful for model development. The pipeline combines reproducible DVC stages, quantitative evaluation, classifier-based checks, and visual explainability to reduce the risk of accepting synthetic samples that look plausible but do not improve the task.

## Current Capabilities

- Build aligned Pix2Pix datasets from X-ray tray images, object masks, tray masks, and annotation exports.
- Generate synthetic shampoo/blade tray scenes using trained Pix2Pix checkpoints.
- Score generated images against real reference data using Mahalanobis-style realism checks.
- Compare generated images with nearest real examples to identify low-novelty outputs.
- Split generated images into accepted and rejected sets.
- Evaluate generator quality with FID on train and test phases.
- Validate CNN classifiers for spatial-overlap and threat-related classification tasks.
- Export Grad-CAM visualizations for generated-image classifier inspection.
- Track generation, filtering, evaluation, and promotion through DVC pipelines.

## Recommended Entry Points

For a first pass through the project, start with the current pipeline files before reviewing the notebooks:

- `dvc.yaml` defines the main end-to-end candidate generation, evaluation, and promotion pipeline.
- `dvc_test/dvc.yaml` defines a smaller smoke-test pipeline for safer iteration.
- `src/xraygen/pipeline/` contains reusable scripts used by the DVC stages and dashboards.
- `Codes_Notebooks/Pix2Pix/` contains Pix2Pix dataset, generation, scoring, and evaluation utilities.
- `Codes_Notebooks/SimpleCNN/` contains the current CNN validation workflow.

The repository is a research codebase with an automation layer, not a minimal Python package. Some notebooks and older scripts reflect intermediate experiments and may require local path checks before use.

## Repository Structure

```text
.
├── src/xraygen/              # reusable project Python code
├── Codes_Notebooks/          # notebooks and experiment scripts
├── external/                 # third-party GAN implementations
├── label_studio/             # annotation and dataset-processing helpers
├── data/                     # raw, interim, processed, and label data
├── datasets/                 # Pix2Pix datasets, masks, generated datasets
├── checkpoints/              # GAN checkpoints
├── models/                   # classifier and promoted model artifacts
├── reports/                  # validation reports, DVC metrics, Grad-CAM outputs
├── results/                  # generated reference outputs and experiment results
├── dvc.yaml                  # main pipeline
├── dvc_test/                 # smaller DVC test pipeline
└── requirements.txt          # pinned Python environment
```

Large datasets, generated images, checkpoints, and reports should be managed with DVC or kept as local artifacts rather than committed directly to Git.

## Main Components

### `src/xraygen`

This directory contains the most reusable project code.

- `src/xraygen/explain/gradcam.py` provides reusable Grad-CAM support.
- `src/xraygen/pipeline/filter_generated.py` filters generated images into accepted and rejected sets.
- `src/xraygen/pipeline/evaluate_candidate.py` evaluates whether a candidate run passes the configured quality gates.
- `src/xraygen/pipeline/promote_if_passed.py` writes production registry information and copies passing artifacts.
- `src/xraygen/pipeline/prepare_dvc_test_pix2pix_data.py` prepares a DVC test dataset workspace.
- `src/xraygen/pipeline/bootstrap_pix2pix_checkpoint.py` initializes a test checkpoint from an existing Pix2Pix checkpoint.
- `src/xraygen/pipeline/dvc_test_unified_dashboard.py` serves the local DVC test dashboard.
- `src/xraygen/pipeline/cnn_gradcam_dashboard.py` serves the generated-image CNN and Grad-CAM dashboard.

### `Codes_Notebooks`

This directory contains the main research notebooks and experiment scripts.

- `Codes_Notebooks/Pix2Pix/` contains dataset builders, mask extraction utilities, generation scripts, FID evaluation, Mahalanobis scoring, and result viewers.
- `Codes_Notebooks/SimpleCNN/` contains the current classifier workflow and two-model validation.
- `Codes_Notebooks/Stage0` through `Codes_Notebooks/Stage3` contain staged classifier experiments.
- `Codes_Notebooks/CDGan/` contains an earlier conditional/DCGAN workflow.
- `Codes_Notebooks/StyleGan/` contains StyleGAN preprocessing, patching, training, and generation helpers.
- `Codes_Notebooks/ClassifierModels/` contains earlier classifier experiments and testing scripts.

### `external`

This directory contains third-party GAN implementations used by the project.

- `external/pix2pix/` is used by the current Pix2Pix training and DVC smoke-test flow.
- `external/stylegan2-ada-pytorch/` is retained for StyleGAN2-ADA experiments and comparison.

### Data And Artifact Directories

- `data/raw` stores source object/tray data and annotation exports.
- `data/interim` stores intermediate preprocessing outputs.
- `data/processed` stores classifier-ready datasets.
- `data/labels` stores train, validation, and test JSON labels.
- `datasets` stores Pix2Pix aligned datasets, matched masks, generated DVC datasets, and test workspaces.
- `checkpoints` stores generator and discriminator checkpoints.
- `models/classifier` stores classifier checkpoints and related artifacts.
- `models/production` and `models/production_test` store promoted artifact records.
- `reports` stores DVC reports, validation summaries, Grad-CAM outputs, and evaluation artifacts.
- `results` stores generated examples and experiment outputs.

## Environment Setup

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The environment includes dependencies for model training, evaluation, notebooks, data processing, and local dashboards. Major dependency groups include:

- PyTorch and CUDA packages for model training and inference.
- DVC for data and pipeline tracking.
- OpenCV, Pillow, numpy, pandas, and pyarrow for image and dataset processing.
- matplotlib, seaborn, Plotly, JupyterLab, and ipywidgets for visualization and notebook workflows.
- FiftyOne and Label Studio-related tools for dataset exploration and annotation processing.
- MLflow, Flask, FastAPI, and related packages for dashboards and local services.
- `gvxr` and `k3d` for simulation and 3D-related experimentation.

If data and model artifacts are stored in a DVC remote, pull them after installing dependencies:

```bash
dvc pull
```

For local imports from `src/xraygen`, run commands from the repository root. If a script cannot find the package, set:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

## Main DVC Pipeline

The top-level `dvc.yaml` is the main candidate evaluation and promotion pipeline. It is the best single file for understanding the current automated workflow.

Pipeline stages:

1. `score_real_reference`
   Scores real test images against real training images to establish a real-data comparison baseline.

2. `generate_real_ab_reference`
   Runs the generator on real aligned A/B inputs for reference output generation.

3. `generate_synthetic`
   Generates synthetic combo scenes using shampoo source images, tray masks, blade masks, and the current Stage24 Pix2Pix checkpoint.

4. `filter_synthetic`
   Reads generated-image realism and novelty outputs, then separates accepted and rejected samples.

5. `evaluate_generator_fid_train`
   Computes FID metrics for the train phase.

6. `evaluate_generator_fid_test`
   Computes FID metrics for the test phase.

7. `validate_classifier`
   Runs the two-model CNN validation workflow from `Codes_Notebooks/SimpleCNN`.

8. `evaluate_candidate`
   Combines filter results, FID metrics, and classifier validation into a single quality-gate report.

9. `promote_candidate`
   Copies generator/classifier artifacts and writes the production model registry when the candidate passes the configured gates.

Run the full pipeline:

```bash
dvc repro
```

Run a single stage:

```bash
dvc repro evaluate_candidate
```

Show tracked metrics:

```bash
dvc metrics show
```

Important main-pipeline outputs:

- `datasets/_dvc_generated_combo/generated`
- `datasets/_dvc_generated_combo/accepted`
- `datasets/_dvc_generated_combo/rejected`
- `reports/dvc/generated_combo_summary.json`
- `reports/dvc/filter_report.json`
- `reports/dvc/candidate_eval.json`
- `reports/dvc/fid_eval_runs/`
- `reports/validation/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models/`
- `models/production/model_registry.json`

## DVC Test Pipeline

The test pipeline in `dvc_test/dvc.yaml` provides a smaller, isolated workflow for iteration. It follows the same general structure as the main pipeline, but writes to `_dvc_test` and `production_test` paths so that test runs do not overwrite main generated outputs or production registry files.

Run it from the `dvc_test` folder:

```bash
cd dvc_test
dvc repro
```

Important test-pipeline outputs:

- `datasets/_dvc_test/SHAMPOOBLADEWITHTRAY_COMPLETE`
- `checkpoints/Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST`
- `datasets/_dvc_test_generated_combo/generated`
- `datasets/_dvc_test_generated_combo/accepted`
- `datasets/_dvc_test_generated_combo/rejected`
- `reports/dvc_test/preprocess_dataset_report.json`
- `reports/dvc_test/generated_combo_summary.json`
- `reports/dvc_test/filter_report.json`
- `reports/dvc_test/candidate_eval.json`
- `models/production_test/model_registry.json`

Use this pipeline when changing generation settings, filtering thresholds, checkpoint bootstrapping, or dashboard behavior.

## Local Dashboards

### DVC Test Dashboard

The DVC test dashboard monitors and controls the local DVC test workflow.

```bash
python src/xraygen/pipeline/dvc_test_unified_dashboard.py
```

Open:

```text
http://127.0.0.1:8770
```

### CNN Grad-CAM Dashboard

The CNN Grad-CAM dashboard evaluates generated images with the classifier and exports Grad-CAM visualizations.

```bash
python src/xraygen/pipeline/cnn_gradcam_dashboard.py
```

Open:

```text
http://127.0.0.1:8750
```

Default inputs:

- Images: `datasets/_dvc_generated_combo/generated`
- Outputs: `reports/generated_cnn_gradcam_dashboard`
- Model: `models/classifier/SHAMPOOBLADEINTRAY_COMPLETEV2/gray_multihead_itemmask_optional_slow/checkpoints/train_best_checkpoint.pt`

## Common Workflows

### Run The Main Candidate Flow

```bash
dvc pull
dvc repro
dvc metrics show
```

Review:

- `reports/dvc/candidate_eval.json`
- `reports/dvc/filter_report.json`
- `datasets/_dvc_generated_combo/accepted`
- `datasets/_dvc_generated_combo/rejected`
- `models/production/model_registry.json`

### Iterate On The Test Flow

```bash
cd dvc_test
dvc repro
```

This workflow is recommended for testing pipeline changes before running the main `dvc.yaml`.

### Validate The CNN Classifiers

```bash
cd Codes_Notebooks/SimpleCNN
python validate_two_separate_models.py
```

Outputs are written under:

```text
reports/validation/SHAMPOOBLADEINTRAY_COMPLETEV2_two_models
```

### Check Label Studio And Environment Setup

```bash
python label_studio/verify_setup.py
```

Use this check before annotation processing or dataset preparation work.

## Important Pix2Pix Scripts

Most generator-related helper scripts are located in `Codes_Notebooks/Pix2Pix/`.

- `build_pix2pix_dataset.py`
  Builds aligned datasets for Pix2Pix.

- `build_pix2pix_dataset (no_resize).py`
  Builds aligned datasets while preserving image-size assumptions.

- `extract_all_tray_mask.py`, `tray_mask.py`, `match_empty_trays.py`
  Prepare tray masks and match tray assets.

- `generate_pix2pix.py`
  Older generation entry point.

- `generate_pix2pixV2.py`
  Updated generation entry point.

- `generate_pix2pixV2_MAHADIST.py`
  Main generation and scoring script used by the DVC pipeline.

- `generate_realscore_pix2pix_MAHADIST.py`
  Scores real images against real reference data.

- `fid_eval.py`
  Runs FID evaluation for Pix2Pix checkpoints.

- `realdataset_mahalanobis.py`
  Computes real-dataset Mahalanobis-style reference statistics.

- `mahal_scene_viewer.py`
  Inspects generated scenes and Mahalanobis-related outputs.

- `fid_epoch_viewer.py`
  Inspects FID results across epochs or runs.

## Classifier And Explainability

The classifier workflow is used both for the target classification task and as a quality check for generated data. The current DVC pipeline expects two classifier models:

- one classifier for spatial-overlap / isolated-object classification
- one classifier for threat / contraband / non-contraband classification

Main classifier locations:

- `Codes_Notebooks/SimpleCNN/`
- `Codes_Notebooks/Stage0/`
- `Codes_Notebooks/Stage1/`
- `Codes_Notebooks/Stage2/`
- `Codes_Notebooks/Stage3/`
- `Codes_Notebooks/ClassifierModels/`

Grad-CAM support:

- `src/xraygen/explain/gradcam.py`
- `Codes_Notebooks/Pix2Pix/evaluate_generated_cnn_gradcam.py`
- `src/xraygen/pipeline/cnn_gradcam_dashboard.py`

## Earlier GAN Experiments

Earlier GAN experiments are retained for comparison, reference, and possible reuse.

- `Codes_Notebooks/CDGan/` contains an earlier conditional/DCGAN workflow.
- `Codes_Notebooks/StyleGan/` contains local StyleGAN helper scripts.
- `external/stylegan2-ada-pytorch/` contains the StyleGAN2-ADA implementation and metrics tools.

Before running older experiments, review paths and artifact assumptions inside the scripts or notebooks. Some files were written during active experimentation and may assume a specific local folder layout.

## Development Notes

- The most reliable current path is the Pix2Pix + CNN + DVC workflow.
- The DVC test pipeline should be used for pipeline changes before running the main pipeline.
- Notebooks remain useful for exploration, but not all notebooks are maintained equally.
- New reusable logic should generally be placed under `src/xraygen/`.
- New generated artifacts should be written under `datasets`, `reports`, `results`, `models`, or `checkpoints` depending on artifact type.
- Important or expensive-to-reproduce artifacts should be tracked with DVC.

## Quick Reference

- Current pipeline: `dvc.yaml`
- Safer test run: `dvc_test/dvc.yaml`
- Maintained helper scripts: `src/xraygen/pipeline`
- Generator utilities: `Codes_Notebooks/Pix2Pix`
- Classifier validation: `Codes_Notebooks/SimpleCNN`
- Grad-CAM support: `src/xraygen/explain` and `src/xraygen/pipeline/cnn_gradcam_dashboard.py`
- Annotation and data-preparation helpers: `label_studio`
- Earlier GAN experiments: `Codes_Notebooks/CDGan`, `Codes_Notebooks/StyleGan`, `external/stylegan2-ada-pytorch`
