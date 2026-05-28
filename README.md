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

## File Purpose Guide

This section explains the purpose of the main project-owned files. It intentionally excludes generated caches such as `__pycache__`, local virtual environments such as `.venv`, large binary model weights, and most vendored third-party files under `external/` and `label_studio/label-studio-ml-backend/`.

### Root Files

| File | Purpose |
| --- | --- |
| `README.md` | Main project documentation, setup guide, workflow guide, and file map. |
| `requirements.txt` | Python dependencies for training, evaluation, dashboards, DVC, image processing, and notebooks. |
| `dvc.yaml` | Main reproducible DVC pipeline for generation, filtering, FID evaluation, classifier validation, candidate evaluation, and promotion. |
| `dvc.lock` | Locked DVC dependency and output state for the main pipeline. |
| `.dvcignore` | Tells DVC which files or folders to ignore when tracking data and artifacts. |
| `.gitignore` | Tells Git which local artifacts, caches, datasets, and generated outputs should not be committed. |
| `.gitmodules` | Records Git submodule configuration, especially for external model repositories. |
| `.python-version` | Records the local Python version expected by the project environment. |
| `LICENSE` | Project license file. |
| `allfiles.txt` | Local inventory/reference list of repository files. |
| `cuda-keyring_1.1-1_all.deb`, `cuda-keyring_1.1-1_all.deb.1` | Local CUDA package installer files used for GPU/CUDA environment setup. |

### Reusable Project Package: `src/xraygen`

| File | Purpose |
| --- | --- |
| `src/xraygen/__init__.py` | Marks `xraygen` as an importable Python package. |
| `src/xraygen/explain/__init__.py` | Marks the explainability module as importable. |
| `src/xraygen/explain/gradcam.py` | Reusable Grad-CAM implementation for visualizing CNN attention on X-ray images. |
| `src/xraygen/pipeline/__init__.py` | Marks the pipeline scripts as an importable module. |
| `src/xraygen/pipeline/filter_generated.py` | Filters generated synthetic images into accepted and rejected sets using realism and novelty metrics. |
| `src/xraygen/pipeline/evaluate_candidate.py` | Combines filter results, FID metrics, and classifier validation metrics into one pass/fail quality-gate report. |
| `src/xraygen/pipeline/promote_if_passed.py` | Promotes generator/classifier artifacts into a production registry only when the candidate evaluation passes. |
| `src/xraygen/pipeline/prepare_dvc_test_pix2pix_data.py` | Builds a small isolated Pix2Pix dataset workspace for safer DVC test runs. |
| `src/xraygen/pipeline/bootstrap_pix2pix_checkpoint.py` | Copies or initializes the checkpoint used by the DVC test pipeline. |
| `src/xraygen/pipeline/dvc_test_unified_dashboard.py` | Local web dashboard for running and monitoring the DVC test workflow. |
| `src/xraygen/pipeline/cnn_gradcam_dashboard.py` | Local web dashboard for classifier inference and Grad-CAM inspection on generated images. |

### Main Pix2Pix And Synthetic-Data Scripts: `Codes_Notebooks/Pix2Pix`

| File | Purpose |
| --- | --- |
| `build_pix2pix_dataset.py` | Builds aligned Pix2Pix A/B datasets from images, masks, and annotation data. |
| `build_pix2pix_dataset (no_resize).py` | Dataset builder variant that preserves original sizing assumptions instead of resizing. |
| `complete_workflow.ipynb` | Notebook version of the broader Pix2Pix workflow for exploration and demonstration. |
| `empty_tray.py` | Utilities for working with empty tray images used as generation backgrounds or references. |
| `evaluate_generated_cnn_gradcam.py` | Runs CNN validation and Grad-CAM export on generated images. |
| `extract_all_tray_mask.py` | Extracts tray masks from available tray images for later Pix2Pix conditioning and scene placement. |
| `fid_epoch_viewer.py` | Viewer/helper for comparing FID metrics across checkpoints, epochs, or evaluation runs. |
| `fid_eval.py` | Computes FID metrics for Pix2Pix-generated outputs against real reference images. |
| `filterrawdata.py` | Filters or cleans raw input data before dataset building. |
| `generate_pix2pix.py` | Earlier Pix2Pix generation entry point retained for reference. |
| `generate_pix2pixV2.py` | Updated Pix2Pix generation script with newer generation options. |
| `generate_pix2pixV2_FID.py` | Generation variant focused on producing outputs for FID evaluation. |
| `generate_pix2pixV2_MAHADIST.py` | Main DVC-used generation script; creates synthetic scenes and scores realism/novelty with Mahalanobis-style metrics. |
| `generate_pix2pixV2_NIQE.py` | Generation/evaluation variant that includes NIQE-style image-quality scoring. |
| `generate_random_masks.py` | Creates random mask inputs for synthetic scene generation experiments. |
| `generate_realscore_pix2pix_MAHADIST.py` | Scores real images against real reference data to establish a baseline for realism filtering. |
| `mahal_scene_viewer.py` | Viewer for inspecting generated scenes and Mahalanobis-related scoring outputs. |
| `match_empty_trays.py` | Matches empty tray assets to dataset scenes or mask sets. |
| `pix2pix_object_library.py` | Shared object, mask, and placement helpers used by Pix2Pix generation scripts. |
| `realdataset_mahalanobis.py` | Computes Mahalanobis-style statistics for real datasets. |
| `tray_mask.py` | Tray mask extraction and processing utilities. |

### Classifier And Validation Workflows

| File | Purpose |
| --- | --- |
| `Codes_Notebooks/SimpleCNN/1_preprocessing.ipynb` | Prepares images, labels, and metadata for the SimpleCNN classifier workflow. |
| `Codes_Notebooks/SimpleCNN/2_freeze_dataset.ipynb` | Freezes processed classifier datasets for repeatable training and validation. |
| `Codes_Notebooks/SimpleCNN/3_train_model.ipynb` | Trains the SimpleCNN classifier models. |
| `Codes_Notebooks/SimpleCNN/4_validate_model.ipynb` | Validates trained classifiers and produces evaluation artifacts. |
| `Codes_Notebooks/SimpleCNN/train_single_task_classifier.ipynb` | Trains a single-task classifier variant. |
| `Codes_Notebooks/SimpleCNN/freeze_two_model_datasets.ipynb` | Freezes datasets for the two-model classifier setup. |
| `Codes_Notebooks/SimpleCNN/validate_two_separate_model.ipynb` | Notebook version of the two-classifier validation workflow. |
| `Codes_Notebooks/SimpleCNN/validate_two_separate_models.py` | Scripted two-classifier validation used by the DVC pipeline. |
| `Codes_Notebooks/ClassifierModels/classifiermodels.py` | Model definitions for earlier classifier experiments. |
| `Codes_Notebooks/ClassifierModels/SimpleCNN.ipynb` | Earlier SimpleCNN training and experimentation notebook. |
| `Codes_Notebooks/ClassifierModels/test.ipynb` | Earlier classifier testing notebook. |
| `Codes_Notebooks/ClassifierModels/test_image_two_models_gradcam.py` | Tests two classifier models on images and exports Grad-CAM-style inspection outputs. |

### Staged Classifier Experiments: `Codes_Notebooks/Stage0` To `Stage3`

Each stage folder stores an earlier or staged classifier experiment with the same broad notebook pattern:

| File Pattern | Purpose |
| --- | --- |
| `1_preprocessing.ipynb` | Prepares the data for that stage. |
| `2_freeze_dataset.ipynb` | Freezes a reproducible dataset split for that stage. |
| `2_freeze_datasetV2.ipynb` | Stage-specific updated dataset-freezing variant where present. |
| `3_train_model.ipynb` | Trains the stage-specific classifier model. |
| `4_validate_model.ipynb` | Validates the stage-specific classifier model. |
| `training.log`, `evaluation.log` | Saved local logs from training or evaluation runs. |
| `gradcam_export_val_all_layers/*.html`, `*.csv` | Generated Grad-CAM galleries and summaries for stage validation outputs. |

### Earlier GAN Experiments

| File | Purpose |
| --- | --- |
| `Codes_Notebooks/CDGan/__init__.py` | Marks the CDGAN experiment folder as a Python module. |
| `Codes_Notebooks/CDGan/augmentation.py` | Image augmentation helpers for the earlier conditional/DCGAN workflow. |
| `Codes_Notebooks/CDGan/data_separation.py` | Data splitting/separation utilities for CDGAN experiments. |
| `Codes_Notebooks/CDGan/dataset.py` | Dataset loader definitions for CDGAN training. |
| `Codes_Notebooks/CDGan/model.py` | Generator/discriminator model definitions for the CDGAN experiment. |
| `Codes_Notebooks/CDGan/modeltraining.py` | Training loop and training utilities for the CDGAN experiment. |
| `Codes_Notebooks/StyleGan/preprocessing.py` | Prepares images for StyleGAN experiments. |
| `Codes_Notebooks/StyleGan/coco_mask.py` | Converts or uses COCO-style masks for StyleGAN-related data preparation. |
| `Codes_Notebooks/StyleGan/make_patches.py` | Builds image patches for StyleGAN training or inspection. |
| `Codes_Notebooks/StyleGan/make_zip.py` | Packages StyleGAN training data into zip format. |
| `Codes_Notebooks/StyleGan/train_stylegan.py` | Local wrapper for launching StyleGAN training. |
| `Codes_Notebooks/StyleGan/generate.py` | Generates images from a StyleGAN checkpoint. |
| `Codes_Notebooks/StyleGan/gui.py` | Experimental GUI helper for StyleGAN generation or inspection. |

### DVC Test Pipeline: `dvc_test`

| File | Purpose |
| --- | --- |
| `dvc_test/README.md` | Documentation for the isolated DVC smoke-test workflow. |
| `dvc_test/dvc.yaml` | Smaller DVC pipeline used to test generation, filtering, evaluation, and promotion safely. |
| `dvc_test/dvc.lock` | Locked DVC state for the test pipeline. |

### Label Studio And Annotation Helpers

| File | Purpose |
| --- | --- |
| `label_studio/README.md` | Notes for local Label Studio setup and annotation workflow. |
| `label_studio/pyproject.toml` | Python project configuration for the Label Studio helper environment. |
| `label_studio/uv.lock` | Locked dependency versions for the Label Studio helper environment. |
| `label_studio/verify_setup.py` | Checks whether Label Studio and related annotation dependencies are installed correctly. |
| `label_studio/dataset_processing.py` | Processes Label Studio annotation exports into dataset-ready files. |
| `label_studio/fiftyone/test.py` | Local FiftyOne test/helper script for dataset exploration. |
| `label_studio/label-studio-ml-backend/` | Vendored/custom Label Studio ML backend used for interactive annotation assistance; most internal files belong to that backend rather than the core X-ray generation pipeline. |

### FiftyOne Helpers

| File | Purpose |
| --- | --- |
| `fiftyone/myowncnnmodels.py` | Local helper code for inspecting or loading CNN model outputs in FiftyOne experiments. |
| `fiftyone/test.py` | Local FiftyOne test script for dataset visualization and experimentation. |

### Data And Artifact Folders

| Folder/File | Purpose |
| --- | --- |
| `data/raw/` | Original source data, including object images and annotation exports. |
| `data/interim/` | Intermediate preprocessing outputs. |
| `data/processed/` | Classifier-ready processed datasets. |
| `data/labels/` | Train, validation, and test label JSON files for staged and current classifier tasks. |
| `datasets/` | Pix2Pix datasets, matched masks, generated datasets, and DVC test workspaces. |
| `checkpoints/` | Generator/discriminator checkpoints and related model weights. |
| `CNN_models/` | Older or local CNN classifier/generator model storage. |
| `models/` | Current classifier checkpoints and promoted production/test registries. |
| `reports/` | DVC metrics, validation reports, Grad-CAM outputs, dashboards outputs, and candidate evaluation files. |
| `results/` | Generated reference outputs and experiment result files. |
| `fid_eval_runs/` | Local FID evaluation run artifacts. |
| `external/pix2pix/` | Third-party Pix2Pix implementation used by the current generator workflow. |
| `external/stylegan2-ada-pytorch/` | Third-party StyleGAN2-ADA implementation retained for earlier experiments and comparison. |

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
