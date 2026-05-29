# DVC Test Pipeline

This folder contains an experimental end-to-end DVC pipeline that keeps the root
`dvc.yaml` untouched.

It is intentionally configured as a smoke test:

- prepares an isolated `_dvc_test` dataset workspace from the current complete Pix2Pix dataset
- bootstraps `Shampoo_NOBGR_pix2pix_StructCond_V1_Stage24_DVC_TEST` from the production Stage24 `latest` checkpoint if needed, then resumes and saves epoch checkpoints every 50 epochs
- runs the same generation, filtering, FID, candidate evaluation, and promotion flow into `_dvc_test`, `reports/dvc_test`, and `models/production_test`

Run from this directory:

```bash
cd dvc_test
dvc repro
```

The test training is still lighter than a full experiment. Increase `n_epochs`,
`n_epochs_decay`, `max_dataset_size`, `num_scenes`, and FID `max_images` in
`dvc_test/dvc.yaml` when you want a full run.

## Dashboards

From the repository root, use the unified dashboard to run and monitor the DVC
test pipeline, continue Pix2Pix training, and run FID evaluation:

```bash
python src/xraygen/pipeline/dvc_unified_dashboard.py
```

Open:

```text
http://127.0.0.1:8770
```

Use the extended FiftyOne dashboard when you also want embedding inspection
before training or real-vs-generated comparison after evaluation:

```bash
python src/xraygen/pipeline/dvc_fiftyone_dashboard.py
```

Open:

```text
http://127.0.0.1:8771
```

The FiftyOne tab can use existing FID `real/` and `fake/` folders, or rerun FID
with `--keep_images` and the selected image-pair count before launching the
FiftyOne app on `http://localhost:5151`.
