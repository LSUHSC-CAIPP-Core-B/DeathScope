# ConvNeXt Training Workflow

This directory contains the notebook workflow for the DeathScope second-stage classifier used to distinguish apoptosis from pyroptosis on cropped cell regions. Despite the `ConvNeXt-Large` notebook filenames, both training notebooks currently default to `CONVNEXT_MODEL_SIZE = "tiny"`. Set it to `tiny`, `small`, `base`, or `large` before training; the large variant uses `convnext_large.fb_in22k_ft_in1k`.

## Notebook workflow

1. [ConvNeXt-Large_DataPreparation.ipynb](ConvNeXt-Large_DataPreparation.ipynb) exports detector-derived candidate crops from phase-contrast images and apoptosis/necroptosis heatmaps. Curate and relabel these crops for the binary classifier before splitting or training.
2. [ImageFolder_Biolgical_split.ipynb](ImageFolder_Biolgical_split.ipynb) creates deterministic, class-stratified folds by biological sample, keeping all timestamps and crops from each sample together. The current configuration exports all five folds.
3. Choose a training workflow:
   - [ConvNeXt-Large_training.ipynb](ConvNeXt-Large_training.ipynb) trains on one existing biological fold by default and supports resuming from `checkpoint_last.pth`.
   - [ConvNeXt-Large_training_5fold_cross_validation.ipynb](ConvNeXt-Large_training_5fold_cross_validation.ipynb) trains one independent model per biological fold and aggregates held-out predictions and metrics.
4. [ConvNeXt-Large_plot.ipynb](ConvNeXt-Large_plot.ipynb) reads a run's history and final metrics and exports training curves, metric summaries, and confusion matrices as PNG, TIFF, and SVG. Set `RUN_DIR` to a specific `5fold_cross_validation/fold_*` directory to plot that fold.
5. Choose an inference workflow:
   - [ConvNeXt-Large_prediction.ipynb](ConvNeXt-Large_prediction.ipynb) uses the packaged `deathscope.pyro_classifier` interface for single-image and batch inference, defaulting to `model_best.pth`.
   - [ConvNeXt_5_model_Ensemble_prediction.ipynb](ConvNeXt_5_model_Ensemble_prediction.ipynb) averages the five models' softmax probabilities with equal weights, defaulting to the last-epoch EMA weights in each fold's `checkpoint_last.pth`.

## Data preparation and biological splitting

The data preparation notebook scans demo phase-image folders under `examples/ConvNeXt_data_demo/` and exports crops to:

```text
training/ConvNeXt/cropped_data/
├── LN/Phase/
│   ├── apoptosis/*.png
│   └── necroptosis/*.png
└── TC/Phase/
    ├── apoptosis/*.png
    └── necroptosis/*.png
```

These detector-labeled candidate crops are not the final apoptosis-versus-pyroptosis dataset. Reorganize and relabel curated images while preserving biological sample and timestamp information. The biological split notebook expects:

```text
training/ConvNeXt/DataSet_Training/
├── apoptosis/<experiment>/<sample>_<timestamp>/<images>
└── pyroptosis/<experiment>/<sample>_<timestamp>/<images>
```

For example, `VID1730_E2_1_00d06h00m` and `VID1730_E2_1_2023y05m17d_20h45m` both belong to sample `VID1730_E2_1`. Adjust `SAMPLE_FOLDER_PATTERN` if your naming differs. Class names are normalized to lowercase; unrecognized sample paths or samples appearing in multiple classes cause an error.

Defaults are `N_FOLDS = 5`, `SEED = 42`, `EXPORT_ALL_FOLDS = True`, and `COPY_FILES = True`. Each class needs at least `N_FOLDS` biological samples. Stratification balances sample counts; image counts can differ between folds. The source data stays in place, and exports use:

```text
training/ConvNeXt/ImageFolder_Biolgical_split/
├── sample_fold_assignments.csv
├── fold_summary.csv
└── fold_1/                         # Also fold_2 through fold_5
    ├── train/<class>/<images>
    ├── val/<class>/<images>
    ├── image_manifest.csv
    ├── biological_sample_manifest.csv
    └── split_config.json
```

Set `EXPORT_ALL_FOLDS = False` to export only `VALIDATION_FOLD`. Set `COPY_FILES = False` for symlinks, which require the source paths to remain available. Rerunning an unchanged export resumes missing files; use a new `OUTPUT_ROOT` when changing assignments or copy mode.

## Training configuration

The single-run notebook defaults to:

```python
RUN_DIR = repo_root / "training/ConvNeXt"
SPLIT_ROOT = RUN_DIR / "ImageFolder_Biolgical_split/fold_1"
USE_EXISTING_SPLIT = True
```

This consumes `train/<class>/` and `val/<class>/` directly without copying or re-splitting images; `SOURCE_DATA_ROOT` is unused in this mode. Change `fold_1` to the desired validation fold.

For five-fold training, first export all biological folds, then use:

```python
RUN_DIR = repo_root / "training/ConvNeXt/5fold_cross_validation"
SPLIT_ROOT = repo_root / "training/ConvNeXt/ImageFolder_Biolgical_split"
USE_BIOLOGICAL_FOLDS = True
K_FOLDS = 5  # Must match the split notebook's N_FOLDS.
```

The notebook validates manifests, class mappings, exported image counts, and sample separation before training. Each biological sample must be held out exactly once. Missing or incomplete biological exports raise an error rather than falling back to image-level splitting.

Both training notebooks currently use 48-pixel RGB inputs, batch size 48, 50 epochs, learning rate `1e-4`, five warmup epochs, and EMA decay `0.9999`. Normalization uses mean `(0.5, 0.5, 0.5)` and standard deviation `(0.25, 0.25, 0.25)`. Set `CONTINUE_TRAINING = True` to resume from the last checkpoint in the run directory (or each fold directory).

Final evaluation uses the last epoch's EMA weights. `model_best.pth` separately retains raw and EMA weights from the best validation-AUC epoch. Fixed split assignments do not guarantee identical training results across hardware or library versions.

### Alternative image-level splitting

To materialize a single image-level split, set `USE_EXISTING_SPLIT = False` and configure `SOURCE_DATA_ROOT` (default: `training/ConvNeXt/source_data`). Supported layouts include flat class folders, grouped class folders, and pre-split `train/<class>` and `val/<class>` directories.

For legacy image-level cross-validation, set `USE_BIOLOGICAL_FOLDS = False` and use a separate `SPLIT_ROOT`, such as `RUN_DIR / "ImageFolder_5fold"`. This mode pools pre-split source images before assigning folds. Use the biological workflow to keep related timestamps and crops together.

## Run artifacts

The single-run notebook writes these files to `training/ConvNeXt/` by default. Five-fold training writes the same artifacts under `5fold_cross_validation/fold_1` through `fold_5`:

- `model_best.pth` and `checkpoint_last.pth`
- `classes.json` (single-run root; shared at the cross-validation root)
- `config.json`
- `history.json` and `history.csv`
- `final_metrics.json` and `val_predictions.json`

The cross-validation root additionally contains:

- `classes.json` and `config.json`
- `history_all_folds.json` and `history_all_folds.csv`
- `fold_best_metrics.csv`
- `cross_validation_metrics.json` and `cross_validation_metrics.csv`
- `validation_predictions_all_folds.csv`

Plotting writes to `RUN_DIR/plots/`. Single-model inference writes overlays and `batch_predictions.csv` to `RUN_DIR/prediction_outputs/`; a missing `classes.json` can be restored from the saved configuration or checkpoint class mapping.

## Five-model ensemble inference

The ensemble notebook reads `5fold_cross_validation/fold_1` through `fold_5`, requiring matching architecture, class ordering, and preprocessing in each fold's saved configuration. Defaults are `CHECKPOINT_NAME = "checkpoint_last.pth"` and `WEIGHTS_KEY = "model_ema"`; select `model_best.pth` to use best-AUC EMA checkpoints instead. Missing EMA weights raise an error.

Set `IMAGE_PATH` and `BATCH_INPUT_DIR` to your prediction crops. The default threshold is `0.5` for class index 1. Outputs go to `5fold_cross_validation/ensemble_prediction_outputs/` and include an overlay, `batch_predictions.csv` with ensemble probabilities, per-fold probabilities and their standard deviations, and `ensemble_prediction_config.json` recording the models and settings used.

Use independent images to assess ensemble performance. Predictions on training images, including folder-label agreement summaries, are not an independent ensemble evaluation.
