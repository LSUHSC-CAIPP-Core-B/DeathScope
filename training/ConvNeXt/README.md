# ConvNeXt Training Workflow

This directory contains the notebook workflow and example artifacts for the DeathScope ConvNeXt-Large second-stage classifier used to distinguish apoptosis from pyroptosis on cropped cell regions.

## Notebook order

- `ConvNeXt-Large_DataPreparation.ipynb`
  Builds crop datasets from phase-contrast images and detector heatmaps.
- `ConvNeXt-Large_training.ipynb`
  Creates a deterministic train/validation split, fine-tunes the classifier, and writes run artifacts.
- `ConvNeXt-Large_plot.ipynb`
  Generates publication-ready training curves and summary figures from the saved run artifacts.
- `ConvNeXt-Large_prediction.ipynb`
  Runs single-image and batch inference using the packaged `deathscope.pyro_classifier` interface.

## Expected data layout

The training notebook supports three input layouts:

- Flat class folders:
  `source_root/apoptosis/*.png`
  `source_root/pyroptosis/*.png`
- Grouped class folders:
  `source_root/group_name/apoptosis/*.png`
  `source_root/group_name/pyroptosis/*.png`
- Pre-split ImageFolder layout:
  `source_root/train/apoptosis/*.png`
  `source_root/val/apoptosis/*.png`

The current checked-in training dataset uses grouped folders under `training/ConvNeXt/source_data/`.

## Run artifacts

The training notebook writes its primary outputs directly into this directory:

- `model_best.pth`
- `checkpoint_last.pth`
- `classes.json`
- `config.json`
- `history.json`
- `history.csv`
- `final_metrics.json`
- `plots/`
- `prediction_outputs/`

## Public-release notes

- The notebooks are intentionally checked in without execution outputs so they do not leak machine-local paths or stale rendered figures.
- The `.pth`, `.json`, `.csv`, and exported figure files in this directory are example run artifacts and should be replaced if a newer training run is designated as the public reference.
- If you publish a new reference model, keep `classes.json` and `config.json` synchronized with the checkpoint used by `ConvNeXt-Large_prediction.ipynb`.
