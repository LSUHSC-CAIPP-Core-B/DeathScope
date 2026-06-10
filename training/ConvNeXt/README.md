# ConvNeXt Training Workflow

This directory contains the notebook workflow and example artifacts for the DeathScope ConvNeXt-Large second-stage classifier used to distinguish apoptosis from pyroptosis on cropped cell regions.

## Notebook order

- `ConvNeXt-Large_DataPreparation.ipynb`
  Builds detector-driven phase-image crop exports from phase-contrast images and apoptosis/necroptosis heatmaps.
- `ConvNeXt-Large_training.ipynb`
  Creates a deterministic train/validation split from a curated binary classifier dataset, fine-tunes the classifier, and writes run artifacts. The notebook also supports continuing from `checkpoint_last.pth`.
- `ConvNeXt-Large_plot.ipynb`
  Generates publication-ready training curves and summary figures from the saved run artifacts.
- `ConvNeXt-Large_prediction.ipynb`
  Runs single-image and batch inference using the packaged `deathscope.pyro_classifier` interface.

## Data preparation output

`ConvNeXt-Large_DataPreparation.ipynb` currently scans the demo phase-image folders under `examples/ConvNeXt_data_demo/` and writes detector-derived candidate crops under:

- `training/ConvNeXt/cropped_data/LN/Phase/apoptosis/*.png`
- `training/ConvNeXt/cropped_data/LN/Phase/necroptosis/*.png`
- `training/ConvNeXt/cropped_data/TC/Phase/apoptosis/*.png`
- `training/ConvNeXt/cropped_data/TC/Phase/necroptosis/*.png`

These exports are detector-labeled candidate crops, not the final binary apoptosis-versus-pyroptosis training set.

## Expected training data layout

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

In the current notebook configuration, `SOURCE_DATA_ROOT` points to `training/ConvNeXt/source_data/`, which is expected to contain a curated binary classifier dataset in one of the layouts above. That curated dataset is not generated automatically from `cropped_data/`; the detector export must be reorganized and relabeled into apoptosis-versus-pyroptosis class folders before training.

## Run artifacts

The training, plotting, and prediction notebooks write their outputs directly into this directory when run:

- `model_best.pth`
- `checkpoint_last.pth`
- `classes.json`
- `config.json`
- `history.json`
- `history.csv`
- `final_metrics.json`
- `plots/`
- `prediction_outputs/`

