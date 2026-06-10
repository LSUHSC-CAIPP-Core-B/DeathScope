# DeathScope Training

This directory contains the files needed to train the DeathScope YOLO cell detector with Ultralytics.

## Contents

- `train_yolo_script.py`: command-line training entrypoint
- `../configs/cell_dataset.yaml`: example dataset configuration
- `train_yolo.ipynb`: notebook version for interactive experiments

## Requirements

- Python 3.10+
- `ultralytics`
- A dataset annotated in YOLO detection format

Install the project dependencies from the repository root before training.

## Dataset Layout

Ultralytics expects image and label files to be arranged by split. The DeathScope dataset uses two classes:

- `0`: apoptosis
- `1`: necroptosis

Example layout:

```text
dataset/
├── Apoptosis/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── Necroptosis/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

Each label file must follow YOLO detection format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized to the image size.

## Dataset Config

Use `configs/cell_dataset.yaml` as a starting point. The file defines:

- `path`: dataset root directory
- `train`, `val`, `test`: image directories for each split
- `nc`: number of classes
- `names`: class names by index

Update the paths to match your local dataset before running training.

## Training

Run training from the repository root:

```bash
python training/train_yolo_script.py \
  --data configs/cell_dataset.yaml \
  --model yolov8l.pt \
  --epochs 100 \
  --batch 8 \
  --imgsz 128 \
  --device 0 \
  --project runs/train \
  --name deathscope_yolov8l
```

### Common Options

- `--data`: dataset YAML file such as `configs/cell_dataset.yaml`
- `--model`: pretrained YOLO weights or a model config
- `--epochs`: number of training epochs
- `--batch`: batch size
- `--imgsz`: image size
- `--device`: training device such as `cpu`, `0`, or `0,1`
- `--project`: output directory for Ultralytics runs
- `--name`: run name inside the project directory

Training outputs, metrics, checkpoints, and plots are written by Ultralytics under the selected `project/name` directory.

## Notebook Workflow

`train_yolo.ipynb` is provided for interactive experimentation, inspection of training results, and ad hoc model evaluation. The script is the recommended entrypoint for repeatable training runs.
