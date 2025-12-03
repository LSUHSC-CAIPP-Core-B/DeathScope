# CellDetection

CellDetection provides an end-to-end workflow for detecting dead cells (apoptosis vs. necroptosis). It bundles dataset preparation notebooks, visualization helpers, and a YOLOv8-based training and inference pipeline so you can go from raw microscope images to deployable models.

## Features
- **Dataset preparation and labeling**: Tools to crop whole-slide images, generate YOLO-format labels, convert label formats (YOLO/XML/TFRecord), and audit images with adjustable preprocessing parameters.
- **Augmentation and splitting**: Notebooks to augment brightness/contrast, generate background negatives, and split the dataset into train/val/test subsets.
- **YOLOv8 training**: Minimal scripts and notebooks for training custom detectors with configurable datasets and pretrained weights.
- **Inference and visualization**: Utilities to run tiled inference on large images, build heat maps from rotated crops, and render detections for inspection.

## Repository structure
- `cell_adjustor/`: Dataset preparation scripts and notebooks (cropping, labeling, augmentation, conversion, visualization).
- `yolo/`: YOLOv8 training/inference scripts, dataset YAML examples, and notebooks.
- `docs/`: Example outputs for detections and heat maps.
- `requirements_cell_mac.txt` / `requirements_cell_ubuntu.txt`: Dependencies for data preparation on macOS and Ubuntu.
- `requirements_ultralytics_mac.txt`: Dependencies for YOLO training on macOS; see below for Ubuntu/CUDA setup.

## Environment setup
Choose the environment that matches the tasks you plan to run.

### Data preparation (cell_adjustor)
- Python 3.9 tested.
- Create and activate a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Install requirements for your platform:
  ```bash
  # macOS
  pip install -r requirements_cell_mac.txt

  # Ubuntu
  pip install -r requirements_cell_ubuntu.txt
  ```

### YOLOv8 training and inference
- Python 3.8 on macOS or 3.9 on Ubuntu tested.
- macOS dependencies:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements_ultralytics_mac.txt
  ```
- Ubuntu with CUDA (tested on Ubuntu 20.04, NVIDIA RTX A5000, CUDA 12.3):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip3 install torch==2.3.0.dev20240213+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121
  pip3 install torchvision==0.18.0.dev20240213+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121
  pip3 install ultralytics
  ```

## Workflow
### 1) Prepare and label data
Key utilities live in `cell_adjustor/`:
- `cell_image_adjustor_matplot.py`: Interactive viewer to tune preprocessing parameters and save configurations to `params.csv`.
- `label_cell_images.ipynb`: Crop whole images into tiles, generate YOLO-format labels, and masks.
- `get_background_images.ipynb`: Extract background-only crops to balance the dataset.
- `augument_cell_images.ipynb`: Apply brightness/contrast augmentations to images, labels, and masks.
- `split_dataset.ipynb`: Split augmented data into train/val/test folders.
- `yolo_to_xml.py`, `xml_to_tfrecord.py`, `change_yolo_class_num.py`: Convert labels between formats or adjust class indices.

### 2) Define the dataset YAML
Follow the example in `yolo/mef1.yaml` or `yolo/cell_dataset.yaml` to point YOLOv8 to your images/labels. The structure should resemble:
```
Dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
```
Set `path` to the dataset root, `train`/`val`/`test` to the subfolders, `nc` to the number of classes, and `names` to the class list.

### 3) Train a model
Use the minimal script in `yolo/train_yolo_script.py` as a starting point and adjust model weights, dataset YAML, and hyperparameters:
```python
from ultralytics import YOLO

model = YOLO("yolov8l.pt")
model.train(
    data="mef1.yaml",
    epochs=2,
    batch=8,
    imgsz=128,
    name="YOLO_run",
    device="0",
)
```
Training outputs (weights, metrics) will be written to `runs/`.

### 4) Run inference and visualize
- `yolo_inference_test.ipynb`: Run tiled inference on a single image and visualize detections (supports optional staining overlays).
- `yolo_inference_heatmap.ipynb`: Generate per-class heat maps by aggregating rotated crops; outputs normalized maps in `[0, 1]`.
- `yolo_inference_comparison.ipynb`: Compare standard detection vs. heat-map inference side-by-side.
- `CellDetector` class (`yolo/CellDetector`): Wrapper for YOLOv8 checkpoints with helpers for cropping, overlapping tiles, heat maps, and visualization.

## Example outputs
- Detection preview: `docs/cell_example.png`
- Heat map example: `docs/heatmap_example.png`
- Side-by-side comparison: `docs/apoptosis_2.png`

---
Feel free to open an issue or PR with questions, improvements, or new workflows.
