"""Cell detection and heatmap utilities for DeathScope."""

from __future__ import annotations

import logging
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
os.environ.setdefault("MPLCONFIGDIR", "/tmp")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

LOGGER = logging.getLogger(__name__)

Box = List[int]
BoxList = List[Box]
ImageLike = np.ndarray
PathLike = Union[str, os.PathLike[str]]

CLASS_APOPTOSIS = 0
CLASS_NECROPTOSIS = 1
DEFAULT_BOX_COLORS = {
    CLASS_APOPTOSIS: ((255, 255, 255), (100, 50, 255)),
    CLASS_NECROPTOSIS: ((0, 0, 0), (255, 100, 50)),
}
PYRO_RGB_SCALE = (192, 125, 0)


@dataclass(frozen=True)
class CropWindow:
    """A crop window in image coordinates."""

    top: int
    left: int
    height: int
    width: int

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def right(self) -> int:
        return self.left + self.width


def random_crop(image: ImageLike, crop_size: int) -> Tuple[ImageLike, int, int]:
    """Return a random square crop and its top-left coordinates."""
    if crop_size <= 0:
        raise ValueError("crop_size must be positive")

    image_h, image_w = image.shape[:2]
    if crop_size > image_h or crop_size > image_w:
        raise ValueError("crop_size must fit inside the image")

    max_left = image_w - crop_size
    max_top = image_h - crop_size
    left = random.randint(0, max_left)
    top = random.randint(0, max_top)
    return image[top : top + crop_size, left : left + crop_size], top, left


def _get_pyro_predictor():
    try:
        from . import pyro_classifier as pyro_module
    except ImportError:  # pragma: no cover - legacy direct-module usage
        import pyro_classifier as pyro_module
    return pyro_module


class CellDetector:
    """Wrapper around a YOLO model used for cell detection and heatmap analysis."""

    IMAGE_SIZE = 128
    CROP_WIDTH = 128
    CROP_HEIGHT = 128
    HEATMAP_PADDING = 128

    def __init__(self, model_path: PathLike, image_size: int = IMAGE_SIZE) -> None:
        self.model = YOLO(str(model_path))
        self.IMAGE_SIZE = int(image_size)
        self.CROP_WIDTH = int(image_size)
        self.CROP_HEIGHT = int(image_size)
        self.HEATMAP_PADDING = int(image_size)

    # |----------------------PREDICTIONS--------------------------------------------|

    def predict(
        self,
        image: ImageLike,
        conf: float,
        iou: float,
        device: Optional[Union[str, int]] = None,
    ) -> BoxList:
        """Predict death-type boxes on a single image."""
        prepared_image = self._ensure_three_channels(image)
        results = self.model.predict(
            prepared_image,
            conf=conf,
            iou=iou,
            imgsz=self.IMAGE_SIZE,
            device=device,
            verbose=False,
        )
        return self.process_results(results)

    def predict_with_crop(
        self,
        big_image: ImageLike,
        conf: float,
        iou: float,
        device: Optional[Union[str, int]] = None,
        withOverlap: bool = False,
        withImage: bool = False,
    ):
        """Run tiled prediction over an image and scale boxes to full-image coordinates."""
        if withOverlap:
            raise NotImplementedError("Overlapping crop inference is not implemented.")

        padded_image, width_shift, height_shift = self.prepare_image_to_crop_no_overlap(
            big_image,
            withWinShift=True,
        )
        image_h, image_w = padded_image.shape[:2]
        windows = self._iter_grid_windows(image_h, image_w, height_shift, width_shift)

        results_all: BoxList = []
        for window in windows:
            crop = padded_image[window.top : window.bottom, window.left : window.right]
            results = self.predict(crop, conf, iou, device=device)
            results_all.extend(self.scale_crop_results(results, window.left, window.top))

        if withImage:
            return results_all, padded_image
        return results_all

    def predict_with_heatmap(
        self,
        big_image: ImageLike,
        conf: float,
        iou: float,
        device: Optional[Union[str, int]] = None,
        desired_coverage: int = 10,
        withImage: bool = False,
        withResults: bool = False,
        Pyro_pred: bool = False,
    ):
        """Generate apoptosis and necroptosis heatmaps from random crop inference."""
        padded_image = self.prepare_image_to_crop_heatmap(big_image)
        image_h, image_w = padded_image.shape[:2]
        total_crops = self._calculate_total_crops(image_h, image_w, desired_coverage)
        LOGGER.info("Running %s random crops for heatmap generation.", total_crops)

        apoptosis_image = np.zeros((image_h, image_w), dtype=np.int32)
        necroptosis_image = np.zeros((image_h, image_w), dtype=np.int32)
        background_image = np.zeros((image_h, image_w), dtype=np.int32)
        results_all: Optional[BoxList] = [] if withResults else None

        for _ in range(total_crops):
            crop, top, left = random_crop(padded_image, self.IMAGE_SIZE)
            results = self.predict(crop, conf, iou, device=device)

            if results_all is not None:
                results_all.extend(self.scale_crop_results(results, left, top))

            apoptosis_crop, necroptosis_crop, background_crop = self.gen_masks(
                np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.int32),
                np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.int32),
                np.ones((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.int32),
                results,
            )
            region = np.s_[top : top + self.IMAGE_SIZE, left : left + self.IMAGE_SIZE]
            apoptosis_image[region] += apoptosis_crop
            necroptosis_image[region] += necroptosis_crop
            background_image[region] += background_crop

        apoptosis_map, necroptosis_map, background_map = self._normalize_heatmaps(
            apoptosis_image,
            necroptosis_image,
            background_image,
        )
        apoptosis_map, necroptosis_map = self._suppress_weaker_regions(
            apoptosis_map,
            necroptosis_map,
        )

        trimmed_phase = self._trim_heatmap_padding(padded_image)
        trimmed_background = background_map

        if Pyro_pred:
            apoptosis_map, necroptosis_map, pyroptosis_map = self._split_pyroptosis(
                apoptosis_map,
                necroptosis_map,
                trimmed_phase,
            )
            outputs: List[object] = [apoptosis_map, necroptosis_map, pyroptosis_map, trimmed_background]
        else:
            outputs = [apoptosis_map, necroptosis_map, trimmed_background]

        if withImage:
            outputs.append(trimmed_phase)
        if results_all is not None:
            outputs.append(results_all)
        return tuple(outputs)

    def predict_with_heatmap_batch(
        self,
        images_dir: PathLike,
        Name: str,
        desired_coverage: int = 60,
        green_img: bool = False,
        pred_images: bool = True,
        cut_off: float = 0,
        g_cut_off: float = 35,
        incucyte: bool = True,
        binary: bool = False,
    ):
        """Run heatmap prediction over a directory of images and export overlays and CSVs."""
        return self._predict_with_heatmap_batch(
            images_dir=images_dir,
            name=Name,
            desired_coverage=desired_coverage,
            green_img=green_img,
            pred_images=pred_images,
            cut_off=cut_off,
            g_cut_off=g_cut_off,
            incucyte=incucyte,
            binary=binary,
            include_pyro=False,
        )

    def predict_with_heatmap_batch_pyro(
        self,
        images_dir: PathLike,
        Name: str,
        desired_coverage: int = 60,
        green_img: bool = False,
        pred_images: bool = True,
        cut_off: float = 0,
        g_cut_off: float = 35,
        incucyte: bool = True,
        binary: bool = False,
    ):
        """Run heatmap prediction with an additional pyroptosis split over a directory."""
        return self._predict_with_heatmap_batch(
            images_dir=images_dir,
            name=Name,
            desired_coverage=desired_coverage,
            green_img=green_img,
            pred_images=pred_images,
            cut_off=cut_off,
            g_cut_off=g_cut_off,
            incucyte=incucyte,
            binary=binary,
            include_pyro=True,
        )

    # |----------------------RESULT PROCESSING--------------------------------------------|

    def scale_crop_results(self, results: Sequence[Sequence[int]], curr_topleft_x: int, curr_topleft_y: int) -> BoxList:
        """Scale crop-level boxes into image-level coordinates."""
        return [
            [
                int(top + curr_topleft_y),
                int(left + curr_topleft_x),
                int(bottom + curr_topleft_y),
                int(right + curr_topleft_x),
                int(cell_cls),
            ]
            for top, left, bottom, right, cell_cls in results
        ]

    def process_results(self, results) -> BoxList:
        """Convert a YOLO result object into `[top, left, bottom, right, class]` boxes."""
        if not results:
            return []

        result = results[0]
        boxes: BoxList = []
        for box in result.boxes:
            left, top, right, bottom = box.xyxy[0].tolist()
            boxes.append(
                [
                    int(round(top)),
                    int(round(left)),
                    int(round(bottom)),
                    int(round(right)),
                    int(box.cls),
                ]
            )
        return boxes

    def rotate_results(self, results: Sequence[Sequence[int]], rotation: int) -> BoxList:
        """Rotate predicted boxes back into the original crop orientation."""
        if rotation not in (0, 1, 2, 3):
            raise ValueError("rotation must be one of 0, 1, 2, or 3")
        if rotation == 3:
            return [list(map(int, result)) for result in results]

        rotated: BoxList = []
        for top, left, bottom, right, cell_cls in results:
            rotated_box = self._rotate_box((top, left, bottom, right), rotation)
            rotated.append([*rotated_box, int(cell_cls)])
        return rotated

    def gen_masks(
        self,
        apoptosis_image: ImageLike,
        necroptosis_image: ImageLike,
        background_image: ImageLike,
        results: Sequence[Sequence[int]],
    ) -> Tuple[ImageLike, ImageLike, ImageLike]:
        """Fill class masks for a crop based on predicted boxes."""
        for top, left, bottom, right, cell_cls in results:
            background_image[top:bottom, left:right] = 0
            if cell_cls == CLASS_APOPTOSIS:
                apoptosis_image[top:bottom, left:right] += 1
            elif cell_cls == CLASS_NECROPTOSIS:
                necroptosis_image[top:bottom, left:right] += 1
        return apoptosis_image, necroptosis_image, background_image

    def min_max_normalize(self, matrix: ImageLike) -> ImageLike:
        """Normalize an array into the `[0, 1]` range."""
        array = np.asarray(matrix, dtype=np.float64)
        min_val = float(np.min(array))
        max_val = float(np.max(array))
        if math.isclose(max_val, min_val):
            return np.zeros_like(array, dtype=np.float64)
        return (array - min_val) / (max_val - min_val)

    # |----------------------PREPROCESSING--------------------------------------------|

    def prepare_image_to_crop_no_overlap(self, image: ImageLike, withWinShift: bool = False):
        """Pad an image so it can be tiled into equal non-overlapping crops."""
        image_h, image_w = image.shape[:2]
        pad_h = (-image_h) % self.CROP_HEIGHT
        pad_w = (-image_w) % self.CROP_WIDTH
        padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

        if withWinShift:
            return padded, self.CROP_WIDTH, self.CROP_HEIGHT
        return padded

    def prepare_image_to_crop_heatmap(self, image: ImageLike) -> ImageLike:
        """Pad an image so random crops can fully cover the original frame."""
        pad = self.HEATMAP_PADDING
        return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT)

    def bftocontrast(self, image_path: PathLike) -> Optional[ImageLike]:
        """Convert a brightfield image into a higher-contrast phase-like representation."""
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            LOGGER.error("Image not loaded. Please check the file path: %s", image_path)
            return None

        blurred = cv2.GaussianBlur(image, (15, 15), 0)
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
        laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.bitwise_not(np.uint8(laplacian))

    # |----------------------VISUALIZATIONS--------------------------------------------|

    def box_image(
        self,
        image: ImageLike,
        boxes: Sequence[Sequence[int]],
        image_green: Optional[ImageLike] = None,
        heatmap: int = 0,
        prepare_type: int = 0,
    ):
        """Draw predicted boxes on a phase image and optionally a green-channel image."""
        del heatmap  # Legacy unused argument kept for compatibility.

        if image_green is not None:
            if prepare_type == 0:
                image_green = self.prepare_image_to_crop_no_overlap(image_green)
            elif prepare_type == 1:
                image_green = self.prepare_image_to_crop_heatmap(image_green)

        for top, left, bottom, right, cell_cls in boxes:
            phase_color, green_color = DEFAULT_BOX_COLORS.get(
                int(cell_cls),
                ((255, 255, 0), (255, 255, 0)),
            )
            cv2.rectangle(image, (left, top), (right, bottom), phase_color, 2)
            if image_green is not None:
                cv2.rectangle(image_green, (left, top), (right, bottom), green_color, 2)

        if image_green is not None:
            return image, image_green
        return image

    # |----------------------EVALUATIONS--------------------------------------------|

    def find_closed_regions(self, matrix: ImageLike) -> Dict[str, List[Tuple[int, int]]]:
        """Return connected foreground regions from a binary matrix."""
        binary = np.asarray(matrix, dtype=np.uint8)
        if binary.ndim != 2:
            raise ValueError("matrix must be 2D")

        num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
        regions: Dict[str, List[Tuple[int, int]]] = {}
        for label in range(1, num_labels):
            coordinates = np.argwhere(labels == label)
            regions[f"region_{label}"] = [tuple(map(int, point)) for point in coordinates]
        return regions

    def visualize_regions(self, matrix: ImageLike, regions: Dict[str, List[Tuple[int, int]]]) -> None:
        """Visualize connected components alongside the original matrix."""
        rows, cols = np.asarray(matrix).shape[:2]
        region_matrix = np.zeros((rows, cols), dtype=np.int32)
        for index, region in enumerate(regions.values(), start=1):
            if region:
                coords = np.asarray(region, dtype=np.int32)
                region_matrix[coords[:, 0], coords[:, 1]] = index

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(matrix, cmap="gray")
        axes[0].set_title("Original Matrix")
        axes[1].imshow(region_matrix, cmap="tab20")
        axes[1].set_title("Regions Identified")
        plt.tight_layout()
        plt.show()

    def calculate_region_totals(
        self,
        image: ImageLike,
        regions: Dict[str, Sequence[Tuple[int, int]]],
    ) -> Dict[str, float]:
        """Calculate the summed pixel intensity for each region."""
        array = np.asarray(image)
        totals: Dict[str, float] = {}
        for region_id, region in regions.items():
            if not region:
                totals[region_id] = 0.0
                continue
            coords = np.asarray(region, dtype=np.int32)
            totals[region_id] = float(array[coords[:, 0], coords[:, 1]].sum())
        return totals

    def compare_and_modify(
        self,
        image1: ImageLike,
        image2: ImageLike,
        regions: Dict[str, Sequence[Tuple[int, int]]],
    ) -> ImageLike:
        """Zero regions in `image1` when the matching region is at least as strong in `image2`."""
        totals_image1 = self.calculate_region_totals(image1, regions)
        totals_image2 = self.calculate_region_totals(image2, regions)
        modified = np.asarray(image1).copy()

        for region_id, region in regions.items():
            if totals_image2.get(region_id, 0.0) >= totals_image1.get(region_id, 0.0) and region:
                coords = np.asarray(region, dtype=np.int32)
                modified[coords[:, 0], coords[:, 1]] = 0
        return modified

    def Predict_evluation(
        self,
        imageA: ImageLike,
        imageN: ImageLike,
        imageG: ImageLike,
        Acutoff: float = 0.0,
        Ncutoff: float = 0.0,
        Gcutoff: float = 0,
    ) -> List[float]:
        """Evaluate predicted apoptosis and necroptosis maps against a green-channel reference."""
        imageG = np.where(imageG > Gcutoff, imageG, 0)
        imageA = np.where(imageA > Acutoff, imageA, 0)
        imageN = np.where(imageN > Ncutoff, imageN, 0)

        regions_necroptosis = self.find_closed_regions(np.where(imageN == 0, 0, 1))
        regions_apoptosis = self.find_closed_regions(np.where(imageA == 0, 0, 1))
        regions_green = self.find_closed_regions(np.where(imageG == 0, 0, 1))

        imageN_m = self.compare_and_modify(imageN, imageA, regions_necroptosis)
        imageA_m = self.compare_and_modify(imageA, imageN, regions_apoptosis)

        totals_imageA = self.calculate_region_totals(imageA_m, regions_green)
        totals_imageN = self.calculate_region_totals(imageN_m, regions_green)
        tp_a = len([value for value in totals_imageA.values() if value > 0])
        tp_n = len([value for value in totals_imageN.values() if value > 0])

        green_region_count = max(len(regions_green), 1)
        tp = (tp_a + tp_n) / green_region_count
        fn = 1 - tp

        totals_imageG_A = self.calculate_region_totals(imageG, regions_apoptosis)
        fp_a = len([value for value in totals_imageG_A.values() if value == 0])

        totals_imageG_N = self.calculate_region_totals(imageG, regions_necroptosis)
        fp_n = len([value for value in totals_imageG_N.values() if value == 0])

        return [tp_a, tp_n, len(regions_green), tp, fn, fp_a, fp_n]

    # |----------------------INTERNAL HELPERS--------------------------------------------|

    def _ensure_three_channels(self, image: ImageLike) -> ImageLike:
        if image.ndim < 3:
            return cv2.merge((image, image, image))
        return image

    def _calculate_total_crops(self, image_h: int, image_w: int, desired_coverage: int) -> int:
        if desired_coverage <= 0:
            raise ValueError("desired_coverage must be positive")
        crop_area = self.IMAGE_SIZE * self.IMAGE_SIZE
        total = (image_h * image_w * desired_coverage) // crop_area
        return max(1, int(total))

    def _iter_grid_windows(
        self,
        image_h: int,
        image_w: int,
        height_shift: int,
        width_shift: int,
    ) -> Iterable[CropWindow]:
        for top in range(0, image_h, height_shift):
            for left in range(0, image_w, width_shift):
                yield CropWindow(top=top, left=left, height=self.CROP_HEIGHT, width=self.CROP_WIDTH)

    def _normalize_heatmaps(
        self,
        apoptosis_image: ImageLike,
        necroptosis_image: ImageLike,
        background_image: ImageLike,
    ) -> Tuple[ImageLike, ImageLike, ImageLike]:
        apoptosis = apoptosis_image.astype(np.float64, copy=False)
        necroptosis = necroptosis_image.astype(np.float64, copy=False)
        background = background_image.astype(np.float64, copy=False)

        total = apoptosis + necroptosis + background
        np.maximum(total, 1.0, out=total)

        return (
            self._trim_heatmap_padding(apoptosis / total),
            self._trim_heatmap_padding(necroptosis / total),
            self._trim_heatmap_padding(background / total),
        )

    def _trim_heatmap_padding(self, image: ImageLike) -> ImageLike:
        pad = self.HEATMAP_PADDING
        return image[pad:-pad, pad:-pad]

    def _binary_regions(self, image: ImageLike) -> Dict[str, List[Tuple[int, int]]]:
        return self.find_closed_regions(np.where(image == 0, 0, 1))

    def _suppress_weaker_regions(self, apoptosis_map: ImageLike, necroptosis_map: ImageLike) -> Tuple[ImageLike, ImageLike]:
        necro_regions = self._binary_regions(necroptosis_map)
        apop_regions = self._binary_regions(apoptosis_map)
        necroptosis_filtered = self.compare_and_modify(necroptosis_map, apoptosis_map, necro_regions)
        apoptosis_filtered = self.compare_and_modify(apoptosis_map, necroptosis_map, apop_regions)
        return apoptosis_filtered, necroptosis_filtered

    def _split_pyroptosis(
        self,
        apoptosis_map: ImageLike,
        necroptosis_map: ImageLike,
        phase_image: ImageLike,
    ) -> Tuple[ImageLike, ImageLike, ImageLike]:
        pyroptosis_map = apoptosis_map.copy()
        apoptosis_regions = self._binary_regions(apoptosis_map)

        for region in apoptosis_regions.values():
            if not region:
                continue
            coords = np.asarray(region, dtype=np.int32)
            min_r, min_c = coords.min(axis=0)
            max_r, max_c = coords.max(axis=0)
            cropped_region = phase_image[min_r : max_r + 1, min_c : max_c + 1]
            cropped_region_rgb = cv2.cvtColor(cropped_region.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            pred_class = _get_pyro_predictor().Pyro_Pred(cropped_region_rgb)

            if pred_class == "pyroptosis":
                apoptosis_map[coords[:, 0], coords[:, 1]] = 0
            else:
                pyroptosis_map[coords[:, 0], coords[:, 1]] = 0

        return apoptosis_map, necroptosis_map, pyroptosis_map

    def _predict_with_heatmap_batch(
        self,
        images_dir: PathLike,
        name: str,
        desired_coverage: int,
        green_img: bool,
        pred_images: bool,
        cut_off: float,
        g_cut_off: float,
        incucyte: bool,
        binary: bool,
        include_pyro: bool,
    ) -> List[Tuple[object, ...]]:
        base_dir = Path(images_dir)
        phase_dir = base_dir / "Phase"
        green_dir = base_dir / "Green"
        image_paths = sorted(path for path in phase_dir.iterdir() if path.is_file())
        results: List[Tuple[object, ...]] = []

        for image_path in image_paths:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                LOGGER.warning("Skipping unreadable phase image: %s", image_path)
                continue

            green_image_filtered = None
            if green_img:
                source_green = green_dir / image_path.name
                img_green = cv2.imread(str(source_green), cv2.IMREAD_UNCHANGED)
                if img_green is None:
                    LOGGER.warning("Skipping unreadable green image: %s", source_green)
                else:
                    img_green = cv2.normalize(img_green, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    green_image_filtered = np.where(img_green > g_cut_off, img_green, 0)

            if include_pyro:
                prediction = self.predict_with_heatmap(
                    img,
                    0.5,
                    0.5,
                    desired_coverage=desired_coverage,
                    withImage=False,
                    withResults=False,
                    device=None,
                    Pyro_pred=True,
                )
                apoptosis_image, necroptosis_image, pyroptosis_image, _background_image = prediction
                apoptosis_image = np.where(apoptosis_image > cut_off, apoptosis_image, 0)
                necroptosis_image = np.where(necroptosis_image > cut_off, necroptosis_image, 0)
                pyroptosis_image = np.where(pyroptosis_image > cut_off, pyroptosis_image, 0)
            else:
                prediction = self.predict_with_heatmap(
                    img,
                    0.5,
                    0.5,
                    desired_coverage=desired_coverage,
                    withImage=False,
                    withResults=False,
                    device=None,
                    Pyro_pred=False,
                )
                apoptosis_image, necroptosis_image, _background_image = prediction
                apoptosis_image = np.where(apoptosis_image > cut_off, apoptosis_image, 0)
                necroptosis_image = np.where(necroptosis_image > cut_off, necroptosis_image, 0)
                pyroptosis_image = None

            results.append(
                self._build_batch_result_row(
                    image_path.stem,
                    apoptosis_image,
                    necroptosis_image,
                    pyroptosis_image,
                    green_image_filtered,
                    green_img,
                    cut_off,
                    g_cut_off,
                )
            )

            if pred_images:
                self._write_prediction_images(
                    base_dir=base_dir,
                    image_stem=image_path.stem,
                    phase_image=img,
                    apoptosis_image=apoptosis_image,
                    necroptosis_image=necroptosis_image,
                    pyroptosis_image=pyroptosis_image,
                    green_image_filtered=green_image_filtered,
                    binary=binary,
                )

        self._write_batch_csvs(
            base_dir=base_dir,
            name=name,
            results=results,
            green_img=green_img,
            include_pyro=include_pyro,
            incucyte=incucyte,
        )
        return results

    def _build_batch_result_row(
        self,
        stem: str,
        apoptosis_image: ImageLike,
        necroptosis_image: ImageLike,
        pyroptosis_image: Optional[ImageLike],
        green_image_filtered: Optional[ImageLike],
        green_required: bool,
        cut_off: float,
        g_cut_off: float,
    ) -> Tuple[object, ...]:
        row: List[object] = [
            stem,
            float(apoptosis_image.sum()),
            float(necroptosis_image.sum()),
        ]
        if pyroptosis_image is not None:
            row.append(float(pyroptosis_image.sum()))

        row.extend(
            [
                int((apoptosis_image > cut_off).sum()),
                int((necroptosis_image > cut_off).sum()),
            ]
        )
        if pyroptosis_image is not None:
            row.append(int((pyroptosis_image > cut_off).sum()))
        if green_required:
            green_count = 0 if green_image_filtered is None else int((green_image_filtered > g_cut_off).sum())
            row.append(green_count)
        return tuple(row)

    def _write_prediction_images(
        self,
        base_dir: Path,
        image_stem: str,
        phase_image: ImageLike,
        apoptosis_image: ImageLike,
        necroptosis_image: ImageLike,
        pyroptosis_image: Optional[ImageLike],
        green_image_filtered: Optional[ImageLike],
        binary: bool,
    ) -> None:
        img_rgb = cv2.cvtColor(phase_image, cv2.COLOR_GRAY2RGB).astype(np.uint8)
        apoptosis_rgb = self._heatmap_to_rgb(apoptosis_image, (255, 0, 0))
        necroptosis_rgb = self._heatmap_to_rgb(necroptosis_image, (0, 0, 255))
        overlays = {
            "apoptosis_image_wPhase": cv2.add(img_rgb, apoptosis_rgb),
            "necroptosis_image_wPhase": cv2.add(img_rgb, necroptosis_rgb),
        }

        death_overlay = cv2.add(overlays["apoptosis_image_wPhase"], necroptosis_rgb)
        if pyroptosis_image is not None:
            pyroptosis_rgb = self._heatmap_to_rgb(pyroptosis_image, PYRO_RGB_SCALE)
            overlays["pyroptosis_image_wPhase"] = cv2.add(img_rgb, pyroptosis_rgb)
            death_overlay = cv2.add(death_overlay, pyroptosis_rgb)
        else:
            pyroptosis_rgb = None
        overlays["death_image_wphase"] = death_overlay

        self._save_overlay_images(base_dir, image_stem, overlays)

        if binary:
            apoptosis_binary_rgb = self._heatmap_to_rgb(np.where(apoptosis_image == 0, 0, 1), (255, 0, 0))
            necroptosis_binary_rgb = self._heatmap_to_rgb(np.where(necroptosis_image == 0, 0, 1), (0, 0, 255))
            binary_overlays = {
                "apoptosis_image_wPhase_b": cv2.add(img_rgb, apoptosis_binary_rgb),
                "necroptosis_image_wPhase_b": cv2.add(img_rgb, necroptosis_binary_rgb),
            }
            death_binary = cv2.add(binary_overlays["apoptosis_image_wPhase_b"], necroptosis_binary_rgb)
            if pyroptosis_rgb is not None and pyroptosis_image is not None:
                pyroptosis_binary_rgb = self._heatmap_to_rgb(np.where(pyroptosis_image == 0, 0, 1), PYRO_RGB_SCALE)
                binary_overlays["pyroptosis_image_wPhase_b"] = cv2.add(img_rgb, pyroptosis_binary_rgb)
                death_binary = cv2.add(death_binary, pyroptosis_binary_rgb)
            binary_overlays["death_image_wphase_b"] = death_binary
            self._save_overlay_images(base_dir, image_stem, binary_overlays)

        if green_image_filtered is not None:
            green_rgb = cv2.merge(
                [
                    np.zeros_like(phase_image, dtype=np.uint8),
                    green_image_filtered.astype(np.uint8),
                    np.zeros_like(phase_image, dtype=np.uint8),
                ]
            )
            green_overlay = cv2.add(img_rgb, green_rgb)
            plt.imsave(str(base_dir / f"{image_stem}_green_image_wphase.png"), green_overlay)
            death_green_overlay = cv2.add(death_overlay, green_rgb)
            plt.imsave(str(base_dir / f"{image_stem}_death_image_green_wphase.png"), death_green_overlay)
            if binary:
                binary_death = death_binary.copy()
                binary_death = cv2.add(binary_death, green_rgb)
                plt.imsave(str(base_dir / f"{image_stem}_death_image_green_wphase_b.png"), binary_death)

    def _save_overlay_images(self, base_dir: Path, image_stem: str, overlays: Dict[str, ImageLike]) -> None:
        for suffix, image in overlays.items():
            plt.imsave(str(base_dir / f"{image_stem}_{suffix}.png"), image)

    def _heatmap_to_rgb(self, image: ImageLike, color: Tuple[int, int, int]) -> ImageLike:
        image_uint8 = np.clip(np.asarray(image) * 255, 0, 255).astype(np.uint8)
        channels = [np.zeros_like(image_uint8) for _ in range(3)]
        for index, scale in enumerate(color):
            if scale == 0:
                continue
            channels[index] = ((image_uint8.astype(np.float32) * scale) / 255.0).astype(np.uint8)
        return cv2.merge(channels)

    def _write_batch_csvs(
        self,
        base_dir: Path,
        name: str,
        results: Sequence[Tuple[object, ...]],
        green_img: bool,
        include_pyro: bool,
        incucyte: bool,
    ) -> None:
        columns = ["fn", "A_Prob_sum", "N_Prob_sum"]
        if include_pyro:
            columns.append("P_Prob_sum")
        columns.extend(["A_Area_sum", "N_Area_sum"])
        if include_pyro:
            columns.append("P_Area_sum")
        if green_img:
            columns.append("Green_Area_sum")

        df = pd.DataFrame(results, columns=columns)
        prob_columns = ["A_Prob_sum", "N_Prob_sum"] + (["P_Prob_sum"] if include_pyro else [])
        area_columns = ["A_Area_sum", "N_Area_sum"] + (["P_Area_sum"] if include_pyro else [])
        df["T_Prob_sum"] = df[prob_columns].sum(axis=1)
        df["T_Area_sum"] = df[area_columns].sum(axis=1)
        df.to_csv(base_dir / f"{name}.csv", index=False)

        if not incucyte or df.empty:
            return

        split_cols = df["fn"].str.split("_", n=3, expand=True)
        if split_cols.shape[1] < 4:
            LOGGER.warning("Skipping Incucyte aggregation because filenames do not match the expected format.")
            return

        df[["ID", "Well", "Field", "Time"]] = split_cols
        group_columns = prob_columns + area_columns + (["Green_Area_sum"] if green_img else [])
        by_well = df.groupby(["Well", "Time"])[group_columns].mean()
        by_well.to_csv(base_dir / f"{name}_agg.csv")
        by_time = df.groupby(["Time"])[group_columns].mean()
        by_time.to_csv(base_dir / f"{name}_agg_Time.csv")

    def _rotate_box(self, box: Tuple[int, int, int, int], rotation: int) -> Tuple[int, int, int, int]:
        top, left, bottom, right = box
        corners = np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float64,
        )
        center = np.array([self.IMAGE_SIZE / 2.0, self.IMAGE_SIZE / 2.0], dtype=np.float64)

        angle = -math.radians((rotation * 90) + 90)
        rotation_matrix = np.array(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        rotated = (corners - center) @ rotation_matrix.T + center
        min_xy = np.floor(rotated.min(axis=0)).astype(int)
        max_xy = np.ceil(rotated.max(axis=0)).astype(int)
        left_r, top_r = min_xy.tolist()
        right_r, bottom_r = max_xy.tolist()
        return top_r, left_r, bottom_r, right_r
