"""Pyroptosis classification utilities.

This module wraps a ConvNeXt classifier used to split apoptosis heatmap regions
into apoptosis and pyroptosis classes. Model construction and checkpoint loading
are lazy and cached, so importing the module is cheap while repeated predictions
reuse the same model instance.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from timm import create_model

LOGGER = logging.getLogger(__name__)

ImageLike = np.ndarray
PathLike = Union[str, Path]

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "pyroptosis" / "convnext_large_model_best.pth"
DEFAULT_CLASSES_PATH = PACKAGE_ROOT / "models" / "pyroptosis" / "classes.json"
DEFAULT_IMAGE_SIZE = 48
DEFAULT_ARCHITECTURE = "convnext_large.fb_in22k_ft_in1k"
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.25, 0.25, 0.25)


@dataclass(frozen=True)
class PyroPrediction:
    """Single-image classification result."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class PyroPredictorConfig:
    """Configuration for the pyroptosis classifier."""

    model_path: Path = DEFAULT_MODEL_PATH
    classes_path: Path = DEFAULT_CLASSES_PATH
    image_size: int = DEFAULT_IMAGE_SIZE
    architecture: str = DEFAULT_ARCHITECTURE
    device: Optional[str] = None

    def normalized(self) -> "PyroPredictorConfig":
        """Return a config with normalized paths and a concrete device string."""
        return PyroPredictorConfig(
            model_path=Path(self.model_path).expanduser().resolve(),
            classes_path=Path(self.classes_path).expanduser().resolve(),
            image_size=int(self.image_size),
            architecture=self.architecture,
            device=self.device or _default_device(),
        )


class PyroptosisPredictor:
    """Reusable ConvNeXt predictor for apoptosis/pyroptosis classification."""

    def __init__(self, config: Optional[PyroPredictorConfig] = None) -> None:
        self.config = (config or PyroPredictorConfig()).normalized()
        if self.config.image_size <= 0:
            raise ValueError("image_size must be positive")

        self.idx_to_class = _load_class_mapping(self.config.classes_path)
        self.class_count = len(self.idx_to_class)
        self.device = torch.device(self.config.device)
        self.transform = _build_transform(self.config.image_size)
        self.model = self._load_model()

    def predict(self, image: ImageLike) -> PyroPrediction:
        """Classify one image region."""
        return self.predict_many([image])[0]

    def predict_many(
        self,
        images: Iterable[ImageLike],
        batch_size: int = 32,
    ) -> List[PyroPrediction]:
        """Classify image regions in batches.

        Parameters
        ----------
        images:
            Iterable of RGB, BGR, grayscale, or RGBA numpy arrays.
        batch_size:
            Number of regions to classify per forward pass.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        prepared_images = list(images)
        if not prepared_images:
            return []

        predictions: List[PyroPrediction] = []
        for start in range(0, len(prepared_images), batch_size):
            batch = prepared_images[start : start + batch_size]
            tensor = torch.stack(
                [self._preprocess(image) for image in batch]
            ).to(self.device, non_blocking=True)

            with torch.inference_mode():
                logits = self.model(tensor)
                probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()

            predictions.extend(self._to_predictions(probabilities))

        return predictions

    def _load_model(self) -> torch.nn.Module:
        if not self.config.model_path.is_file():
            raise FileNotFoundError(f"Pyroptosis checkpoint not found: {self.config.model_path}")

        model = create_model(
            self.config.architecture,
            pretrained=False,
            num_classes=self.class_count,
        )
        checkpoint = _torch_load(self.config.model_path, self.device)
        state_dict = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            LOGGER.warning(
                "Loaded pyroptosis checkpoint with missing keys=%s and unexpected keys=%s.",
                incompatible.missing_keys,
                incompatible.unexpected_keys,
            )

        model.to(self.device)
        model.eval()
        return model

    def _preprocess(self, image: ImageLike) -> torch.Tensor:
        image_rgb = _ensure_rgb_uint8(image)
        augmented = self.transform(image=image_rgb)
        return augmented["image"].float()

    def _to_predictions(self, probabilities: np.ndarray) -> List[PyroPrediction]:
        predictions: List[PyroPrediction] = []
        for row in probabilities:
            pred_idx = int(np.argmax(row))
            label = self.idx_to_class[pred_idx]
            predictions.append(
                PyroPrediction(
                    label=label,
                    confidence=float(row[pred_idx]),
                    probabilities={
                        self.idx_to_class[index]: float(probability)
                        for index, probability in enumerate(row)
                    },
                )
            )
        return predictions


def Pyro_Pred(image: ImageLike) -> str:
    """Return the predicted class label for a single image region.

    This preserves the original public API used by ``CellDetector``.
    """
    return get_predictor().predict(image).label


def predict(image: ImageLike) -> PyroPrediction:
    """Return the full prediction result for a single image region."""
    return get_predictor().predict(image)


def predict_many(images: Iterable[ImageLike], batch_size: int = 32) -> List[PyroPrediction]:
    """Return full prediction results for multiple image regions."""
    return get_predictor().predict_many(images, batch_size=batch_size)


def get_predictor(
    model_path: PathLike = DEFAULT_MODEL_PATH,
    classes_path: PathLike = DEFAULT_CLASSES_PATH,
    image_size: int = DEFAULT_IMAGE_SIZE,
    architecture: str = DEFAULT_ARCHITECTURE,
    device: Optional[str] = None,
) -> PyroptosisPredictor:
    """Return a cached predictor for the supplied configuration."""
    config = PyroPredictorConfig(
        model_path=Path(model_path),
        classes_path=Path(classes_path),
        image_size=image_size,
        architecture=architecture,
        device=device,
    ).normalized()
    return _get_cached_predictor(
        str(config.model_path),
        str(config.classes_path),
        config.image_size,
        config.architecture,
        config.device,
    )


@lru_cache(maxsize=4)
def _get_cached_predictor(
    model_path: str,
    classes_path: str,
    image_size: int,
    architecture: str,
    device: Optional[str],
) -> PyroptosisPredictor:
    config = PyroPredictorConfig(
        model_path=Path(model_path),
        classes_path=Path(classes_path),
        image_size=image_size,
        architecture=architecture,
        device=device,
    )
    return PyroptosisPredictor(config)


def _build_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_REFLECT_101,
            ),
            A.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
            ToTensorV2(),
        ]
    )


def _load_class_mapping(classes_path: Path) -> Dict[int, str]:
    if not classes_path.is_file():
        raise FileNotFoundError(f"Class mapping file not found: {classes_path}")

    with classes_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    idx_to_class = payload.get("idx_to_class")
    if not isinstance(idx_to_class, dict) or not idx_to_class:
        raise ValueError(
            f"Invalid class mapping in {classes_path}: expected non-empty 'idx_to_class'"
        )

    mapping = {int(index): str(label) for index, label in idx_to_class.items()}
    expected_indexes = set(range(len(mapping)))
    if set(mapping) != expected_indexes:
        raise ValueError(
            f"Invalid class mapping in {classes_path}: indexes must be contiguous "
            f"from 0 to {len(mapping) - 1}"
        )
    return mapping


def _ensure_rgb_uint8(image: ImageLike) -> ImageLike:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")

    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must be a grayscale, RGB/BGR, or RGBA numpy array")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(array)


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions do not support weights_only.
        return torch.load(path, map_location=device)


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = [
    "DEFAULT_ARCHITECTURE",
    "DEFAULT_CLASSES_PATH",
    "DEFAULT_IMAGE_SIZE",
    "DEFAULT_MODEL_PATH",
    "PyroPrediction",
    "PyroPredictorConfig",
    "PyroptosisPredictor",
    "Pyro_Pred",
    "get_predictor",
    "predict",
    "predict_many",
]
