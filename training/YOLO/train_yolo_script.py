#!/usr/bin/env python3
"""Train the DeathScope YOLO cell detector with Ultralytics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA = REPO_ROOT / "configs" / "cell_dataset.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a DeathScope YOLO detection model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the Ultralytics dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8l.pt",
        help="Pretrained weights or model config to initialize training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Batch size.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=128,
        help="Training image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Training device, for example 'cpu', '0', or '0,1'.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Output project directory used by Ultralytics.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="deathscope_yolo",
        help="Run name inside the project directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained weights when supported by the selected model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in the selected run directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> Path:
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    if args.patience < 0:
        raise ValueError("--patience must be non-negative")

    return data_path


def build_train_kwargs(args: argparse.Namespace, data_path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "project": args.project,
        "name": args.name,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "pretrained": args.pretrained,
        "resume": args.resume,
    }
    if args.device is not None:
        kwargs["device"] = args.device
    return kwargs


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    data_path = validate_args(args)

    model = YOLO(args.model)
    model.train(**build_train_kwargs(args, data_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
