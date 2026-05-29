"""Package CLI entry points for DeathScope."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from . import CellDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

LOGGER = logging.getLogger("deathscope.batch_predict")


class TqdmStream:
    """Send logging output through tqdm so progress bars remain stable."""

    def write(self, msg: str) -> None:
        msg = msg.rstrip("\n")
        if msg:
            tqdm.write(msg, file=sys.stderr)

    def flush(self) -> None:
        sys.stderr.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch DeathScope heatmap prediction over one or more dataset directories.",
    )
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "examples")
    parser.add_argument("--pattern", default="*")
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "models" / "yolo" / "deathscope_yolov8l.pt",
    )
    parser.add_argument("--desired-coverage", type=int, default=1)
    parser.add_argument("--cut-off", type=float, default=0.0)
    parser.add_argument("--green-cut-off", type=float, default=30.0)
    parser.add_argument("--green-img", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pred-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--incucyte", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pyro", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "outputs" / "logs")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser


def configure_logging(log_dir: Path, log_level: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"batch_predict_{timestamp}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, log_level))

    console_handler = logging.StreamHandler(TqdmStream())  # type: ignore[arg-type]
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    return log_file


def discover_datasets(input_root: Path, pattern: str, limit: int | None) -> List[Path]:
    datasets = sorted(path for path in input_root.glob(pattern) if path.is_dir())
    if limit is not None:
        datasets = datasets[:limit]
    return datasets


def count_phase_images(dataset_dir: Path) -> int:
    phase_dir = dataset_dir / "Phase"
    if not phase_dir.is_dir():
        return 0
    return sum(1 for path in phase_dir.iterdir() if path.is_file())


def write_summary(log_dir: Path, run_name: str, summary_rows: Iterable[dict]) -> Path:
    summary_path = log_dir / f"{run_name}_summary.json"
    payload = {
        "run_name": run_name,
        "generated_at": datetime.now().isoformat(),
        "datasets": list(summary_rows),
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path


def main() -> int:
    args = build_parser().parse_args()

    log_file = configure_logging(args.log_dir, args.log_level)
    run_name = log_file.stem

    if not args.input_root.exists():
        LOGGER.error("Input root does not exist: %s", args.input_root)
        return 1
    if not args.model.exists():
        LOGGER.error("Model file does not exist: %s", args.model)
        return 1

    datasets = discover_datasets(args.input_root, args.pattern, args.limit)
    if not datasets:
        LOGGER.error("No dataset directories found in %s matching %r.", args.input_root, args.pattern)
        return 1

    detector = CellDetector(args.model)
    summary_rows = []
    outer = tqdm(datasets, desc="Datasets", unit="dataset", file=sys.stdout, dynamic_ncols=True, leave=True, position=0)

    for dataset_dir in outer:
        dataset_name = dataset_dir.name
        phase_count = count_phase_images(dataset_dir)
        outer.set_description(f"Datasets [{dataset_name}]")
        outer.set_postfix(images=phase_count)

        inner = tqdm(total=phase_count, desc="  Images", unit="img", file=sys.stdout, dynamic_ncols=True, leave=False, position=1)
        cell_logger = logging.getLogger("deathscope.cell_detector")
        original_info = cell_logger.info

        def _advancing_info(msg, *a, _inner=inner, _orig=original_info, **kw):
            _orig(msg, *a, **kw)
            _inner.update(1)

        cell_logger.info = _advancing_info  # type: ignore[method-assign]
        started = time.perf_counter()
        status = "success"
        error_message = None

        try:
            if args.pyro:
                detector.predict_with_heatmap_batch_pyro(
                    dataset_dir,
                    dataset_name,
                    desired_coverage=args.desired_coverage,
                    green_img=args.green_img,
                    pred_images=args.pred_images,
                    cut_off=args.cut_off,
                    g_cut_off=args.green_cut_off,
                    incucyte=args.incucyte,
                    binary=args.binary,
                )
            else:
                detector.predict_with_heatmap_batch(
                    dataset_dir,
                    dataset_name,
                    desired_coverage=args.desired_coverage,
                    green_img=args.green_img,
                    pred_images=args.pred_images,
                    cut_off=args.cut_off,
                    g_cut_off=args.green_cut_off,
                    incucyte=args.incucyte,
                    binary=args.binary,
                )
        except Exception as exc:  # pragma: no cover
            status = "failed"
            error_message = str(exc)
            LOGGER.exception("Dataset %s failed.", dataset_name)
        finally:
            cell_logger.info = original_info  # type: ignore[method-assign]
            inner.n = phase_count
            inner.refresh()
            inner.close()
            elapsed_seconds = round(time.perf_counter() - started, 3)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_dir": str(dataset_dir),
                    "phase_image_count": phase_count,
                    "status": status,
                    "elapsed_seconds": elapsed_seconds,
                    "error": error_message,
                }
            )

    outer.set_description("Datasets")
    outer.set_postfix_str("Done")
    outer.close()

    summary_path = write_summary(args.log_dir, run_name, summary_rows)
    success_count = sum(1 for row in summary_rows if row["status"] == "success")
    failure_count = len(summary_rows) - success_count
    LOGGER.info("Summary file: %s", summary_path)
    tqdm.write(f"\nBatch complete: {success_count} succeeded, {failure_count} failed.", file=sys.stdout)
    return 0 if failure_count == 0 else 2


__all__ = ["build_parser", "main"]
