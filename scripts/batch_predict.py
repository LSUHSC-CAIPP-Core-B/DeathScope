#!/usr/bin/env python3
"""Thin wrapper for the packaged DeathScope batch CLI."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from deathscope.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
