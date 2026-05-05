"""Portable path constants for the anonymous OranBench code artifact."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = Path(os.environ.get("OSIM_DATA_DIR", str(REPO_ROOT / "data")))

KUAIRAND_DIR: Path = DATA_DIR / "kuairand"
KUAIRAND_RAW: Path = KUAIRAND_DIR / "raw"
KUAIRAND_PROCESSED: Path = KUAIRAND_DIR / "processed"
