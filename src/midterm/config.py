"""Project paths and configuration loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "settings.yaml"
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

for _p in (RAW, INTERIM, PROCESSED, FIG, TAB):
    _p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def settings() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)
