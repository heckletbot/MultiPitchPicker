#!/usr/bin/env python3
"""step4 共用路径。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STEP_DIR.parent
CONFIG_PATH = STEP_DIR / "config_api.json"
STEP3_SELECTED = REPO_ROOT / "step3_eval_export" / "work" / "selected.json"
STEP2_SELECTED = REPO_ROOT / "step2_train" / "work" / "selected.json"


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def python_bin() -> str:
    return sys.executable


def default_assets(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return repo_path(cfg["assets"])


def load_selected_thresh() -> float | None:
    for p in (STEP3_SELECTED, STEP2_SELECTED):
        if p.is_file():
            obj = json.loads(p.read_text(encoding="utf-8"))
            if obj.get("best_thresh") is not None:
                return float(obj["best_thresh"])
    return None
