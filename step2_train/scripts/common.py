#!/usr/bin/env python3
"""step2 共用路径与子进程调用。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STEP_DIR.parent
WORK = STEP_DIR / "work"
CONFIG_PATH = STEP_DIR / "config_train.json"


def load_config(path: Path | None = None) -> dict:
    return json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))


def ensure_work(*parts: str) -> Path:
    d = WORK.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else REPO_ROOT / p


def python_bin() -> str:
    return sys.executable


def run_repo(script: str, extra: list[str]) -> None:
    cmd = [python_bin(), str(REPO_ROOT / script), *extra]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def ckpt_path(run_name: str) -> Path:
    return REPO_ROOT / "data" / "runs" / run_name / "best.pt"


def pick_ckpt(cfg: dict) -> Path:
    mix = ckpt_path(cfg["mix"]["run_name"])
    base = ckpt_path(cfg["base"]["run_name"])
    if mix.is_file():
        return mix
    if base.is_file():
        return base
    raise FileNotFoundError("没有 data/runs/*/best.pt，请先训练")
