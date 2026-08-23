#!/usr/bin/env python3
"""step3 共用路径与 checkpoint 选取。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

STEP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STEP_DIR.parent
WORK = STEP_DIR / "work"
CONFIG_PATH = STEP_DIR / "config_eval.json"
STEP2_SELECTED = REPO_ROOT / "step2_train" / "work" / "selected.json"


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


def load_selected() -> tuple[dict | None, Path | None]:
    """优先本步 work/selected.json，否则 step2 的 selected.json。"""
    for p in (WORK / "selected.json", STEP2_SELECTED):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8")), p
    return None, None


def pick_ckpt(cfg: dict) -> tuple[Path, dict]:
    """从 selected.json 读 ckpt；没有则 mix 优先，否则 base。"""
    sel, _src = load_selected()
    if sel and sel.get("ckpt"):
        ckpt = Path(sel["ckpt"])
        if not ckpt.is_file():
            ckpt = REPO_ROOT / ckpt
        if ckpt.is_file():
            return ckpt, sel
    mix = ckpt_path(cfg["mix_run"])
    base = ckpt_path(cfg["base_run"])
    if mix.is_file():
        return mix, {}
    if base.is_file():
        return base, {}
    raise FileNotFoundError(
        "没有 checkpoint。请先跑 step2_train，或把 best.pt 放到 data/runs/{mix,base}/"
    )


def as_posix(path: Path | str) -> str:
    return str(path).replace("\\", "/")
