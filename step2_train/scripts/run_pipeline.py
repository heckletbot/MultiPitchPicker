#!/usr/bin/env python3
"""步骤二一键：检查 → 训基座 →（可选）伪标微调 → 合成 test 扫阈值。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import load_config, python_bin, repo_path  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [python_bin(), str(SCRIPTS / script), *(extra or [])]
    print("\n===", script, "===")
    subprocess.run(cmd, cwd=str(SCRIPTS.parent), check=True)


def _has_real(cfg: dict) -> bool:
    real = repo_path(cfg["real_dir"])
    if not real.is_dir():
        return False
    return any(real.glob(ext) for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"))


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="step2 训练流水线")
    ap.add_argument("--smoke", action="store_true", help="各阶段 1 epoch，只验证能跑通")
    ap.add_argument("--skip-real", action="store_true", help="即使有真录也不微调")
    ap.add_argument("--from", dest="start", default="check",
                    choices=["check", "base", "pseudo", "mix", "eval"])
    args = ap.parse_args()

    order = ["check", "base", "pseudo", "mix", "eval"]
    i0 = order.index(args.start)
    smoke = ["--smoke"] if args.smoke else []
    do_real = _has_real(cfg) and not args.skip_real

    if i0 <= 0:
        _run("01_check_ready.py")
    if i0 <= 1:
        _run("02_train_base.py", smoke)
    if i0 <= 2 and do_real:
        _run("03_pseudo_label.py")
    elif i0 <= 2 and not do_real:
        print("无 data/real（或 --skip-real），跳过伪标签，最终模型 = base")
    if i0 <= 3 and do_real:
        _run("04_merge_and_finetune.py", smoke)
    if i0 <= 4:
        _run("05_eval_and_sweep.py")
    print("\nstep2 完成。权重与推荐阈值见 step2_train/work/selected.json")
    print("下一步：导出 ONNX（步骤三）。")


if __name__ == "__main__":
    main()
