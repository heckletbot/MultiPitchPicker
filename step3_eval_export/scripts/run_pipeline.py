#!/usr/bin/env python3
"""步骤三一键：合成 test 评测 → 校准 CQT bias → 导出 ONNX + meta（可选真录健康度）。"""
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
    ap = argparse.ArgumentParser(description="step3 测评 + 导出流水线")
    ap.add_argument("--from", dest="start", default="eval",
                    choices=["eval", "calibrate", "export", "health"])
    ap.add_argument("--reuse-selected", action="store_true",
                    help="跳过重跑合成 test，用已有 selected.json 的 τ 直接导出")
    ap.add_argument("--skip-health", action="store_true", help="即使有 data/real 也不跑健康度")
    args = ap.parse_args()

    order = ["eval", "calibrate", "export", "health"]
    i0 = order.index(args.start)
    do_health = _has_real(cfg) and not args.skip_health

    if i0 <= 0:
        extra = ["--reuse-selected"] if args.reuse_selected else []
        _run("01_eval_testset.py", extra)
    if i0 <= 1:
        _run("03_calibrate_cqt.py")
    if i0 <= 2:
        _run("04_export_onnx.py")
    if i0 <= 3 and do_health:
        _run("02_real_health.py")
    elif i0 <= 3 and not do_health:
        print("无 data/real（或 --skip-health），跳过真录健康度。")

    print("\nstep3 完成。")
    print("  测评报告: step3_eval_export/work/test_report.json")
    print("  ONNX:     export/model.onnx")
    print("  meta:     export/meta.json")
    print("下一步：step4_api HTTP 转录接口。")


if __name__ == "__main__":
    main()
