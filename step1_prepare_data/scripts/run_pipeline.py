#!/usr/bin/env python3
"""一步跑完数据采集之后的预处理（下载需单独执行，避免误打 Iowa 服务器）。

默认路径：已有 work/raw/iowa_marimba 或 --raw。
自定义乐器：python run_pipeline.py --raw 你的单音或半音阶目录
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STEP = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import WORK, python_bin  # noqa: E402


def _run(script: str, extra: list[str] | None = None) -> None:
    cmd = [python_bin(), str(SCRIPTS / script), *(extra or [])]
    print("\n===", script, "===")
    subprocess.run(cmd, cwd=str(STEP), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="step1 预处理流水线")
    ap.add_argument("--raw", default=str(WORK / "raw" / "iowa_marimba"),
                    help="原始音频目录（Iowa 下载物或自己的录音）")
    ap.add_argument("--from", dest="start", default="split",
                    choices=["split", "resample", "index", "qc", "clarity", "report", "synth"],
                    help="从哪一步开始（前面的已跑过可跳过）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-qc", action="store_true", help="跳过 pyin 音级质检（CPU 慢时）")
    ap.add_argument("--skip-synth", action="store_true", help="只做单音库，不合成 8s 片段")
    args = ap.parse_args()

    raw = Path(args.raw)
    if not raw.exists() and args.start == "split":
        raise SystemExit(
            f"找不到 {raw}\n"
            f"马林巴请先: python scripts/01_download_iowa_marimba.py\n"
            f"其它乐器: python scripts/02_ingest_raw.py --src <目录>"
        )

    order = ["split", "resample", "index", "qc", "clarity", "report", "synth"]
    start_i = order.index(args.start)

    if start_i <= 0:
        _run("03_split_chromatic.py", ["--src", str(raw)])
    if start_i <= 1:
        _run("04_resample_16k.py")
    if start_i <= 2:
        _run("05_build_index.py")
    if start_i <= 3 and not args.skip_qc:
        _run("06_qc_pitch.py")
    elif start_i <= 3 and args.skip_qc:
        # 无质检时把 raw 索引直接当 qc
        src = WORK / "singles" / "index.raw.jsonl"
        dst = WORK / "singles" / "index.qc.jsonl"
        dst.write_bytes(src.read_bytes())
        print("跳过音级质检，复制 index.raw.jsonl -> index.qc.jsonl")
    if start_i <= 4:
        _run("07_filter_clarity.py")
    if start_i <= 5:
        try:
            _run("08_report_coverage.py")
        except subprocess.CalledProcessError as e:
            print("覆盖率有缺口（exit 2）。可继续合成，但缺音高的 MIDI 不会出现在标签里。")
            if e.returncode not in (0, 2):
                raise
    if start_i <= 6 and not args.skip_synth:
        extra = ["--smoke"] if args.smoke else []
        _run("09_make_synth_and_cqt.py", extra)
    print("\nstep1 完成。")


if __name__ == "__main__":
    main()
