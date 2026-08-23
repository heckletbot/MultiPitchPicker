#!/usr/bin/env python3
"""校准图内 CQT 与 librosa 的每 bin log 偏置，写出 data/runs/torch_cqt_bias.pt。

换乐器或改 MP_CQT_GAMMA 后必须重跑。调用根目录 torch_cqt.py，不改算法。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import load_config, repo_path, run_repo  # noqa: E402


def main() -> None:
    cfg = load_config()
    cal = cfg["calibrate"]
    ap = argparse.ArgumentParser(description="校准图内 CQT bias")
    ap.add_argument("--check", default=None, help="jsonl，默认 validation.jsonl")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default=None, help="bias 保存路径")
    args = ap.parse_args()

    check = Path(args.check) if args.check else repo_path(cal["check_jsonl"])
    if not check.is_file():
        raise SystemExit(f"找不到 {check}，请先跑 step1。")
    out = Path(args.out) if args.out else repo_path(cfg["bias"])
    n = args.n if args.n is not None else int(cal["n"])
    extra = ["--check", str(check), "--n", str(n), "--save-bias", str(out)]
    run_repo("torch_cqt.py", extra)
    print(f"bias -> {out}")


if __name__ == "__main__":
    main()
