#!/usr/bin/env python3
"""阶段 A：纯合成域训基座，按 val note-level F1 存 data/runs/base/best.pt。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import load_config, run_repo  # noqa: E402


def main() -> None:
    cfg = load_config()
    b = cfg["base"]
    ap = argparse.ArgumentParser(description="训基座")
    ap.add_argument("--smoke", action="store_true", help="1 epoch 冒烟")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lookahead", action="store_true", help="低延迟非对称感受野")
    args = ap.parse_args()

    epochs = 1 if args.smoke else (args.epochs or b["epochs"])
    extra = [
        "--data", cfg["synth_dir"],
        "--run-name", b["run_name"],
        "--epochs", str(epochs),
        "--lr", str(b["lr"]),
        "--batch-size", str(b["batch_size"]),
        "--workers", str(b["workers"]),
    ]
    if args.lookahead:
        extra.append("--lookahead")
    run_repo("train.py", extra)
    print(f"基座权重: data/runs/{b['run_name']}/best.pt")


if __name__ == "__main__":
    main()
