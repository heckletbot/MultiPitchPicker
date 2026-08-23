#!/usr/bin/env python3
"""合成 8000 + 伪标×2 混合，从基座微调得到 data/runs/mix/best.pt。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ckpt_path, load_config, repo_path, run_repo  # noqa: E402


def main() -> None:
    cfg = load_config()
    m = cfg["mix"]
    ap = argparse.ArgumentParser(description="混合微调")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    synth_train = repo_path(cfg["synth_dir"]) / "train.jsonl"
    pseudo = repo_path(cfg["pseudo_dir"]) / "train.jsonl"
    mixed = repo_path(cfg["synth_dir"]) / "train_mixed.jsonl"
    if not pseudo.is_file():
        raise SystemExit(f"找不到 {pseudo}，请先跑 03_pseudo_label.py，或没有真录就停在 base")

    run_repo("merge_jsonl.py", [
        "--synth", str(synth_train),
        "--synth-take", str(m["synth_take"]),
        "--pseudo", str(pseudo),
        "--pseudo-repeat", str(m["pseudo_repeat"]),
        "--out", str(mixed),
    ])
    epochs = 1 if args.smoke else (args.epochs or m["epochs"])
    run_repo("train.py", [
        "--data", cfg["synth_dir"],
        "--train-jsonl", str(mixed),
        "--init", str(ckpt_path(cfg["base"]["run_name"])),
        "--run-name", m["run_name"],
        "--epochs", str(epochs),
        "--lr", str(m["lr"]),
        "--batch-size", str(m["batch_size"]),
        "--workers", str(m["workers"]),
    ])
    print(f"最终权重: data/runs/{m['run_name']}/best.pt")


if __name__ == "__main__":
    main()
