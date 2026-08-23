#!/usr/bin/env python3
"""基座对 data/real 独奏打低阈值伪标签，并预计算 CQT。

无真录可跳过。混有人声/其它乐器的录音不要放进来。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ckpt_path, load_config, python_bin, repo_path, run_repo  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="真录伪标签")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--real-dir", default=cfg["real_dir"])
    ap.add_argument("--out", default=cfg["pseudo_dir"])
    ap.add_argument("--thresh", type=float, default=cfg["pseudo"]["thresh"])
    args = ap.parse_args()

    real = repo_path(args.real_dir)
    songs = [p for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a") for p in real.glob(ext)]
    if not songs:
        raise SystemExit(f"{real} 下没有音频。见 第二步模型训练.md「真录独奏采集」")

    ckpt = args.ckpt or str(ckpt_path(cfg["base"]["run_name"]))
    if not Path(ckpt).is_file() and not repo_path(ckpt).is_file():
        raise SystemExit(f"找不到 {ckpt}，请先跑 02_train_base.py")

    run_repo("pseudo_label.py", [
        "--ckpt", ckpt,
        "--real-dir", args.real_dir,
        "--out", args.out,
        "--thresh", str(args.thresh),
        "--min-notes", str(cfg["pseudo"]["min_notes"]),
    ])
    jsonl = repo_path(args.out) / "train.jsonl"
    if not jsonl.is_file():
        raise SystemExit("伪标没有写出 train.jsonl")
    pre = repo_path("step1_prepare_data/scripts/precompute.py")
    if not pre.is_file():
        raise SystemExit(f"找不到 {pre}，请先放入第一步")
    cmd = [python_bin(), str(pre), "--jsonl", str(jsonl),
           "--workers", str(max(2, cfg["base"]["workers"] // 2))]
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_path(".")), check=True)
    print(f"伪标清单: {jsonl}")


if __name__ == "__main__":
    main()
