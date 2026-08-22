#!/usr/bin/env python3
"""把合格单音库接到仓库契约目录，并跑本目录 synth_clips + precompute。

产物：
  <repo>/data/singles/*.wav + index.jsonl
  <repo>/data/synth/{train,validation,test}.jsonl 与 wav
  每条 wav 旁的 .cqt285.npy 缓存

这一步结束即可进入训练（train.py）。真录独奏不在本步范围。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    REPO_ROOT, ensure_work, load_config, python_bin, read_jsonl, write_jsonl,
)


def _run(cmd: list[str]) -> None:
    print(">", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def install_singles(index_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(index_path)
    out_rows = []
    for r in rows:
        src = Path(r["wav"])
        if not src.is_file():
            src = REPO_ROOT / r["wav"]
        dest = dest_dir / src.name
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        q = dict(r)
        q["wav"] = dest.relative_to(REPO_ROOT).as_posix()
        out_rows.append(q)
    dest_index = dest_dir / "index.jsonl"
    write_jsonl(dest_index, out_rows)
    print(f"安装 {len(out_rows)} 条单音 -> {dest_dir}")
    return dest_index


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="安装单音库并合成训练片段 + CQT")
    ap.add_argument("--index", default=str(ensure_work("singles") / "index.jsonl"))
    ap.add_argument("--singles-dest", default=str(REPO_ROOT / "data" / "singles"))
    ap.add_argument("--synth-out", default=str(REPO_ROOT / "data" / "synth"))
    ap.add_argument("--smoke", action="store_true", help="小规模试跑（32/8/8）")
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-cqt", action="store_true")
    args = ap.parse_args()

    index = Path(args.index)
    if not index.is_file():
        raise SystemExit(f"找不到 {index}，请先跑到 07_filter_clarity.py")

    dest_index = install_singles(index, Path(args.singles_dest))
    syn = cfg["synth"]
    n_train, n_val, n_test = syn["n_train"], syn["n_val"], syn["n_test"]
    if args.smoke:
        n_train, n_val, n_test = 32, 8, 8

    train_t = ",".join(cfg["train_timbres"])
    hold_t = ",".join(cfg["heldout_timbres"])
    py = python_bin()
    if not args.skip_synth:
        cmd = [
            py, str(_SCRIPTS / "synth_clips.py"),
            "--index", str(dest_index),
            "--out", args.synth_out,
            "--n-train", str(n_train), "--n-val", str(n_val), "--n-test", str(n_test),
            "--workers", str(syn["workers"]),
        ]
        if train_t:
            cmd += ["--train-timbres", train_t]
        if hold_t:
            cmd += ["--heldout-timbres", hold_t]
        _run(cmd)
    if not args.skip_cqt:
        synth = Path(args.synth_out)
        jsonls = [synth / "train.jsonl", synth / "validation.jsonl", synth / "test.jsonl"]
        missing = [p for p in jsonls if not p.is_file()]
        if missing:
            raise SystemExit(f"缺少 {missing}，请先合成")
        _run([
            py, str(_SCRIPTS / "precompute.py"),
            "--jsonl", *[str(p) for p in jsonls],
            "--workers", str(max(2, syn["workers"] // 2)),
        ])
    print("预处理完成。产物在仓库根目录 data/singles/ 与 data/synth/。")


if __name__ == "__main__":
    main()
