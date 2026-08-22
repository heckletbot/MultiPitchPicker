#!/usr/bin/env python3
"""生成仓库契约的 data/singles/index.jsonl 字段：wav, midi, dynamic, timbre。

wav 写成相对仓库根目录的路径（data/singles/...），方便 synth_clips.py 直接读。
本步只写 work/singles/index.raw.jsonl；09 会拷到仓库 data/singles/。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ensure_work, read_jsonl, repo_rel, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="写单音索引 jsonl")
    ap.add_argument("--src", default=str(ensure_work("singles") / "resample_manifest.jsonl"))
    ap.add_argument("--out", default=str(ensure_work("singles") / "index.raw.jsonl"))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.src))
    if not rows:
        raise SystemExit(f"空清单: {args.src}，请先跑 04_resample_16k.py")
    out_rows = []
    for r in rows:
        out_rows.append({
            "wav": repo_rel(Path(r["wav"])),
            "midi": int(r["midi"]),
            "dynamic": r["dynamic"],
            "timbre": r["timbre"],
        })
    write_jsonl(Path(args.out), out_rows)
    n_midi = len({r["midi"] for r in out_rows})
    n_tim = len({r["timbre"] for r in out_rows})
    print(f"{len(out_rows)} 行, {n_midi} 个音高, {n_tim} 种 timbre -> {args.out}")


if __name__ == "__main__":
    main()
