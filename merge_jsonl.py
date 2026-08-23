#!/usr/bin/env python3
"""把合成 train.jsonl 与真实伪标 jsonl 混成一条训练清单。

默认配比（马林巴原实验值，可调）：伪标整表重复 2 遍 + 合成随机抽 8000 条。
不改 wav，只重写 jsonl。

用法：
  python3 merge_jsonl.py \
      --synth data/synth/train.jsonl --synth-take 8000 \
      --pseudo data/pseudo/train.jsonl --pseudo-repeat 2 \
      --out data/synth/train_mixed.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--synth-take", type=int, default=8000, help="合成条数；0=全用")
    ap.add_argument("--pseudo-repeat", type=int, default=2, help="伪标重复遍数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    synth = _load(Path(args.synth))
    pseudo = _load(Path(args.pseudo))
    rng = random.Random(args.seed)
    if args.synth_take and args.synth_take < len(synth):
        synth = rng.sample(synth, args.synth_take)

    rows = []
    for k in range(args.pseudo_repeat):
        for r in pseudo:
            q = dict(r)
            q["id"] = f"{r['id']}_r{k}"
            rows.append(q)
    rows.extend(synth)
    rng.shuffle(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"mixed {len(rows)}  (pseudo×{args.pseudo_repeat}={len(pseudo)*args.pseudo_repeat} "
          f"+ synth {len(synth)}) -> {out}")


if __name__ == "__main__":
    main()
