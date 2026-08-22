#!/usr/bin/env python3
"""覆盖率报告：每个 MIDI 有哪些 timbre/dynamic，对照 config 音域找缺口。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ensure_work, load_config, midi_to_note, read_jsonl  # noqa: E402


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="单音库覆盖率")
    ap.add_argument("--index", default=str(ensure_work("singles") / "index.jsonl"))
    ap.add_argument("--out", default=str(ensure_work("reports") / "coverage.json"))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.index))
    if not rows:
        raise SystemExit(f"空索引: {args.index}")

    lo, hi = int(cfg["midi_lo"]), int(cfg["midi_hi"])
    want_t = [t.lower() for t in cfg["timbres"]]
    want_d = ["pp", "mf", "ff"]
    by: dict[int, set[tuple[str, str]]] = defaultdict(set)
    for r in rows:
        by[int(r["midi"])].add((r["timbre"].lower(), r["dynamic"].lower()))

    missing_pitch = [m for m in range(lo, hi + 1) if m not in by]
    thin = []
    for m in range(lo, hi + 1):
        have = by.get(m, set())
        lack = []
        for t in want_t:
            for d in want_d:
                if (t, d) not in have:
                    lack.append(f"{t}.{d}")
        if lack:
            thin.append({"midi": m, "note": midi_to_note(m), "n": len(have), "missing": lack})

    report = {
        "n_rows": len(rows),
        "midi_lo": lo,
        "midi_hi": hi,
        "n_pitches_present": len(by),
        "n_pitches_expected": hi - lo + 1,
        "missing_pitches": [{"midi": m, "note": midi_to_note(m)} for m in missing_pitch],
        "incomplete_cells": thin[:80],
        "n_incomplete": len(thin),
        "timbres_in_index": sorted({r["timbre"] for r in rows}),
        "train_timbres": cfg["train_timbres"],
        "heldout_timbres": cfg["heldout_timbres"],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    present = sorted(by)
    holes = [m for m in missing_pitch if present and present[0] < m < present[-1]]
    print(f"样本 {len(rows)}，音高 {len(by)}/{hi - lo + 1}，缺音 {len(missing_pitch)}（音域内空洞 {len(holes)}），格子不齐 {len(thin)}")
    if missing_pitch[:12]:
        print("缺音举例:", ", ".join(midi_to_note(m) for m in missing_pitch[:12]))
    print(f"报告 {args.out}")
    if holes:
        print("音域中段有空洞，合成时这些 MIDI 不会出现。请补采后再跑。")
        sys.exit(2)


if __name__ == "__main__":
    main()
