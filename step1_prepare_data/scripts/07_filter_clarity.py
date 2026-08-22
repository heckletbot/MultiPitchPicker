#!/usr/bin/env python3
"""按音高保留最清晰的 K 条（马林巴原实验为 top7）。

3 槌 × 3 力度 = 每音最多 9 条，丢掉最糊的 2 条。自定义乐器样本少时可把
config_prep.json 的 clarity_top_k 调大，或 --top-k 0 表示全留。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    REPO_ROOT, ensure_work, load_config, load_mono, read_jsonl, write_jsonl,
)


def _hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def clarity_score(y: np.ndarray, sr: int, midi: int) -> float:
    """起音对比度 × 基频谐波能量占比。越大越干净。"""
    if y.size < sr * 0.1:
        return 0.0
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-9)
    # 起音：前 30ms 的峰值 vs 整段中位数
    att = float(np.max(np.abs(y[: max(1, int(0.03 * sr))])))
    med = float(np.median(np.abs(y)) + 1e-9)
    attack = np.clip(att / med, 0.0, 80.0)
    # 谐波柱：f0..4f0 的窄带能量 / 总能量
    n_fft = 4096
    spec = np.abs(np.fft.rfft(y[: min(len(y), int(0.5 * sr))], n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f0 = _hz(midi)
    harm = 0.0
    for h in (1, 2, 3, 4):
        f = h * f0
        if f >= sr / 2:
            break
        band = (freqs > f * 0.97) & (freqs < f * 1.03)
        harm += float(spec[band].sum())
    tot = float(spec.sum()) + 1e-9
    return float(np.log1p(attack) * (harm / tot) * (peak / (med + 1e-6)))


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="每音高保留 top-K 清晰样本")
    ap.add_argument("--index", default=str(ensure_work("singles") / "index.qc.jsonl"))
    ap.add_argument("--out", default=str(ensure_work("singles") / "index.jsonl"))
    ap.add_argument("--top-k", type=int, default=int(cfg["clarity_top_k"]),
                    help="每 MIDI 保留条数；0=全留")
    ap.add_argument("--keep-mismatch", action="store_true",
                    help="保留 winner_midi≠midi 的行（默认丢掉，与 synth 行为一致）")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.index))
    if not rows:
        raise SystemExit(f"空索引: {args.index}")

    kept = []
    for r in rows:
        if (not args.keep_mismatch) and "winner_midi" in r and int(r["winner_midi"]) != int(r["midi"]):
            continue
        wav = Path(r["wav"])
        if not wav.is_file():
            wav = REPO_ROOT / r["wav"]
        y, sr = load_mono(wav, sr=None)
        q = dict(r)
        q["clarity"] = round(clarity_score(y, sr, int(r["midi"])), 4)
        kept.append(q)

    by_midi: dict[int, list[dict]] = defaultdict(list)
    for r in kept:
        by_midi[int(r["midi"])].append(r)
    out_rows = []
    for midi, group in sorted(by_midi.items()):
        group.sort(key=lambda x: -x["clarity"])
        take = group if args.top_k <= 0 else group[: args.top_k]
        out_rows.extend(take)

    out_rows.sort(key=lambda r: (r["midi"], r["timbre"], r["dynamic"]))
    write_jsonl(Path(args.out), out_rows)
    pm: dict[str, int] = defaultdict(int)
    for r in out_rows:
        pm[str(r["midi"])] += 1
    report = {
        "in": len(rows),
        "after_pitch_filter": len(kept),
        "out": len(out_rows),
        "top_k": args.top_k,
        "per_midi": dict(sorted(pm.items(), key=lambda kv: int(kv[0]))),
    }
    rp = ensure_work("reports") / "clarity.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} -> 音级过滤 {len(kept)} -> top{args.top_k or 'all'} = {len(out_rows)}")
    print(f"写出 {args.out}")
    print(f"报告 {rp}")


if __name__ == "__main__":
    main()
