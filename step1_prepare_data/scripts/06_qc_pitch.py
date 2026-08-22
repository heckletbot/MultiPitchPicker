#!/usr/bin/env python3
"""音级质检：估计基频 → winner_midi。与标称 midi 不一致的行会被 synth 自动跳过。

马林巴泛音强，估计时把搜索窗限制在标称音高附近，减少八度跳错。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import REPO_ROOT, ensure_work, load_mono, read_jsonl, write_jsonl  # noqa: E402


def _midi_hz(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def estimate_midi(y: np.ndarray, sr: int, expected: int) -> tuple[int, float]:
    import librosa

    # 只用起音后 80–600ms，避开敲击噪声和过长余振
    n0 = int(0.08 * sr)
    n1 = min(len(y), int(0.60 * sr))
    seg = y[n0:n1] if n1 - n0 > sr * 0.12 else y
    fmin = _midi_hz(expected - 3)
    fmax = _midi_hz(expected + 3)
    f0, _, voiced = librosa.pyin(
        seg, fmin=max(40.0, fmin), fmax=min(sr / 2 - 1, fmax),
        sr=sr, fill_na=np.nan,
    )
    good = f0[np.isfinite(f0) & (voiced > 0.4)] if voiced is not None else f0[np.isfinite(f0)]
    if good.size < 3:
        # 放宽到 ±12 半音再试
        f0, _, voiced = librosa.pyin(
            seg, fmin=max(40.0, _midi_hz(expected - 12)),
            fmax=min(sr / 2 - 1, _midi_hz(expected + 12)),
            sr=sr, fill_na=np.nan,
        )
        good = f0[np.isfinite(f0)]
    if good.size == 0:
        return expected, 0.0
    hz = float(np.median(good))
    midi = int(round(69 + 12 * np.log2(hz / 440.0)))
    conf = float(np.clip(good.size / max(1, f0.size), 0, 1))
    return midi, conf


def main() -> None:
    ap = argparse.ArgumentParser(description="填写 winner_midi 音级质检")
    ap.add_argument("--index", default=str(ensure_work("singles") / "index.raw.jsonl"))
    ap.add_argument("--out", default=str(ensure_work("singles") / "index.qc.jsonl"))
    ap.add_argument("--report", default=str(ensure_work("reports") / "pitch_qc.json"))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.index))
    if not rows:
        raise SystemExit(f"空索引: {args.index}")

    mismatch, fail = [], 0
    out_rows = []
    for i, r in enumerate(rows, 1):
        wav = Path(r["wav"])
        if not wav.is_file():
            wav = REPO_ROOT / r["wav"]
        try:
            y, sr = load_mono(wav, sr=None)
            win, conf = estimate_midi(y, sr, int(r["midi"]))
        except Exception as e:  # noqa: BLE001 — 单条失败不中断整库
            win, conf = int(r["midi"]), 0.0
            fail += 1
            print(f"  失败 {wav.name}: {e}", flush=True)
        q = dict(r)
        q["winner_midi"] = int(win)
        q["pitch_conf"] = round(conf, 3)
        if int(win) != int(r["midi"]):
            mismatch.append({**q, "wav": r["wav"]})
        out_rows.append(q)
        if i % 40 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  mismatch={len(mismatch)}", flush=True)

    write_jsonl(Path(args.out), out_rows)
    import json
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "n": len(out_rows),
        "mismatch": len(mismatch),
        "fail": fail,
        "keep_if_match": len(out_rows) - len(mismatch),
        "examples": mismatch[:20],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"不一致 {len(mismatch)}/{len(out_rows)} -> {args.out}")
    print(f"报告 {args.report}")


if __name__ == "__main__":
    main()
