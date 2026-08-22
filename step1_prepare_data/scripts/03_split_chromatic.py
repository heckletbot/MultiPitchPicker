#!/usr/bin/env python3
"""把 Iowa 式「半音阶长文件」切成逐音符 wav（保留余振到下一击）。

文件名如 Marimba.yarn.pp.C2B2.aif：从 C2 半音走到 B2，共 12 音。
切分按能量包络找起音；数量与文件名音域不一致时，取最强的 N 个峰并报警。
已经是单音的文件（音域只有一个 MIDI）原样写出。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    ensure_work, iter_audio, load_config, load_mono, midi_to_note,
    parse_filename, write_pcm16_wav,
)


def _onset_times(y: np.ndarray, sr: int, n_expect: int) -> np.ndarray:
    import librosa

    hop = max(1, sr // 100)  # 10 ms
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # 等待 80ms，避免把一次敲击拆成多个峰
    times = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="time",
        backtrack=True, wait=8, delta=0.08, pre_max=3, post_max=3,
    )
    times = np.asarray(times, dtype=np.float64)
    times = times[(times > 0.02) & (times < len(y) / sr - 0.05)]
    if n_expect <= 1:
        return np.array([0.0]) if times.size == 0 else times[:1]
    if times.size == n_expect:
        return times
    # 数量不对：从 onset_strength 里取 N 个最强局部峰
    peaks = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=3,
                                   post_avg=5, delta=0.05, wait=8)
    if peaks.size == 0:
        # 均匀切（最后手段）
        dur = len(y) / sr
        return np.linspace(0.05, dur * (n_expect - 0.5) / n_expect, n_expect)
    peak_t = peaks * hop / sr
    peak_v = env[peaks]
    if peak_t.size > n_expect:
        idx = np.argsort(peak_v)[-n_expect:]
        peak_t = np.sort(peak_t[idx])
    elif peak_t.size < n_expect:
        print(f"    警告: 只找到 {peak_t.size}/{n_expect} 个起音", flush=True)
    return peak_t.astype(np.float64)


def split_file(path: Path, out_dir: Path, skip_art: set[str], pre_roll: float) -> list[dict]:
    meta = parse_filename(path)
    if meta is None:
        print(f"  跳过（文件名无法解析）: {path.name}", flush=True)
        return []
    if meta["timbre"] in skip_art:
        print(f"  跳过 articulation: {path.name}", flush=True)
        return []
    y, sr = load_mono(path, sr=None)
    midis = meta["midis"]
    n = len(midis)
    rows = []
    if n == 1:
        onsets = np.array([0.0])
        ends = np.array([len(y) / sr])
    else:
        onsets = _onset_times(y, sr, n)
        if onsets.size != n:
            print(f"  警告 {path.name}: 起音 {onsets.size} 个，期望 {n}，按对齐截断/补齐",
                  flush=True)
        if onsets.size < n:
            # 不够就放弃多余 MIDI（从高音丢掉，通常是文件末尾静音）
            midis = midis[: onsets.size]
            n = len(midis)
        elif onsets.size > n:
            onsets = onsets[:n]
        ends = np.append(onsets[1:], len(y) / sr)
    for midi, t0, t1 in zip(midis, onsets, ends):
        a = max(0, int((t0 - pre_roll) * sr))
        b = min(len(y), int(t1 * sr))
        if b - a < sr * 0.12:
            continue
        seg = y[a:b]
        note = midi_to_note(int(midi))
        name = f"{note}.{meta['timbre']}.{meta['dynamic']}.wav"
        dest = out_dir / name
        # 同名（多种来源）加 stem 前缀避免覆盖
        if dest.exists():
            dest = out_dir / f"{path.stem}.{name}"
        write_pcm16_wav(dest, seg, sr)
        rows.append({
            "src": str(path), "wav": str(dest), "midi": int(midi),
            "dynamic": meta["dynamic"], "timbre": meta["timbre"],
            "sr_src": sr, "dur_sec": round((b - a) / sr, 4),
        })
    return rows


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="半音阶长文件 → 逐音符 wav")
    ap.add_argument("--src", default=str(ensure_work("raw", "iowa_marimba")),
                    help="原始目录（Iowa 下载目录或 02 的 inbox）")
    ap.add_argument("--out", default=str(ensure_work("split")))
    ap.add_argument("--pre-roll", type=float, default=0.02)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise SystemExit(f"找不到 {src}")
    out.mkdir(parents=True, exist_ok=True)
    skip = {s.lower() for s in cfg["skip_articulations"]}
    files = iter_audio(src)
    if not files:
        raise SystemExit(f"{src} 下没有音频")

    manifest = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {p.name}", flush=True)
        manifest.extend(split_file(p, out, skip, args.pre_roll))
    man_path = out / "split_manifest.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"切出 {len(manifest)} 条单音 -> {out}")
    print(f"清单 {man_path}")


if __name__ == "__main__":
    main()
