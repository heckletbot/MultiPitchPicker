#!/usr/bin/env python3
"""基座模型自训练伪标：整曲滑窗推理 -> τ 提取音符(带力度) -> 切 8s 片段。

可选的 <real-dir>/pool_filters.json 提供听审过滤（exclude / keep_until_sec /
trim_end_sec，见 examples/pool_filters.example.json）；没有该文件则全部整曲使用。
输出：
  <out>/wav/<song>_<idx>.wav        8s 16k mono 片段
  <out>/train.jsonl                 {"id", "audio", "notes":[{midi,onset,offset,velocity}]}

用法：
  python3 pseudo_label.py --ckpt data/runs/base/best.pt \
      --real-dir data/real --out data/pseudo --thresh 0.3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config import MIDI_LO, NUM_PITCHES  # noqa: E402
from dataset import CLIP_SEC, CQT_FPS, SR, cqt_from_audio  # noqa: E402
from model import NoteCNN  # noqa: E402
from train import decode_from_peaks, find_peaks  # noqa: E402

CLIP_N = int(CLIP_SEC * SR)


def transcribe(y: np.ndarray, model, device):
    """整曲滑窗（8s，hop 4s）+ Hann 中心加权，拼成与训练同分布的概率图。

    图内/训练 CQT 都是「按当前片段标准化」，所以不能一次喂整曲。
    """
    import torch

    hop_n = CLIP_N // 2
    n_fr = int(np.ceil(len(y) / SR * CQT_FPS)) + 2
    on_acc = np.zeros((NUM_PITCHES, n_fr), dtype=np.float32)
    vel_acc = np.zeros((NUM_PITCHES, n_fr), dtype=np.float32)
    wgt = np.zeros(n_fr, dtype=np.float32)
    for s in range(0, max(1, len(y) - hop_n), hop_n):
        seg = y[s: s + CLIP_N]
        if len(seg) < CLIP_N:
            seg = np.pad(seg, (0, CLIP_N - len(seg)))
        x = cqt_from_audio(seg)
        with torch.no_grad():
            xt = torch.from_numpy(x[None, None]).to(device)
            on_l, _fr, vel_l = model(xt)
            on_p = torch.sigmoid(on_l.float())[0].cpu().numpy()
            vel_p = torch.sigmoid(vel_l.float())[0].cpu().numpy()
        f0 = int(round(s / SR * CQT_FPS))
        n_t = on_p.shape[1]
        w = np.hanning(n_t) + 1e-3
        e = min(n_fr, f0 + n_t)
        on_acc[:, f0:e] += on_p[:, : e - f0] * w[: e - f0]
        vel_acc[:, f0:e] += vel_p[:, : e - f0] * w[: e - f0]
        wgt[f0:e] += w[: e - f0]
    return on_acc / np.maximum(wgt, 1e-9), vel_acc / np.maximum(wgt, 1e-9)


def main() -> None:
    import librosa
    import soundfile as sf
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/runs/base/best.pt")
    ap.add_argument("--real-dir", default="data/real")
    ap.add_argument("--out", default="data/pseudo")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min-notes", type=int, default=4, help="片段最少音符数，低于则丢弃")
    args = ap.parse_args()

    ws = _DIR
    real_dir = ws / args.real_dir if not Path(args.real_dir).is_absolute() else Path(args.real_dir)
    out = ws / args.out if not Path(args.out).is_absolute() else Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)

    filters_path = real_dir / "pool_filters.json"
    filters = json.loads(filters_path.read_text()) if filters_path.is_file() else {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ws / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt,
                    map_location=device)
    model = NoteCNN(lookahead=bool(ck.get("lookahead", False))).to(device).eval()
    model.load_state_dict(ck["model"])

    songs = sorted(p for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a")
                   for p in real_dir.glob(ext))
    rows, n_skip = [], 0
    for si, song in enumerate(songs):
        rule = filters.get(song.name, {})
        if rule.get("exclude"):
            n_skip += 1
            continue
        y, _ = librosa.load(song, sr=SR, mono=True)
        if "keep_until_sec" in rule:
            y = y[: int(rule["keep_until_sec"] * SR)]
        if "trim_end_sec" in rule:
            y = y[: max(0, len(y) - int(rule["trim_end_sec"] * SR))]
        if len(y) < CLIP_N:
            continue
        on_p, vel_p = transcribe(y, model, device)
        notes = decode_from_peaks(find_peaks(on_p), args.thresh, CQT_FPS)  # [(pitch, t, prob)]
        # 每音符力度取 onset 帧的 vel 头输出；offset = 下一次同音高起音或 +0.6s
        by_pitch: dict[int, list[float]] = {}
        for p, t, _v in notes:
            by_pitch.setdefault(p, []).append(t)
        events = []
        for p, t, _v in notes:
            fr = min(int(round(t * CQT_FPS)), vel_p.shape[1] - 1)
            vel = int(round(float(vel_p[p, fr]) * 127))
            nxt = [x for x in by_pitch[p] if x > t]
            off = min(t + 0.6, min(nxt) if nxt else t + 0.6)
            events.append({"midi": p + MIDI_LO, "onset": t, "offset": off,
                           "velocity": max(20, min(127, vel))})
        events.sort(key=lambda e: e["onset"])

        stem = f"real{si:03d}"
        n_clips = int(len(y) // CLIP_N)
        for ci in range(n_clips):
            t0, t1 = ci * CLIP_SEC, (ci + 1) * CLIP_SEC
            ev = [{**e, "onset": round(e["onset"] - t0, 4),
                   "offset": round(min(e["offset"], t1) - t0, 4)}
                  for e in events if t0 <= e["onset"] < t1]
            if len(ev) < args.min_notes:
                continue
            wav_path = out / "wav" / f"{stem}_{ci:03d}.wav"
            sf.write(wav_path, y[int(t0 * SR): int(t1 * SR)], SR, subtype="PCM_16")
            rows.append({"id": f"{stem}_{ci:03d}",
                         "audio": str(wav_path.relative_to(ws)),
                         "notes": ev})
        print(f"[{si+1}/{len(songs)}] {song.name[:40]}  音符 {len(events)}", flush=True)

    with (out / "train.jsonl").open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_notes = sum(len(r["notes"]) for r in rows)
    print(f"\n排除 {n_skip} 首 | 片段 {len(rows)} 条 | 伪标音符 {n_notes} "
          f"(均 {n_notes/max(1,len(rows)):.1f}/条) -> {out}/train.jsonl", flush=True)


if __name__ == "__main__":
    main()
