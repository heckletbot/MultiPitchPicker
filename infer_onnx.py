#!/usr/bin/env python3
"""端侧参考实现：仅用 numpy + onnxruntime 做整曲转录（8s 滑窗 hanning 拼接 + 峰值解码）。

这份解码逻辑可作为端侧对拍基准。

用法：
  python3 infer_onnx.py --audio song.mp3 \
      [--assets export] [--thresh 0.8] [--out events.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SR = 16000
FPS = 62.5
WIN_S, HOP_S = 8.0, 4.0
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def sci(m: int) -> str:
    return f"{NAMES[m % 12]}{m // 12 - 1}"


def decode_onset_maps(on_p: np.ndarray, vel_p: np.ndarray, meta: dict, thresh: float,
                      t0: float = 0.0) -> list[dict]:
    """onset/vel 概率图 → events。on_p/vel_p 形状 (音高数, T)。time 为 t0 + 帧/FPS。"""
    floor, gap = meta["peak_floor"], meta["min_gap_frames"]
    n_p = on_p.shape[0]
    events = []
    for p in range(n_p):
        row = on_p[p]
        kept: list[tuple[int, float]] = []
        for t in range(2, len(row) - 2):
            v = row[t]
            if v < max(floor, thresh) or v < row[t - 2: t + 3].max() - 1e-9:
                continue
            if kept and t - kept[-1][0] < gap:
                if v > kept[-1][1]:
                    kept[-1] = (t, v)
            else:
                kept.append((t, v))
        for t, v in kept:
            midi = p + meta["midi_lo"]
            events.append({
                "time": round(t0 + t / FPS, 3),
                "midis": [midi], "notes": [sci(midi)],
                "probs": [round(float(v), 4)],
                "velocity": int(round(float(vel_p[p, t]) * 127)),
            })
    events.sort(key=lambda e: e["time"])
    return events


def transcribe_chunk(y: np.ndarray, sess, meta: dict, thresh: float | None = None,
                     t0: float = 0.0) -> dict:
    """一段 PCM 直接送 ONNX（不补零到 8 秒）。用于 2 秒退化窗。

    y: 1-D float32，16 kHz 单声道。
    返回 duration / thresh / n_notes / notes（字段与 transcribe_file 相同）。
    duration 为本段秒数；notes[].time 相对本段起点，再加上 t0。
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    th = float(thresh) if thresh is not None else float(meta["thresh_default"])
    on, vel = sess.run(None, {"audio": y[None]})
    events = decode_onset_maps(on[0], vel[0], meta, th, t0=t0)
    notes = flatten_note_events(events)
    return {
        "duration": round(len(y) / SR, 3),
        "thresh": th,
        "n_notes": len(notes),
        "notes": notes,
        "events": events,
    }


def transcribe(y: np.ndarray, sess, meta: dict, thresh: float):
    """与训练伪标同一套：8s/4s 滑窗、Hann 拼接、局部极大解码。"""
    win_n, hop_n = int(WIN_S * SR), int(HOP_S * SR)
    n_fr = int(np.ceil(len(y) / SR * FPS)) + 2
    n_p = meta["num_pitches"]
    on_acc = np.zeros((n_p, n_fr), dtype=np.float32)
    vel_acc = np.zeros((n_p, n_fr), dtype=np.float32)
    wgt = np.zeros(n_fr, dtype=np.float32)
    for s in range(0, max(1, len(y) - hop_n), hop_n):
        seg = y[s: s + win_n]
        if len(seg) < win_n:
            seg = np.pad(seg, (0, win_n - len(seg)))
        on, vel = sess.run(None, {"audio": seg[None].astype(np.float32)})
        on, vel = on[0], vel[0]
        f0 = int(round(s / SR * FPS))
        t = on.shape[1]
        w = np.hanning(t) + 1e-3
        e = min(n_fr, f0 + t)
        on_acc[:, f0:e] += on[:, : e - f0] * w[: e - f0]
        vel_acc[:, f0:e] += vel[:, : e - f0] * w[: e - f0]
        wgt[f0:e] += w[: e - f0]
    on_p = on_acc / np.maximum(wgt, 1e-9)
    vel_p = vel_acc / np.maximum(wgt, 1e-9)
    return decode_onset_maps(on_p, vel_p, meta, thresh)


def flatten_note_events(events: list[dict]) -> list[dict]:
    """API / 客户端简表：{time, midi, note, velocity}。解码仍用 transcribe()。"""
    out = []
    for e in events:
        midi = int(e["midis"][0])
        out.append({
            "time": e["time"],
            "midi": midi,
            "note": e["notes"][0],
            "velocity": int(e["velocity"]),
        })
    return out


def load_session(assets: str | Path | None = None):
    """加载 export/ 下的 ONNX 与 meta.json。返回 (sess, meta, onnx_path)。"""
    import onnxruntime as ort

    root = Path(__file__).resolve().parent
    assets = Path(assets) if assets else root / "export"
    meta_path = assets / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"找不到 {meta_path}，请先跑 step3_eval_export")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    onnx_files = sorted(assets.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"{assets} 下没有 .onnx 模型")
    sess = ort.InferenceSession(str(onnx_files[0]), providers=["CPUExecutionProvider"])
    return sess, meta, onnx_files[0]


def transcribe_file(
    path: str | Path,
    thresh: float | None = None,
    assets: str | Path | None = None,
    sess=None,
    meta: dict | None = None,
) -> dict:
    """整曲转录（解码与 CLI / HTTP 相同）。

    返回 duration / thresh / n_notes、
    notes（{time, midi, note, velocity}）、events（原 jsonl 字段）。
    传入已加载的 sess/meta 可避免每次重建会话（HTTP API 用）。
    """
    import librosa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到音频: {path}")
    if sess is None or meta is None:
        sess, meta, _ = load_session(assets)
    th = float(thresh) if thresh is not None else float(meta["thresh_default"])
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    events = transcribe(y.astype(np.float32), sess, meta, th)
    notes = flatten_note_events(events)
    dur = float(len(y) / SR)
    return {
        "audio": str(path).replace("\\", "/"),
        "duration": round(dur, 3),
        "thresh": th,
        "n_notes": len(notes),
        "notes": notes,
        "events": events,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--assets", default=str(Path(__file__).parent / "export"))
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--out", default=None, help="写出原格式 jsonl（midis/notes/probs）")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="写出 notes 数组 JSON：[{time,midi,note,velocity}, ...]")
    args = ap.parse_args()

    result = transcribe_file(args.audio, thresh=args.thresh, assets=args.assets)
    events = result["events"]
    midis = [e["midi"] for e in result["notes"]]
    dur, th = result["duration"], result["thresh"]
    print(f"时长 {dur:.1f}s | 阈值 {th} | 音符 {len(events)} ({len(events) / max(dur, 1e-9):.2f}/s)")
    if midis:
        print(f"音域 {sci(min(midis))} ~ {sci(max(midis))}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as w:
            for e in events:
                w.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"-> {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(result["notes"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {args.json_out}")


if __name__ == "__main__":
    main()
