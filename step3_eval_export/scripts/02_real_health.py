#!/usr/bin/env python3
"""对 data/real/ 做无监督健康度（无乐谱）：音符密度、音域、调内占比。

解码走 infer_onnx.transcribe_file（与 CLI / HTTP 同一套 onset 峰检测），
因此需要先导出 ONNX。没有真录就跳过。
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
    REPO_ROOT, as_posix, ensure_work, load_config, load_selected, repo_path, write_json,
)

sys.path.insert(0, str(REPO_ROOT))
from infer_onnx import sci, transcribe_file  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_MAJ_PCS = {0, 2, 4, 5, 7, 9, 11}
_MIN_PCS = {0, 2, 3, 5, 7, 8, 10}


def key_stats(midis: list[int]) -> dict:
    """Krumhansl–Kessler 相关估调 + 调内占比。转调乐曲占比会偏低，属正常。"""
    if not midis:
        return {"key": None, "in_key_pct": None, "corr": None}
    pc = np.bincount(np.array(midis) % 12, minlength=12).astype(float)
    best = (-2.0, "?", set())
    for shift in range(12):
        h = np.roll(pc, -shift)
        for prof, mode, sc in ((_KK_MAJOR, "大调", _MAJ_PCS), (_KK_MINOR, "小调", _MIN_PCS)):
            if float(h.std()) < 1e-9:
                r = 0.0
            else:
                r = float(np.corrcoef(h, prof)[0, 1])
            if r > best[0]:
                best = (r, f"{_NAMES[shift]}{mode}", {(s + shift) % 12 for s in sc})
    ik = sum(1 for m in midis if m % 12 in best[2]) / len(midis)
    return {"key": best[1], "in_key_pct": round(ik * 100, 1), "corr": round(float(best[0]), 3)}


def list_audio(root: Path) -> list[Path]:
    files = []
    if not root.is_dir():
        return files
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="真录无监督健康度")
    ap.add_argument("--real-dir", default=None)
    ap.add_argument("--assets", default=None, help="ONNX + meta.json 目录，默认 export/")
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    real = Path(args.real_dir) if args.real_dir else repo_path(cfg["real_dir"])
    files = list_audio(real)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        print(f"没有真录音频（{real}），跳过健康度。")
        write_json(ensure_work() / "real_health.json", {
            "skipped": True, "reason": "no_real_audio", "real_dir": as_posix(real),
        })
        return

    assets = Path(args.assets) if args.assets else repo_path(cfg["export_dir"])
    onnx = list(assets.glob("*.onnx")) if assets.is_dir() else []
    if not assets.is_dir() or not (assets / "meta.json").is_file() or not onnx:
        raise SystemExit(
            f"健康度需要已导出的 ONNX（{assets}/model.onnx + meta.json）。\n"
            "请先跑 04_export_onnx.py，或 python scripts/run_pipeline.py"
        )

    sel, _ = load_selected()
    thresh = args.thresh
    if thresh is None and sel and sel.get("best_thresh") is not None:
        thresh = float(sel["best_thresh"])

    from infer_onnx import load_session  # noqa: WPS433
    sess, meta, onnx_path = load_session(assets)
    if thresh is None:
        thresh = float(meta["thresh_default"])

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}", flush=True)
        try:
            result = transcribe_file(path, thresh=thresh, sess=sess, meta=meta)
        except Exception as e:  # noqa: BLE001
            rows.append({"path": as_posix(path), "ok": False, "error": str(e)})
            continue
        notes = result["notes"]
        midis = [n["midi"] for n in notes]
        dur = float(result["duration"])
        ks = key_stats(midis)
        rows.append({
            "path": as_posix(path),
            "ok": True,
            "duration_sec": round(dur, 2),
            "n_notes": len(notes),
            "notes_per_sec": round(len(notes) / max(dur, 1e-9), 2),
            "midi_lo": min(midis) if midis else None,
            "midi_hi": max(midis) if midis else None,
            "note_lo": sci(min(midis)) if midis else None,
            "note_hi": sci(max(midis)) if midis else None,
            "mean_velocity": round(float(np.mean([n["velocity"] for n in notes])), 1) if notes else None,
            **ks,
        })

    ok_rows = [r for r in rows if r.get("ok")]
    in_keys = [r["in_key_pct"] for r in ok_rows if r.get("in_key_pct") is not None]
    densities = [r["notes_per_sec"] for r in ok_rows]
    report = {
        "decoder": "infer_onnx.transcribe_file",
        "onnx": as_posix(onnx_path),
        "thresh": thresh,
        "n_files": len(files),
        "n_ok": len(ok_rows),
        "mean_notes_per_sec": round(float(np.mean(densities)), 2) if densities else None,
        "mean_in_key_pct": round(float(np.mean(in_keys)), 1) if in_keys else None,
        "files": rows,
        "note": "无乐谱，调内占比用音级直方图估调。转调乐曲会偏低（马林巴 Until Dawn ≈73.9% 仍算正常）。",
        "marimba_reference": cfg["marimba_reference"]["real_health"],
    }
    out = ensure_work() / "real_health.json"
    write_json(out, report)
    print(json.dumps({
        "n_files": report["n_files"], "n_ok": report["n_ok"],
        "thresh": thresh,
        "mean_notes_per_sec": report["mean_notes_per_sec"],
        "mean_in_key_pct": report["mean_in_key_pct"],
    }, ensure_ascii=False, indent=2))
    print(f"报告 {out}")


if __name__ == "__main__":
    main()
