#!/usr/bin/env python3
"""命令行转录：调用 infer_onnx.transcribe_file，写出 {time, midi, note, velocity}。

  python step4_api/scripts/transcribe_cli.py --audio song.wav
  python step4_api/scripts/transcribe_cli.py --audio song.mp3 --thresh 0.8 --out notes.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import REPO_ROOT, default_assets, load_selected_thresh  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from infer_onnx import sci, transcribe_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="ONNX 整曲转录 CLI")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--assets", default=None)
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--out", default=None, help="写出 notes JSON 数组")
    args = ap.parse_args()

    assets = Path(args.assets) if args.assets else default_assets()
    thresh = args.thresh if args.thresh is not None else load_selected_thresh()
    result = transcribe_file(args.audio, thresh=thresh, assets=assets)
    notes = result["notes"]
    midis = [n["midi"] for n in notes]
    dur, th = result["duration"], result["thresh"]
    print(f"时长 {dur:.1f}s | 阈值 {th} | 音符 {len(notes)} ({len(notes) / max(dur, 1e-9):.2f}/s)")
    if midis:
        print(f"音域 {sci(min(midis))} ~ {sci(max(midis))}")
    payload = {
        "duration": result["duration"],
        "thresh": result["thresh"],
        "n_notes": result["n_notes"],
        "notes": notes,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"-> {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
