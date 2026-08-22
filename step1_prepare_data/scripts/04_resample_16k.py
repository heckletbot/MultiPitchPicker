#!/usr/bin/env python3
"""重采样为仓库训练所需格式：16 kHz、单声道、16-bit PCM wav。

synth_clips.py 用 wave 模块直接读，不重采样，所以单音库必须已经是 16 kHz。
不做响度归一化：pp/mf/ff 的自然响度差留给听审；合成阶段会再做峰值归一化。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    ensure_work, iter_audio, load_config, load_mono, parse_filename,
    write_jsonl, write_pcm16_wav,
)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="重采样到 16 kHz mono PCM16")
    ap.add_argument("--src", default=str(ensure_work("split")))
    ap.add_argument("--out", default=str(ensure_work("singles")))
    ap.add_argument("--sr", type=int, default=int(cfg["target_sr"]))
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    man_src = src / "split_manifest.jsonl"
    rows_in = []
    if man_src.is_file():
        for line in man_src.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows_in.append(json.loads(line))
    else:
        for p in iter_audio(src):
            meta = parse_filename(p)
            if meta is None or len(meta["midis"]) != 1:
                print(f"  跳过 {p.name}（需要单音或 split_manifest）")
                continue
            rows_in.append({
                "wav": str(p), "midi": meta["midis"][0],
                "dynamic": meta["dynamic"], "timbre": meta["timbre"],
            })
    if not rows_in:
        raise SystemExit(f"{src} 没有可重采样的单音")

    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for i, r in enumerate(rows_in, 1):
        p = Path(r["wav"])
        y, _ = load_mono(p, sr=args.sr)
        dest = out / p.with_suffix(".wav").name
        write_pcm16_wav(dest, y, args.sr)
        rows_out.append({
            "wav": str(dest), "midi": int(r["midi"]),
            "dynamic": r["dynamic"], "timbre": r["timbre"],
            "sr": args.sr, "n_samples": int(y.size),
        })
        if i % 50 == 0 or i == len(rows_in):
            print(f"  {i}/{len(rows_in)}", flush=True)
    write_jsonl(out / "resample_manifest.jsonl", rows_out)
    print(f"{len(rows_out)} 条 -> {out}")


if __name__ == "__main__":
    main()
