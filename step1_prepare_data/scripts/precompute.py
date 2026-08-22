#!/usr/bin/env python3
"""并行预计算 CQT 缓存（<wav>.cqt285.npy）。

jsonl 里的 audio 路径相对仓库根目录。
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO_ROOT = _SCRIPTS.parent.parent


def _job(wav_str: str) -> int:
    import numpy as np

    from cqt_feature import CACHE_SUFFIX, CLIP_SEC, SR, cqt_from_audio

    wav = Path(wav_str)
    cache = wav.with_suffix(wav.suffix + CACHE_SUFFIX)
    if cache.is_file():
        return 0
    import librosa

    y, _ = librosa.load(wav, sr=SR, mono=True)
    target = int(CLIP_SEC * SR)
    y = np.pad(y, (0, max(0, target - len(y))))[:target]
    x = cqt_from_audio(y)
    tmp = cache.with_suffix(f".tmp{mp.current_process().pid}.npy")
    np.save(tmp, x.astype(np.float16))
    tmp.replace(cache)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    wavs = []
    for j in args.jsonl:
        for line in Path(j).read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = Path(json.loads(line)["audio"])
                wavs.append(str(p if p.is_absolute() else REPO_ROOT / p))
    print(f"共 {len(wavs)} 条", flush=True)
    t0 = time.time()
    with mp.Pool(args.workers) as pool:
        done = 0
        for i, _ in enumerate(pool.imap_unordered(_job, wavs, chunksize=16), 1):
            done += 1
            if done % 2000 == 0 or done == len(wavs):
                print(f"  {done}/{len(wavs)} {time.time()-t0:.0f}s", flush=True)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
