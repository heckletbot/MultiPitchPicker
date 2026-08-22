#!/usr/bin/env python3
"""合成训练数据：单音采样库 → 带 velocity 标签的多音 8s 片段。

输入：data/singles/index.jsonl（字段见 examples/singles_index.example.jsonl，
每行 {wav, midi, dynamic, timbre}，dynamic ∈ pp/mf/ff，timbre 为音色变体名）。

设计要点（换乐器时按真实语汇改 CHORD_TEMPLATES 等常量）：
  1. 按和弦模板生成（单音/五度/八度/三和弦/七和弦…），不是随机叠音
  2. 角色区分：旋律(顶音,强) / 内声部(中间,弱) / 低音(根,中)
  3. velocity 双通道：选 dynamic 采样层(给音色) + 增益(给响度)
  4. 音色留出：--heldout-timbres 指定的 timbre 只出现在 val/test（泛化关卡）

标签每音含 {midi, onset, offset, velocity, role}，供 onset/frame/velocity 三头监督。

用法：
  python3 synth_clips.py --index data/singles/index.jsonl --out data/synth \
      --train-timbres bright --heldout-timbres dark \
      --n-train 12000 --n-val 400 --n-test 400 --workers 16
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import wave
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
STEP_DIR = _SCRIPTS.parent
REPO_ROOT = STEP_DIR.parent
_WS = REPO_ROOT
if str(STEP_DIR) not in sys.path:
    sys.path.insert(0, str(STEP_DIR))

from config import MIDI_HI, MIDI_LO  # noqa: E402

SR = 16000
CLIP_SEC = 8.0

# 和弦模板（半音音程，相对根音）与权重
CHORD_TEMPLATES = {
    (0,): 0.30,                    # 单音旋律
    (0, 7): 0.10,                  # 五度
    (0, 12): 0.08,                 # 八度
    (0, 4, 7): 0.16,               # 大三
    (0, 3, 7): 0.14,               # 小三
    (0, 5, 7): 0.05,               # sus4
    (0, 4, 7, 11): 0.06,           # 大七
    (0, 3, 7, 10): 0.06,           # 小七
    (0, 4, 7, 12): 0.05,           # 大三+八度
}
# velocity 分配（角色）
VEL_MELODY = (88, 122)   # 顶音=旋律，强
VEL_INNER = (34, 70)     # 内声部，系统性弱 ← 核心
VEL_BASS = (58, 90)      # 低音，中

GAP_LOG_RANGE = (0.12, 0.7)
LONG_GAP_PROB = 0.08
LONG_GAP_RANGE = (0.8, 1.5)
SPREAD_MS = (0.0, 0.028)   # 和弦内错峰（琶音/滚奏）
RESTRIKE_PROB = 0.12
RING_DB = -35.0
RING_CAP = 2.0
AUG_PROB = 0.5

_POOL: dict = {}  # midi -> dynamic -> [(wav, ring_sec)]


def _read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    y = data.astype(np.float32) / 32768.0
    # 干净库为听音做了限增益(pp 本就轻)。训练时统一峰值归一化：
    # 音色(谐波/mallet 差异)保留，响度完全交给 velocity 增益控制。
    return y / (np.abs(y).max() + 1e-9) * 0.9


def _ring_sec(y: np.ndarray) -> float:
    w = 800
    n = len(y) // w
    if n == 0:
        return len(y) / SR
    env = np.abs(y[: n * w]).reshape(n, w).max(axis=1)
    peak = env.max() + 1e-9
    th = peak * (10 ** (RING_DB / 20))
    pk = int(env.argmax())
    for i in np.where(env < th)[0]:
        if i > pk:
            return min(i * w / SR, RING_CAP)
    return min(len(y) / SR, RING_CAP)


def _init_worker(pool_index: dict):
    global _POOL
    _POOL = {}
    for midi_s, dyn_map in pool_index.items():
        midi = int(midi_s)
        _POOL[midi] = {}
        for dyn, files in dyn_map.items():
            _POOL[midi][dyn] = [(_read_wav(_WS / f), _ring_sec(_read_wav(_WS / f)))
                                for f in files]


def _vel_to_dyn(vel: int) -> str:
    if vel < 52:
        return "pp"
    if vel < 88:
        return "mf"
    return "ff"


def _vel_to_gain(vel: int) -> float:
    # 响度由 velocity 控（采样已归一化，无自带动态）；内声部弱音在此拉低
    return float(np.clip((vel / 127.0) ** 1.15, 0.12, 1.0))


def _sample_events(rng: np.random.Generator) -> list[dict]:
    """抽一条 8s 内的和弦/单音序列：模板 → 根音 → 角色 velocity → 组内错峰。"""
    templates = list(CHORD_TEMPLATES)
    weights = np.array(list(CHORD_TEMPLATES.values()))
    weights = weights / weights.sum()
    events = []
    t = float(rng.uniform(0.15, 0.5))
    prev_top = None
    while t < CLIP_SEC - 0.6:
        tmpl = templates[int(rng.choice(len(templates), p=weights))]
        # 根音：让旋律(顶音)落在中高音区
        span = tmpl[-1]
        root = int(rng.integers(MIDI_LO, MIDI_HI - span + 1))
        midis = [root + iv for iv in tmpl]
        midis = [m for m in midis if MIDI_LO <= m <= MIDI_HI]
        if not midis:
            continue
        top = max(midis)
        low = min(midis)
        spread = float(rng.uniform(*SPREAD_MS))
        group = []
        for k, m in enumerate(sorted(midis)):
            if m == top:
                role, vr = "melody", VEL_MELODY
            elif m == low and len(midis) >= 3:
                role, vr = "bass", VEL_BASS
            else:
                role, vr = "inner", VEL_INNER
            vel = int(rng.uniform(*vr))
            # 从低到高微错峰（琶音向上）
            onset = t + spread * k
            group.append({"midi": m, "onset": onset, "velocity": vel, "role": role})
        events.extend(group)
        prev_top = top
        if rng.random() < LONG_GAP_PROB:
            gap = float(rng.uniform(*LONG_GAP_RANGE))
        else:
            lo, hi = GAP_LOG_RANGE
            gap = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        t += gap
    return events


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    import scipy.signal
    if rng.random() < AUG_PROB:
        t60 = rng.uniform(0.15, 0.8)
        n_ir = int(t60 * SR)
        ir = np.exp(-np.arange(n_ir) / (t60 * SR / 3)) * rng.standard_normal(n_ir)
        for _ in range(int(rng.integers(1, 3))):
            d = int(rng.uniform(0.01, 0.05) * SR)
            if d < n_ir:
                ir[d] += rng.uniform(0.3, 0.8)
        ir /= np.sqrt((ir ** 2).sum()) + 1e-9
        wet = rng.uniform(0.1, 0.4)
        x = (1 - wet) * x + wet * scipy.signal.fftconvolve(x, ir, mode="full")[: len(x)]
    if rng.random() < AUG_PROB:
        snr = rng.uniform(30, 50)
        x = x + rng.standard_normal(len(x)) * (np.sqrt((x ** 2).mean()) / 10 ** (snr / 20))
    if rng.random() < AUG_PROB:
        g = rng.uniform(1.2, 2.5)
        x = np.tanh(g * x) / np.tanh(g)
    if rng.random() < AUG_PROB:
        x = x * float(10 ** (rng.uniform(-0.3, 0.3) / 20))
    return x


def _render_job(job: dict) -> dict:
    rng = np.random.default_rng(job["seed"])
    timbre = job["timbre"]  # 该 clip 固定音色变体
    n = int(CLIP_SEC * SR)
    mix = np.zeros(n, dtype=np.float32)
    notes = []
    for ev in _sample_events(rng):
        midi = ev["midi"]
        dyn = _vel_to_dyn(ev["velocity"])
        cands = _POOL.get(midi, {}).get(dyn)
        if not cands:  # 该层无样本则回退任意已有层
            for d2 in _POOL.get(midi, {}):
                cands = _POOL[midi][d2]
                if cands:
                    break
        if not cands:
            continue
        y, ring = cands[int(rng.integers(len(cands)))]
        s = int(ev["onset"] * SR)
        if s >= n:
            continue
        seg = y[: n - s]
        mix[s: s + len(seg)] += seg * _vel_to_gain(ev["velocity"])
        notes.append({
            "midi": midi, "onset": round(ev["onset"], 4),
            "offset": round(min(ev["onset"] + ring, CLIP_SEC), 4),
            "velocity": ev["velocity"], "role": ev["role"],
        })
    mix = _augment(mix, rng)
    peak = np.abs(mix).max() + 1e-9
    mix = mix / peak * 0.95
    p = Path(job["wav_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())
    notes.sort(key=lambda x: (x["onset"], x["midi"]))
    return {"id": job["id"], "audio": job["wav_path"], "timbre": timbre,
            "duration_sec": CLIP_SEC, "notes": notes}


def _build_pool_index(index_path: Path, timbres: list[str]) -> dict:
    rows = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
    idx: dict = {}
    for r in rows:
        if timbres and r.get("timbre", "default") not in timbres:
            continue
        # 可选质检字段：索引若带 winner_midi（听审出的实际音级），只用与标称一致的行
        if "winner_midi" in r and r["winner_midi"] != r["midi"]:
            continue
        idx.setdefault(str(r["midi"]), {}).setdefault(r["dynamic"], []).append(r["wav"])
    return idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(_WS / "data/singles/index.jsonl"),
                    help="单音索引 jsonl，schema 见 examples/")
    ap.add_argument("--out", default=str(_WS / "data/synth"))
    ap.add_argument("--n-train", type=int, default=12000)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-timbres", default="",
                    help="逗号分隔，训练用音色变体；留空 = 全部")
    ap.add_argument("--heldout-timbres", default="",
                    help="逗号分隔，只给 val/test 的留出音色；留空 = 与训练同池（无留出，泛化评估会偏乐观）")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out).resolve()
    index_path = Path(args.index).resolve()
    rng = np.random.default_rng(args.seed)

    train_timbres = [s for s in args.train_timbres.split(",") if s]
    heldout_timbres = [s for s in args.heldout_timbres.split(",") if s]
    pool_train = _build_pool_index(index_path, train_timbres)
    pool_held = _build_pool_index(index_path, heldout_timbres) if heldout_timbres else pool_train
    if not heldout_timbres:
        print("警告：未指定 --heldout-timbres，val/test 与训练同音色池，泛化评估会偏乐观", flush=True)
    if not pool_train:
        raise SystemExit(f"索引 {index_path} 里没有可用单音（检查 timbre 过滤与 winner_midi）")

    jobs = {"train": [], "validation": [], "test": []}
    for split, n_clip in [("train", args.n_train), ("validation", args.n_val), ("test", args.n_test)]:
        tset = (train_timbres if split == "train" else heldout_timbres or train_timbres) or ["default"]
        for i in range(n_clip):
            cid = f"clip_{split}_{i:06d}"
            jobs[split].append({
                "id": cid, "seed": int(rng.integers(0, 2 ** 31)),
                "timbre": tset[int(rng.integers(len(tset)))],
                "wav_path": str(out / "wav" / split / f"{cid}.wav"),
            })

    for split in ["train", "validation", "test"]:
        pool = pool_train if split == "train" else pool_held
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(pool,)) as pool_p:
            rows = []
            for i, row in enumerate(pool_p.imap_unordered(_render_job, jobs[split], chunksize=8), 1):
                rows.append(row)
                if i % 2000 == 0 or i == len(jobs[split]):
                    print(f"  {split} [{i}/{len(jobs[split])}] {time.time()-t0:.0f}s", flush=True)
        rows.sort(key=lambda r: r["id"])
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} -> {out / f'{split}.jsonl'}", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "sr": SR, "clip_sec": CLIP_SEC, "chord_templates": {str(k): v for k, v in CHORD_TEMPLATES.items()},
        "vel_melody": VEL_MELODY, "vel_inner": VEL_INNER, "vel_bass": VEL_BASS,
        "train_timbres": train_timbres, "heldout_timbres": heldout_timbres,
        "seed": args.seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
