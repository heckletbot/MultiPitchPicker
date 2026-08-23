#!/usr/bin/env python3
"""CQT 数据集（三头转录模型用）：8s 序列 -> log-CQT + onset/frame/velocity roll。

特征：CQT 285 bins（fmin=C1 32.7Hz，36 bins/八度，hop 256 → 62.5 fps），
对数幅度 + 每片段标准化。谐波堆叠在模型内做（固定 bin 位移，利于 ONNX 整图导出）。

特征缓存：首次计算后存 <wav>.cqt285.npy（float16，~285KB/条），之后直接读。

低延迟变体：设环境变量 MP_CQT_GAMMA>0（推荐 2）时改用 VQT——
带宽 = α·f + γ Hz，低音滤波器显著截短（C2: 0.79s→0.31s），缓存后缀随 γ 区分。
详见 README「低延迟变体」一节。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config import MIDI_LO, NUM_PITCHES  # noqa: E402

SR = 16000
CLIP_SEC = 8.0
CQT_HOP = 256                 # 62.5 fps（8 个八度的 CQT 要求 hop 为 128 的倍数）
CQT_FPS = SR / CQT_HOP
CQT_FMIN = 32.7032            # C1：C2 的次谐波也在图内
CQT_BINS = 285                # 36 bins/八度，最高 ~7.6kHz < Nyquist
CQT_BPO = 36

# γ>0 → VQT（带宽 α·f+γ）：低音分析滤波器截短，换取低延迟；γ=0 为标准 CQT。
# 缓存后缀随 γ 区分，两种特征可并存互不覆盖。
CQT_GAMMA = float(os.environ.get("MP_CQT_GAMMA", "0") or 0)
CACHE_SUFFIX = ".cqt285.npy" if CQT_GAMMA <= 0 else f".vqt285g{CQT_GAMMA:g}.npy"


def cqt_from_audio(y: np.ndarray) -> np.ndarray:
    """log-CQT/VQT (285, T) float32，每片段标准化。"""
    import librosa

    if CQT_GAMMA > 0:
        C = np.abs(librosa.vqt(y=y, sr=SR, hop_length=CQT_HOP, fmin=CQT_FMIN,
                               n_bins=CQT_BINS, bins_per_octave=CQT_BPO,
                               gamma=CQT_GAMMA))
    else:
        C = np.abs(librosa.cqt(y=y, sr=SR, hop_length=CQT_HOP, fmin=CQT_FMIN,
                               n_bins=CQT_BINS, bins_per_octave=CQT_BPO))
    x = np.log(C + 1e-6).astype(np.float32)
    return (x - x.mean()) / (x.std() + 1e-5)


def rasterize(notes, n_frames: int):
    """音符事件 → 三张 roll，与 hop=256 的帧对齐。

    onset：起音帧扩成 [f0-1, f0+2)，减轻 ±1 帧标注误差。
    frame：从 onset 到 offset（含）置 1。
    vel：只写在 onset 带上（与 loss mask 一致）。
    """
    onset = np.zeros((NUM_PITCHES, n_frames), dtype=np.float32)
    frame = np.zeros((NUM_PITCHES, n_frames), dtype=np.float32)
    vel = np.zeros((NUM_PITCHES, n_frames), dtype=np.float32)
    for nt in notes:
        p = nt["midi"] - MIDI_LO
        if not (0 <= p < NUM_PITCHES):
            continue
        f0 = int(round(nt["onset"] * CQT_FPS))
        f1 = int(round(nt["offset"] * CQT_FPS))
        a, b = max(0, f0 - 1), min(n_frames, f0 + 2)
        onset[p, a:b] = 1.0
        vel[p, a:b] = nt.get("velocity", 80) / 127.0
        frame[p, max(0, f0): max(0, min(n_frames, f1 + 1))] = 1.0
    return onset, frame, vel


class CqtOafDataset(Dataset):
    def __init__(self, jsonl_path):
        self.rows = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines() if l.strip()]
        self.ws = Path(__file__).resolve().parent

    def __len__(self):
        return len(self.rows)

    def _audio(self, row):
        p = Path(row["audio"])
        return p if p.is_absolute() else self.ws / p

    def _feat(self, row) -> np.ndarray:
        wav = self._audio(row)
        cache = wav.with_suffix(wav.suffix + CACHE_SUFFIX)
        if cache.is_file():
            return np.load(cache).astype(np.float32)
        import librosa

        y, _ = librosa.load(wav, sr=SR, mono=True)
        target = int(CLIP_SEC * SR)
        y = np.pad(y, (0, max(0, target - len(y))))[:target]
        x = cqt_from_audio(y)
        tmp = cache.with_suffix(".tmp.npy")
        np.save(tmp, x.astype(np.float16))
        tmp.replace(cache)
        return x

    def __getitem__(self, i):
        row = self.rows[i]
        x = self._feat(row)[None]  # (1, 285, T)
        onset, frame, vel = rasterize(row["notes"], x.shape[-1])
        return (torch.from_numpy(x.copy()), torch.from_numpy(onset),
                torch.from_numpy(frame), torch.from_numpy(vel), row["id"])
