#!/usr/bin/env python3
"""log-CQT 特征（与训练 DataLoader 同一套常数），本步只用于预计算缓存。"""
from __future__ import annotations

import os

import numpy as np

SR = 16000
CLIP_SEC = 8.0
CQT_HOP = 256
CQT_FMIN = 32.7032
CQT_BINS = 285
CQT_BPO = 36

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
