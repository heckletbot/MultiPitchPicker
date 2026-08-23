#!/usr/bin/env python3
"""OAF 解码与 note-level 评测。

decode_notes: onset 概率 [65,T] -> [(pitch_idx, onset_sec, prob)]
note_f1:      预测 vs 真值，onset ±tol 内同音高即匹配（贪心一对一）。
"""
from __future__ import annotations

import numpy as np

FPS = 100.0


def decode_notes(onset_prob: np.ndarray, thresh: float = 0.5,
                 min_gap_frames: int = 5) -> list[tuple[int, float, float]]:
    """局部极大 + 阈值；同音高相邻峰间隔 < min_gap_frames 只保留最高。"""
    n_p, n_t = onset_prob.shape
    out = []
    for p in range(n_p):
        row = onset_prob[p]
        cand = []
        for t in range(n_t):
            v = row[t]
            if v < thresh:
                continue
            lo, hi = max(0, t - 2), min(n_t, t + 3)
            if v >= row[lo:hi].max() - 1e-9:
                cand.append((t, v))
        # 邻近峰去重
        kept: list[tuple[int, float]] = []
        for t, v in cand:
            if kept and t - kept[-1][0] < min_gap_frames:
                if v > kept[-1][1]:
                    kept[-1] = (t, v)
            else:
                kept.append((t, v))
        out.extend((p, t / FPS, float(v)) for t, v in kept)
    out.sort(key=lambda x: x[1])
    return out


def note_f1(pred: list[tuple[int, float, float]],
            gt: list[tuple[int, float]], tol: float = 0.05) -> dict:
    matched_gt = [False] * len(gt)
    tp = 0
    for p, t, _ in pred:
        best, best_d = -1, tol + 1e-9
        for j, (gp, gt_t) in enumerate(gt):
            if matched_gt[j] or gp != p:
                continue
            d = abs(gt_t - t)
            if d <= tol and d < best_d:
                best, best_d = j, d
        if best >= 0:
            matched_gt[best] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}
