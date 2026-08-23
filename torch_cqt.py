#!/usr/bin/env python3
"""图内 CQT：按八度分组的 conv1d 直接卷积，可整图导出 ONNX（端侧无 librosa）。

与训练用 librosa.cqt 参数对齐：sr16k / hop256 / fmin C1 / 285 bins / 36 bpo。
核为 hann 加窗复指数，l1 归一 + sqrt(N) 缩放（对应 librosa norm=1, scale=True）。
残余的每 bin 幅度差在 log 域是常数偏置，用 calibrate() 拟合后存为图内 bias。
MP_CQT_GAMMA>0 时核长按 VQT 规则截短（与训练特征一致），bias 需重新校准。

用法（验证对齐 / 校准并保存 bias）：
  python3 torch_cqt.py --check data/synth/validation.jsonl --n 16
  python3 torch_cqt.py --check ... --save-bias data/runs/torch_cqt_bias.pt
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from dataset import CQT_BINS, CQT_BPO, CQT_FMIN, CQT_GAMMA, CQT_HOP, SR  # noqa: E402

ALPHA = 2.0 ** (1.0 / CQT_BPO) - 1.0
Q = 1.0 / ALPHA


def _bin_len(f: float) -> int:
    """bin 的滤波器采样长度；γ>0 时为 VQT（带宽 α·f+γ，低音截短），与 librosa.vqt 对齐。"""
    return int(np.ceil(Q * SR / (f + CQT_GAMMA / ALPHA)))


def _kernels():
    """每 bin 复核 -> 按八度分组 [(f_lo_bin, 核数组 (n,2,L), pad)]，低八度核长高八度核短。"""
    freqs = CQT_FMIN * (2.0 ** (np.arange(CQT_BINS) / CQT_BPO))
    groups = []
    for o in range(int(np.ceil(CQT_BINS / CQT_BPO))):
        lo, hi = o * CQT_BPO, min((o + 1) * CQT_BPO, CQT_BINS)
        fs = freqs[lo:hi]
        L = _bin_len(fs[0])
        L += L % 2  # 偶数长度方便 pad
        ker = np.zeros((len(fs), 2, L), dtype=np.float32)
        t = (np.arange(L) - L / 2) / SR
        for i, f in enumerate(fs):
            n = _bin_len(f)
            win = np.hanning(n)
            win /= win.sum()                       # l1 归一
            s = (L - n) // 2
            ph = 2 * np.pi * f * t[s: s + n]
            ker[i, 0, s: s + n] = (win * np.cos(ph)) * np.sqrt(n)
            ker[i, 1, s: s + n] = (win * np.sin(ph)) * np.sqrt(n)
        groups.append((lo, ker, L // 2))
    return groups


class TorchCQT(nn.Module):
    """(B, n_samples) -> log|CQT| (B, 285, T) + 每片段标准化。T = n_samples/hop + 1。"""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList()
        self.slices = []
        for lo, ker, pad in _kernels():
            n, _, L = ker.shape
            c = nn.Conv1d(1, n * 2, L, stride=CQT_HOP, padding=pad, bias=False)
            c.weight.data = torch.from_numpy(ker.reshape(n * 2, 1, L))
            c.weight.requires_grad_(False)
            self.convs.append(c)
            self.slices.append((lo, n))
        # 与 librosa 的每 bin log 域偏置校准项（calibrate() 填充）
        self.register_buffer("log_bias", torch.zeros(CQT_BINS))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        x = y[:, None, :]  # (B,1,N)
        outs = []
        t_min = None
        for c in self.convs:
            o = c(x)  # (B, 2n, T)
            B, C2, T = o.shape
            o = o.reshape(B, C2 // 2, 2, T)
            mag = torch.sqrt(o[:, :, 0] ** 2 + o[:, :, 1] ** 2 + 1e-12)
            outs.append(mag)
            t_min = T if t_min is None else min(t_min, T)
        cqt = torch.cat([o[:, :, :t_min] for o in outs], dim=1)  # (B,285,T)
        x = torch.log(cqt + 1e-6) + self.log_bias[None, :, None]
        mu = x.mean(dim=(1, 2), keepdim=True)
        sd = x.std(dim=(1, 2), keepdim=True)
        return (x - mu) / (sd + 1e-5)


@torch.no_grad()
def calibrate(module: TorchCQT, wavs: list[np.ndarray], device="cpu") -> None:
    """用若干音频拟合每 bin 的 log 域偏置，使 torch 特征均值对齐 librosa。"""
    import librosa

    diffs = []
    for y in wavs:
        if CQT_GAMMA > 0:
            C = np.abs(librosa.vqt(y=y, sr=SR, hop_length=CQT_HOP, fmin=CQT_FMIN,
                                   n_bins=CQT_BINS, bins_per_octave=CQT_BPO,
                                   gamma=CQT_GAMMA))
        else:
            C = np.abs(librosa.cqt(y=y, sr=SR, hop_length=CQT_HOP, fmin=CQT_FMIN,
                                   n_bins=CQT_BINS, bins_per_octave=CQT_BPO))
        ref = np.log(C + 1e-6)
        yt = torch.from_numpy(y[None]).to(device)
        x = yt[:, None, :]
        outs, t_min = [], None
        for c in module.convs:
            o = c(x)
            B, C2, T = o.shape
            o = o.reshape(B, C2 // 2, 2, T)
            outs.append(torch.sqrt(o[:, :, 0] ** 2 + o[:, :, 1] ** 2 + 1e-12))
            t_min = T if t_min is None else min(t_min, T)
        raw = torch.cat([o[:, :, :t_min] for o in outs], dim=1)[0].cpu().numpy()
        t = min(ref.shape[1], raw.shape[1])
        diffs.append(ref[:, :t].mean(axis=1) - np.log(raw[:, :t] + 1e-6).mean(axis=1))
    module.log_bias.copy_(torch.from_numpy(np.mean(diffs, axis=0).astype(np.float32)))


def main() -> None:
    import argparse
    import json

    import librosa

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="jsonl，取前 n 条音频做对齐验证")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--save-bias", default=None,
                    help="校准后把 log_bias 存到此路径（供 export_onnx --bias 用）")
    args = ap.parse_args()

    ws = _DIR
    rows = [json.loads(l) for l in open(args.check, encoding="utf-8") if l.strip()][: args.n]
    wavs = []
    for r in rows:
        p = Path(r["audio"])
        y, _ = librosa.load(p if p.is_absolute() else ws / p, sr=SR, mono=True)
        wavs.append(y[: SR * 8].astype(np.float32))

    m = TorchCQT().eval()
    calibrate(m, wavs[: args.n // 2])

    # 用另一半验证（标准化后特征的逐点误差）
    from dataset import cqt_from_audio

    errs = []
    for y in wavs[args.n // 2:]:
        ref = cqt_from_audio(y)                    # librosa 版（训练特征）
        with torch.no_grad():
            got = m(torch.from_numpy(y[None]))[0].numpy()
        t = min(ref.shape[1], got.shape[1])
        errs.append(np.abs(ref[:, :t] - got[:, :t]))
    e = np.concatenate([x.ravel() for x in errs])
    print(f"标准化特征逐点误差: mean {e.mean():.4f} | p95 {np.percentile(e, 95):.4f} "
          f"| max {e.max():.4f}  (特征本身 std=1)")

    if args.save_bias:
        out = Path(args.save_bias)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_bias": m.log_bias}, out)
        print(f"log_bias -> {out}  (gamma={CQT_GAMMA:g})")


if __name__ == "__main__":
    main()
