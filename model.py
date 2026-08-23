#!/usr/bin/env python3
"""BP 式乐器转录模型：log-CQT 输入 → 谐波堆叠(图内) → 纯卷积 → onset/frame/vel 三头。

设计对齐 Spotify Basic Pitch (ICASSP 2022) 的关键思想：
  1. CQT 频点按音高对数分布 → 低音分辨率结构性解决
  2. 谐波堆叠：h∈{0.5,1..7} 的能量按固定 bin 位移对齐到基频行，
     小卷积核即可看见完整谐波列 → 抑制"把泛音当音符"
  3. onset 头只对新起音发火 → 快速跑动不把残响糊成和弦
  4. 纯卷积（无 RNN）→ 手机流式/ONNX 导出无障碍

输入 (B,1,285,T) log-CQT；输出 onset/frame/vel 各 (B,65,T) logits。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config import MIDI_LO, NUM_PITCHES  # noqa: E402

CQT_BINS = 285
PITCH_OFF = (MIDI_LO - 24) * 3       # 最低音相对 C1(MIDI 24) 的 bin 偏移，3 bins/半音
PITCH_BINS = NUM_PITCHES * 3         # 每音高 3 bins（默认 195 = 65 × 3）
# h = [0.5, 1, 2, 3, 4, 5, 6, 7] → 位移 = round(36*log2(h))
HARM_SHIFTS = (-36, 0, 36, 57, 72, 84, 93, 101)


class HarmonicStack(nn.Module):
    """把每个基频行对齐到同一套谐波通道。

    输入 (B,1,285,T) → 输出 (B,8,195,T)。
    195 = 65 半音 × 3 bins/半音（C2 起）。超出 285 的位移补零。
    通道顺序对应 HARM_SHIFTS：次谐波、基频、2～7 次谐波。
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_hi = PITCH_OFF + PITCH_BINS + max(HARM_SHIFTS) - CQT_BINS  # 上方需补的行数
        xp = F.pad(x, (0, 0, 0, max(0, pad_hi)))
        chans = [xp[:, :, PITCH_OFF + s: PITCH_OFF + s + PITCH_BINS, :] for s in HARM_SHIFTS]
        return torch.cat(chans, dim=1)


class TimePad(nn.Module):
    """时间轴非对称零填充 (past, future)：低延迟版用它替代对称 padding。"""

    def __init__(self, past: int, future: int):
        super().__init__()
        self.past, self.future = past, future

    def forward(self, x):
        return F.pad(x, (self.past, self.future))


def _blk(cin, cout, k, stride=(1, 1), dil=(1, 1), tpad=None):
    """tpad=None：时间轴对称填充；tpad=(past, future)：非对称（低延迟版）。

    非对称不改变每层的总时间跨度 (k-1)*dil，只是把窗口向过去偏移：
    输出帧 t 参考 [t-past, t+future]，牺牲一部分"未来"换更多"过去"。
    """
    fpad = (k[0] - 1) // 2 * dil[0] if stride[0] == 1 else 0
    if tpad is None:
        pad = (fpad, (k[1] - 1) // 2 * dil[1])
        return nn.Sequential(nn.Conv2d(cin, cout, k, stride=stride, padding=pad, dilation=dil),
                             nn.BatchNorm2d(cout), nn.ReLU(inplace=True))
    return nn.Sequential(TimePad(*tpad),
                         nn.Conv2d(cin, cout, k, stride=stride, padding=(fpad, 0), dilation=dil),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class NoteCNN(nn.Module):
    """三头全卷积：onset / frame / velocity，输出均为 (B, 65, T) logits。

    lookahead=False：时间感受野对称 ±47 帧（±0.75s @62.5fps）——精度优先（离线用）。
    lookahead=True：非对称——未来 13 帧（0.21s）/ 过去 81 帧（1.30s），
      总跨度不变，用于低延迟实时。逐层未来预算见 tpad 注释。
    """

    def __init__(self, lookahead: bool = False):
        super().__init__()
        self.lookahead = lookahead
        # lookahead 时逐层 (past, future) 帧预算，future 合计 = 1+1+2+4+4+头1 = 13 帧
        la = (lambda p, f: (p, f)) if lookahead else (lambda p, f: None)
        self.stack = HarmonicStack()
        # 频轴：先在 3 bins/半音上混谐波，再 stride=3 收到 1 行/半音
        self.b1 = _blk(8, 32, (5, 5), tpad=la(3, 1))
        self.b2 = _blk(32, 48, (3, 5), stride=(3, 1), tpad=la(3, 1))   # 195 -> 65
        # 时间轴膨胀卷积，对称时感受野约 ±47 帧 ≈ ±0.75s（实时要等这段上下文）
        self.b3 = _blk(48, 64, (3, 7), dil=(1, 2), tpad=la(10, 2))
        self.b4 = _blk(64, 64, (3, 7), dil=(1, 4), tpad=la(20, 4))
        self.b5 = _blk(64, 64, (3, 7), dil=(1, 8), tpad=la(44, 4))
        self.head_on = nn.Conv2d(64, 1, (3, 3), padding=1)
        self.head_fr = nn.Conv2d(64, 1, (3, 3), padding=1)
        self.head_vel = nn.Conv2d(64, 1, (3, 3), padding=1)

    def forward(self, x):
        h = self.stack(x)
        h = self.b1(h)
        h = self.b2(h)
        h = self.b3(h)
        h = self.b4(h)
        h = self.b5(h)
        on = self.head_on(h).squeeze(1)    # (B,65,T)
        fr = self.head_fr(h).squeeze(1)
        vel = self.head_vel(h).squeeze(1)
        return on, fr, vel


if __name__ == "__main__":
    x = torch.randn(2, 1, 285, 501)
    for la in (False, True):
        m = NoteCNN(lookahead=la)
        n = sum(p.numel() for p in m.parameters())
        on, fr, vel = m(x)
        print(f"lookahead={la} | params {n/1e3:.0f}K | out {tuple(on.shape)}")
