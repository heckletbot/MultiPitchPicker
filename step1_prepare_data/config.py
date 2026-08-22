#!/usr/bin/env python3
"""音高范围与运行目录。换乐器先改 MIDI_LO / MIDI_HI，并与同目录 config_prep.json 保持一致。

默认音域 MIDI 36–100（C2–E7，马林巴），65 个音高类。改音域时注意耦合：

  NUM_PITCHES = MIDI_HI - MIDI_LO + 1          # 三头输出行数（自动）
  model.py:  PITCH_BINS = NUM_PITCHES * 3      # 自动；3 bins/半音
             PITCH_OFF  = (MIDI_LO - 24) * 3   # 自动；24 = C1 = CQT 最低 bin 的 MIDI
  dataset.py: CQT_FMIN=C1(32.7Hz), CQT_BINS=285 —— 下界须 ≤ 最低音基频的一半
             （次谐波通道要在图内）。默认支持 MIDI_LO ≥ 36；更低的乐器
             （如钢琴 A0=21）要把 CQT_FMIN 再降一个八度并同步加 CQT_BINS。

改完必须重跑本步预处理，并重训、重导出。
"""
from __future__ import annotations

from pathlib import Path

MIDI_LO = 36
MIDI_HI = 100
NUM_PITCHES = MIDI_HI - MIDI_LO + 1

# 本文件在 step1_prepare_data/ 内；runs 目录在仓库根 data/runs
RUNS_DIR = Path(__file__).resolve().parent.parent / "data" / "runs"
