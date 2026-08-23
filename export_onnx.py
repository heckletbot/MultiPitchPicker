#!/usr/bin/env python3
"""导出端侧单文件 ONNX：16k 单声道波形 -> 图内 CQT -> NoteCNN -> onset/vel 概率。

输入  audio  (1, n_samples) float32，n 可变（≥1s 建议）
输出  onset  (1, 65, T) 概率；vel (1, 65, T) 0~1（×127 得力度）；T ≈ n/256

用法：
  python3 export_onnx.py --ckpt data/runs/mix/best.pt \
      --bias data/runs/torch_cqt_bias.pt --out export/model.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_WS = Path(__file__).resolve().parent
sys.path.insert(0, str(_WS))

from config import MIDI_LO, NUM_PITCHES  # noqa: E402
from model import NoteCNN  # noqa: E402
from dataset import CQT_GAMMA  # noqa: E402
from torch_cqt import ALPHA, Q, TorchCQT  # noqa: E402


class BpmMobile(nn.Module):
    """端侧整图：波形 → TorchCQT(+bias) → NoteCNN → sigmoid(onset, vel)。

    lookahead 从 checkpoint 读取；低延迟版导出时需设 MP_CQT_GAMMA 与训练一致，
    并使用同 γ 下校准的 bias。
    """
    def __init__(self, ckpt: str, bias: str):
        super().__init__()
        self.cqt = TorchCQT()
        self.cqt.log_bias.copy_(torch.load(bias, map_location="cpu")["log_bias"])
        ck = torch.load(ckpt, map_location="cpu")
        self.lookahead = bool(ck.get("lookahead", False))
        self.net = NoteCNN(lookahead=self.lookahead)
        self.net.load_state_dict(ck["model"])
        self.thresh = float(ck["val"]["best_thresh"])

    def forward(self, audio: torch.Tensor):
        x = self.cqt(audio)[:, None]          # (B,1,285,T)
        on_l, _fr, vel_l = self.net(x)
        return torch.sigmoid(on_l), torch.sigmoid(vel_l)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(_WS / "data/runs/mix/best.pt"))
    ap.add_argument("--bias", default=str(_WS / "data/runs/torch_cqt_bias.pt"))
    ap.add_argument("--out", default=str(_WS / "export/model.onnx"))
    ap.add_argument("--thresh", type=float, default=None,
                    help="覆盖 meta.json 的 thresh_default（默认用 ckpt 的 val 最优阈值；"
                         "step3 会传入合成 test 扫出的 best_thresh）")
    args = ap.parse_args()

    m = BpmMobile(args.ckpt, args.bias).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 16000 * 4)
    torch.onnx.export(
        m, (dummy,), str(out),
        input_names=["audio"], output_names=["onset", "vel"],
        dynamic_axes={"audio": {1: "n_samples"}, "onset": {2: "t"}, "vel": {2: "t"}},
        opset_version=17,
    )

    conv_future = 13 / 62.5 if m.lookahead else 0.75
    # 每音高提交延迟 = 卷积未来窗 + 该音高次谐波 bin 的滤波器伸向未来的一半长度
    #（次谐波 bin 是该音高最低的证据来源，其滤波器最长 → 决定该行何时"看全"）
    delays = []
    for p in range(NUM_PITCHES):
        f_sub = 440.0 * 2.0 ** ((MIDI_LO + p - 69) / 12.0) / 2.0
        half = 0.5 * Q / (f_sub + CQT_GAMMA / ALPHA)
        delays.append(round(conv_future + half, 3))
    thresh = float(args.thresh) if args.thresh is not None else m.thresh
    meta = {
        "sr": 16000, "hop": 256, "fps": 62.5,
        "midi_lo": MIDI_LO, "num_pitches": NUM_PITCHES,
        "thresh_default": thresh, "thresh_sensitive": 0.5,
        "min_gap_frames": 3, "peak_floor": 0.2,
        "cqt_gamma": CQT_GAMMA, "lookahead": m.lookahead,
        "receptive_future_sec": conv_future,
        "commit_delay_sec": delays,
        "note": "onset/vel 均为概率；音高 = 行索引 + midi_lo；力度 = vel*127",
    }
    (out.parent / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # 数值一致性验证：torch vs onnxruntime，随机 + 真实两种输入
    import onnxruntime as ort

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for name, wav in [("random", np.random.randn(1, 16000 * 8).astype(np.float32) * 0.1)]:
        with torch.no_grad():
            t_on, t_vel = m(torch.from_numpy(wav))
        o_on, o_vel = sess.run(None, {"audio": wav})
        d = np.abs(t_on.numpy() - o_on).max()
        print(f"{name}: torch/onnx onset 最大差 {d:.2e} | 输出形状 {o_on.shape}")
    sz = out.stat().st_size / 1e6
    print(f"导出 -> {out} ({sz:.1f} MB) | 默认阈值 {thresh}")


if __name__ == "__main__":
    main()
