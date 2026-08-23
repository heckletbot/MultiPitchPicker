#!/usr/bin/env python3
"""训练三头转录模型（CQT + 谐波堆叠 + onset/frame/vel 三头）。

loss = BCE(frame) + 2*BCE(onset, pos_weight) + masked_MSE(velocity)
每 epoch 在 val 做 note-level onset F1（阈值扫描，62.5fps 解码），按 F1 存 best。

用法：
  python3 train.py --data data/synth --run-name base --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from config import MIDI_LO, RUNS_DIR  # noqa: E402
from dataset import CQT_FPS, CqtOafDataset  # noqa: E402
from metrics import note_f1  # noqa: E402
from model import NoteCNN  # noqa: E402


# ---------- 解码：先找峰，再按阈值筛，供 val / 伪标 / 端侧共用 ----------
def find_peaks(onset_prob: np.ndarray, floor: float = 0.2) -> list[list[tuple[int, float]]]:
    """每音高的局部极大点（±2 帧窗内最大且 >= floor），向量化，一次算完供全部阈值复用。"""
    from scipy.ndimage import maximum_filter1d

    local_max = maximum_filter1d(onset_prob, size=5, axis=1, mode="nearest")
    is_peak = (onset_prob >= local_max - 1e-9) & (onset_prob >= floor)
    peaks: list[list[tuple[int, float]]] = []
    for p in range(onset_prob.shape[0]):
        ts = np.where(is_peak[p])[0]
        peaks.append([(int(t), float(onset_prob[p, t])) for t in ts])
    return peaks


def decode_from_peaks(peaks, thresh: float, fps: float,
                      min_gap_frames: int = 3) -> list[tuple[int, float, float]]:
    """峰值列表 -> 音符事件（62.5fps 下 min_gap=3 ≈ 48ms）。"""
    out = []
    for p, cand in enumerate(peaks):
        kept: list[tuple[int, float]] = []
        for t, v in cand:
            if v < thresh:
                continue
            if kept and t - kept[-1][0] < min_gap_frames:
                if v > kept[-1][1]:
                    kept[-1] = (t, v)
            else:
                kept.append((t, v))
        out.extend((p, t / fps, float(v)) for t, v in kept)
    out.sort(key=lambda x: x[1])
    return out


def collate(batch):
    xs, ons, frs, vels, ids = zip(*batch)
    return (torch.stack(xs), torch.stack(ons), torch.stack(frs),
            torch.stack(vels), list(ids))


@torch.no_grad()
def evaluate(model, loader, rows_by_id, device, threshes):
    model.eval()
    agg = {th: {"tp": 0, "fp": 0, "fn": 0} for th in threshes}
    vel_abs_err, vel_n = 0.0, 0
    for x, on_t, _, vel_t, ids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            on_l, _, vel_l = model(x)
        on_prob = torch.sigmoid(on_l.float()).cpu().numpy()
        vel_pred = torch.sigmoid(vel_l.float()).cpu().numpy()
        mask = on_t.numpy() > 0.5
        if mask.sum() > 0:
            vel_abs_err += np.abs(vel_pred[mask] - vel_t.numpy()[mask]).sum()
            vel_n += int(mask.sum())
        for b, cid in enumerate(ids):
            gt = [(nt["midi"] - MIDI_LO, nt["onset"]) for nt in rows_by_id[cid]["notes"]]
            peaks = find_peaks(on_prob[b])
            for th in threshes:
                m = note_f1(decode_from_peaks(peaks, th, CQT_FPS), gt)
                for k in ("tp", "fp", "fn"):
                    agg[th][k] += m[k]
    out = {}
    for th, a in agg.items():
        prec = a["tp"] / max(1, a["tp"] + a["fp"])
        rec = a["tp"] / max(1, a["tp"] + a["fn"])
        out[th] = {"precision": prec, "recall": rec,
                   "f1": 2 * prec * rec / max(1e-9, prec + rec)}
    best_th = max(out, key=lambda t: out[t]["f1"])
    vel_mae = (vel_abs_err / max(1, vel_n)) * 127.0
    return {"best_thresh": best_th, "vel_mae": vel_mae, **out[best_th], "sweep": out}


def main():
    # 阶段 A：--data 合成集，无 --init
    # 阶段 B：--train-jsonl 混合清单 + --init 基座 + 更小 lr
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth")
    ap.add_argument("--train-jsonl", default=None, help="覆盖 <data>/train.jsonl（混合微调用）")
    ap.add_argument("--init", default=None, help="从已有 checkpoint 初始化（微调）")
    ap.add_argument("--lookahead", action="store_true",
                    help="非对称感受野：未来 0.21s/过去 1.3s（低延迟版，配合 MP_CQT_GAMMA=2）")
    ap.add_argument("--run-name", default="base")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--onset-pos-weight", type=float, default=4.0)
    ap.add_argument("--onset-loss-weight", type=float, default=2.0)
    ap.add_argument("--vel-loss-weight", type=float, default=1.0)
    args = ap.parse_args()

    ws = _DIR
    data = (ws / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    run_dir = RUNS_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = CqtOafDataset(args.train_jsonl if args.train_jsonl else data / "train.jsonl")
    val_ds = CqtOafDataset(data / "validation.jsonl")
    val_rows = {r["id"]: r for r in val_ds.rows}
    pw = args.workers > 0
    pin = device == "cuda"
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, collate_fn=collate,
                          pin_memory=pin, drop_last=True, persistent_workers=pw)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(4, args.workers) if args.workers else 0,
                        collate_fn=collate, persistent_workers=pw)

    model = NoteCNN(lookahead=args.lookahead).to(device)
    if args.init:
        ck = torch.load(args.init, map_location=device)
        model.load_state_dict(ck["model"])
        print(f"init from {args.init} (epoch {ck.get('epoch')})", flush=True)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M | "
          f"train {len(train_ds)} val {len(val_ds)} | fps {CQT_FPS}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    pos_w = torch.full((1,), args.onset_pos_weight, device=device)
    bce_on = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bce_fr = nn.BCEWithLogitsLoss()
    threshes = [round(t, 2) for t in np.arange(0.25, 0.91, 0.05)]

    best_f1 = -1.0
    log_path = run_dir / "train_log.jsonl"
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n_step = 0.0, 0
        for x, on_t, fr_t, vel_t, _ in train_ld:
            x = x.to(device, non_blocking=True)
            on_t = on_t.to(device, non_blocking=True)
            fr_t = fr_t.to(device, non_blocking=True)
            vel_t = vel_t.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                on_l, fr_l, vel_l = model(x)
                on_l, fr_l, vel_l = on_l.float(), fr_l.float(), vel_l.float()
                # velocity 只在真起音帧上回归，避免静音帧把力度拉向 0
                mask = (on_t > 0.5).float()
                vel_mse = (((torch.sigmoid(vel_l) - vel_t) ** 2) * mask).sum() / mask.sum().clamp(min=1.0)
                loss = (bce_fr(fr_l, fr_t)
                        + args.onset_loss_weight * bce_on(on_l, on_t)
                        + args.vel_loss_weight * vel_mse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            n_step += 1
        sched.step()
        ev = evaluate(model, val_ld, val_rows, device, threshes)
        rec = {"epoch": ep, "loss": tot / max(1, n_step), "val_f1": ev["f1"],
               "val_precision": ev["precision"], "val_recall": ev["recall"],
               "vel_mae": ev["vel_mae"], "best_thresh": ev["best_thresh"],
               "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
        print(json.dumps(rec), flush=True)
        with log_path.open("a") as f:
            f.write(json.dumps({**rec, "sweep": ev["sweep"]}) + "\n")
        ck = {"model": model.state_dict(), "epoch": ep, "val": ev,
              "arch": "note_cnn", "lookahead": args.lookahead}
        torch.save(ck, run_dir / "last.pt")
        if ev["f1"] > best_f1:
            best_f1 = ev["f1"]
            torch.save(ck, run_dir / "best.pt")
            print(f"  -> new best f1={best_f1:.4f} @{ev['best_thresh']} "
                  f"vel_mae={ev['vel_mae']:.1f}", flush=True)
    print(f"done. best val f1={best_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
