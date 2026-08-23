#!/usr/bin/env python3
"""在合成 test 上扫描阈值，写出完整 test_report.json 并推荐 τ。

比 step2 的 05 更完整：P/R/F1（小数 + 百分数）、力度 MAE、全 sweep、马林巴对照。
换乐器必须重扫，不要照抄马林巴的 0.8。

解码与 train.evaluate 相同（onset 峰 + ±50ms 匹配）。CPU 上不用 CUDA autocast。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    REPO_ROOT, STEP2_SELECTED, as_posix, ensure_work, load_config, pick_ckpt, write_json,
)

sys.path.insert(0, str(REPO_ROOT))
from dataset import CQT_FPS, CqtOafDataset  # noqa: E402
from model import NoteCNN  # noqa: E402
from train import collate, decode_from_peaks, find_peaks  # noqa: E402
from config import MIDI_LO  # noqa: E402
from metrics import note_f1  # noqa: E402


@torch.no_grad()
def evaluate_sweep(model, loader, rows_by_id, device, threshes):
    """与 train.evaluate 同一套解码；CPU 上跳过 CUDA autocast。"""
    model.eval()
    agg = {th: {"tp": 0, "fp": 0, "fn": 0} for th in threshes}
    vel_abs_err, vel_n = 0.0, 0
    use_amp = str(device).startswith("cuda")
    for x, on_t, _, vel_t, ids in loader:
        x = x.to(device, non_blocking=use_amp)
        if use_amp:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                on_l, _, vel_l = model(x)
        else:
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
        out[th] = {
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec),
            "tp": int(a["tp"]), "fp": int(a["fp"]), "fn": int(a["fn"]),
        }
    best_th = max(out, key=lambda t: out[t]["f1"])
    vel_mae = (vel_abs_err / max(1, vel_n)) * 127.0
    return {"best_thresh": best_th, "vel_mae": vel_mae, **out[best_th], "sweep": out}


def main() -> None:
    cfg = load_config()
    evcfg = cfg["eval"]
    ap = argparse.ArgumentParser(description="合成 test 阈值扫描（完整报告）")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--reuse-selected", action="store_true",
                    help="不重跑评测，只把已有 selected.json / test_report 拷到本步 work/")
    args = ap.parse_args()

    if args.reuse_selected:
        _reuse(cfg)
        return

    ckpt = Path(args.ckpt) if args.ckpt else pick_ckpt(cfg)[0]
    if not ckpt.is_file():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.is_file():
        raise SystemExit(f"找不到 checkpoint: {ckpt}\n请先跑 step2_train。")
    test = Path(args.test) if args.test else REPO_ROOT / cfg["synth_dir"] / "test.jsonl"
    if not test.is_file():
        raise SystemExit(f"找不到 test: {test}\n请先跑 step1_prepare_data。")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} | ckpt={ckpt}", flush=True)
    ck = torch.load(ckpt, map_location=device)
    lookahead = bool(ck.get("lookahead", False))
    model = NoteCNN(lookahead=lookahead).to(device).eval()
    model.load_state_dict(ck["model"])

    ds = CqtOafDataset(str(test))
    rows = {r["id"]: r for r in ds.rows}
    bs = args.batch_size if args.batch_size is not None else int(evcfg["batch_size"])
    ld = DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=collate)
    threshes = [round(t, 2) for t in np.arange(
        evcfg["thresh_lo"], evcfg["thresh_hi"] + 1e-9, evcfg["thresh_step"])]
    ev = evaluate_sweep(model, ld, rows, device, threshes)

    sweep_out = {}
    for th, met in ev["sweep"].items():
        sweep_out[f"{float(th):.2f}"] = {
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f1": round(met["f1"], 4),
            "tp": met["tp"], "fp": met["fp"], "fn": met["fn"],
        }

    best_th = float(ev["best_thresh"])
    warn = float(ev["f1"]) < float(cfg["accept"]["synth_f1_warn_below"])
    report = {
        "ckpt": as_posix(ckpt.resolve()),
        "test": as_posix(test.resolve()),
        "n": len(ds),
        "onset_tolerance_ms": int(evcfg["onset_tolerance_ms"]),
        "best_thresh": best_th,
        "precision": round(ev["precision"], 4),
        "recall": round(ev["recall"], 4),
        "f1": round(ev["f1"], 4),
        "precision_pct": round(ev["precision"] * 100, 2),
        "recall_pct": round(ev["recall"] * 100, 2),
        "f1_pct": round(ev["f1"] * 100, 2),
        "velocity_mae": round(ev["vel_mae"], 2),
        "lookahead": lookahead,
        "sweep": sweep_out,
        "warn_f1_below_accept": warn,
        "accept_note": cfg["accept"]["note"],
        "marimba_reference": {
            "f1": cfg["marimba_reference"]["f1"],
            "precision": cfg["marimba_reference"]["precision"],
            "recall": cfg["marimba_reference"]["recall"],
            "velocity_mae": cfg["marimba_reference"]["velocity_mae"],
            "thresh": cfg["marimba_reference"]["thresh"],
        },
    }
    selected = {
        "ckpt": ckpt.resolve().as_posix(),
        "best_thresh": best_th,
        "f1": report["f1"],
        "velocity_mae": report["velocity_mae"],
        "lookahead": lookahead,
    }
    run_dir = ckpt.parent
    write_json(run_dir / "test_report.json", report)
    work = ensure_work()
    write_json(work / "test_report.json", report)
    write_json(work / "selected.json", selected)
    write_json(STEP2_SELECTED, selected)

    summary = {k: report[k] for k in (
        "ckpt", "n", "best_thresh", "precision", "recall", "f1",
        "f1_pct", "velocity_mae", "lookahead", "warn_f1_below_accept")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if warn:
        print(f"警告: F1={ev['f1']:.4f} 低于 {cfg['accept']['synth_f1_warn_below']}，"
              "先回头查单音库 / 切分 / 音域，不要急着导出。")
    print(f"推荐阈值 τ={best_th}  （导出 meta.json 和客户端都用这个，不要照抄 0.8）")
    print(f"报告 {run_dir / 'test_report.json'}")
    print(f"报告 {work / 'test_report.json'}")


def _reuse(cfg: dict) -> None:
    from common import load_selected

    sel, src = load_selected()
    if not sel:
        raise SystemExit("没有 selected.json，无法 --reuse-selected。请先跑评测或 step2 的 05。")
    work = ensure_work()
    write_json(work / "selected.json", sel)
    cand = [
        Path(sel["ckpt"]).parent / "test_report.json" if sel.get("ckpt") else None,
        src.parent / "test_report.json" if src else None,
        REPO_ROOT / "data" / "runs" / cfg["mix_run"] / "test_report.json",
        REPO_ROOT / "data" / "runs" / cfg["base_run"] / "test_report.json",
    ]
    copied = False
    for p in cand:
        if p and p.is_file():
            report = json.loads(p.read_text(encoding="utf-8"))
            write_json(work / "test_report.json", report)
            copied = True
            print(f"复用报告 {p}")
            break
    print(json.dumps(sel, ensure_ascii=False, indent=2))
    print(f"复用 {src}")
    if not copied:
        print("没有现成 test_report.json，只有 selected.json。完整 sweep 请去掉 --reuse-selected。")


if __name__ == "__main__":
    main()
