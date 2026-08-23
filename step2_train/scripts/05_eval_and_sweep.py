#!/usr/bin/env python3
"""在合成 test 上扫描阈值，写出报告并推荐 τ。

换乐器必须重扫，不要照抄马林巴的 0.8。
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
    REPO_ROOT, ensure_work, load_config, pick_ckpt, write_json,
)

sys.path.insert(0, str(REPO_ROOT))
from dataset import CqtOafDataset  # noqa: E402
from model import NoteCNN  # noqa: E402
from train import collate, evaluate  # noqa: E402


def main() -> None:
    cfg = load_config()
    evcfg = cfg["eval"]
    ap = argparse.ArgumentParser(description="合成 test 阈值扫描")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else pick_ckpt(cfg)
    if not ckpt.is_file():
        ckpt = REPO_ROOT / ckpt
    test = Path(args.test) if args.test else REPO_ROOT / cfg["synth_dir"] / "test.jsonl"
    if not test.is_file():
        raise SystemExit(f"找不到 test: {test}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt, map_location=device)
    model = NoteCNN(lookahead=bool(ck.get("lookahead", False))).to(device).eval()
    model.load_state_dict(ck["model"])

    ds = CqtOafDataset(str(test))
    rows = {r["id"]: r for r in ds.rows}
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    threshes = [round(t, 2) for t in np.arange(
        evcfg["thresh_lo"], evcfg["thresh_hi"] + 1e-9, evcfg["thresh_step"])]
    ev = evaluate(model, ld, rows, device, threshes)

    sweep_out = {}
    for th, met in ev["sweep"].items():
        sweep_out[f"{float(th):.2f}"] = {
            "precision": round(met["precision"], 4),
            "recall": round(met["recall"], 4),
            "f1": round(met["f1"], 4),
        }

    best_th = float(ev["best_thresh"])
    warn = float(ev["f1"]) < float(cfg["accept"]["synth_f1_warn_below"])
    report = {
        "ckpt": str(ckpt).replace("\\", "/"),
        "test": str(test).replace("\\", "/"),
        "n": len(ds),
        "best_thresh": best_th,
        "precision": round(ev["precision"], 4),
        "recall": round(ev["recall"], 4),
        "f1": round(ev["f1"], 4),
        "velocity_mae": round(ev["vel_mae"], 2),
        "sweep": sweep_out,
        "warn_f1_below_accept": warn,
        "accept_note": cfg["accept"]["note"],
        "marimba_reference": {"f1": 0.994, "velocity_mae": 7.0, "thresh": 0.8},
    }
    run_dir = ckpt.parent
    write_json(run_dir / "test_report.json", report)
    write_json(ensure_work() / "test_report.json", report)
    write_json(ensure_work() / "selected.json", {
        "ckpt": ckpt.resolve().as_posix(),
        "best_thresh": best_th,
        "f1": report["f1"],
        "velocity_mae": report["velocity_mae"],
        "lookahead": bool(ck.get("lookahead", False)),
    })
    print(json.dumps({k: report[k] for k in
                      ("ckpt", "n", "best_thresh", "precision", "recall", "f1", "velocity_mae",
                       "warn_f1_below_accept")}, ensure_ascii=False, indent=2))
    if warn:
        print(f"警告: F1={ev['f1']:.4f} 低于 {cfg['accept']['synth_f1_warn_below']}，"
              "先回头查单音库 / 切分 / 音域，不要急着导出。")
    print(f"推荐阈值 τ={best_th}  （下一步导出和客户端都用这个）")
    print(f"报告 {run_dir / 'test_report.json'}")


if __name__ == "__main__":
    main()
