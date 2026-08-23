#!/usr/bin/env python3
"""checkpoint + CQT bias → 单文件 ONNX，并写出 export/meta.json。

阈值用测评选出的 best_thresh（selected.json），不是 checkpoint 里 val 的 τ。
meta 字段对齐 MultiPitchPicker-2/deploy/assets_bpm/meta.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import (  # noqa: E402
    as_posix, ensure_work, load_config, load_selected, pick_ckpt, repo_path, run_repo, write_json,
)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="导出 ONNX + meta.json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bias", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--thresh", type=float, default=None)
    args = ap.parse_args()

    ckpt, sel = pick_ckpt(cfg)
    if args.ckpt:
        ckpt = Path(args.ckpt)
        if not ckpt.is_file():
            ckpt = repo_path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"找不到 checkpoint: {ckpt}")

    bias = Path(args.bias) if args.bias else repo_path(cfg["bias"])
    if not bias.is_file():
        raise SystemExit(f"找不到 CQT bias: {bias}\n请先跑 03_calibrate_cqt.py")

    out = Path(args.out) if args.out else repo_path(cfg["export_dir"]) / cfg["onnx_name"]
    thresh = args.thresh
    if thresh is None and sel and sel.get("best_thresh") is not None:
        thresh = float(sel["best_thresh"])
    extra = ["--ckpt", str(ckpt), "--bias", str(bias), "--out", str(out)]
    if thresh is not None:
        extra.extend(["--thresh", str(thresh)])
    run_repo("export_onnx.py", extra)

    meta_path = out.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if thresh is not None:
        meta["thresh_default"] = float(thresh)
    meta["source_ckpt"] = as_posix(ckpt.resolve())
    if sel:
        if sel.get("f1") is not None:
            meta["eval_f1"] = sel["f1"]
        if sel.get("velocity_mae") is not None:
            meta["eval_velocity_mae"] = sel["velocity_mae"]
        if sel.get("lookahead") is not None:
            meta["lookahead"] = bool(sel["lookahead"])
    write_json(meta_path, meta)
    write_json(ensure_work() / "meta.json", meta)
    print(f"ONNX -> {out}")
    print(f"meta -> {meta_path}  thresh_default={meta['thresh_default']}")


if __name__ == "__main__":
    main()
