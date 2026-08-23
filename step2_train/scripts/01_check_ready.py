#!/usr/bin/env python3
"""训练前环境检查：第一步数据齐不齐、音域能不能读到、GPU 有没有、真录独奏有没有。

不训练、不改数据。通过则打印 JSON 并写入 step2_train/work/check.json；
缺 train/val/test 清单或读不到 config.py / torch 则 exit 1。
缺 CQT 缓存、没有 CUDA、没有真录只警告，不阻断。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ensure_work, load_config, repo_path, write_json  # noqa: E402


def main() -> None:
    cfg = load_config()  # step2_train/config_train.json
    synth = repo_path(cfg["synth_dir"])  # 默认仓库根下 data/synth
    report: dict = {"ok": True, "errors": [], "warnings": []}

    # 1) 第一步合成清单：train / validation / test 三份 jsonl 必须在
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl"):
        p = synth / name
        if not p.is_file():
            report["ok"] = False
            report["errors"].append(f"缺少 {p}（先完成 step1）")
            continue
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        report[name] = n
        if name == "train.jsonl" and n < 32:
            report["warnings"].append(f"train 只有 {n} 条，正式训练建议 8000+")

    # 2) CQT 预计算缓存：没有也能训，但每条音频会现场算，很慢
    caches = list(synth.glob("**/*.cqt285.npy")) if synth.exists() else []
    report["cqt_cache_files"] = len(caches)
    if not caches:
        report["warnings"].append("没有 .cqt285.npy，训练时会现场算 CQT，会慢很多。请跑 step1 的 precompute。")

    # 3) 音域：优先仓库根 config.py，否则用第一步的 step1_prepare_data/config.py
    midi_ok = False
    last_err = None
    for cand in (repo_path("."), repo_path("step1_prepare_data")):
        try:
            sys.path.insert(0, str(cand))
            import config as repo_cfg  # noqa: WPS433
            report["midi_lo"] = repo_cfg.MIDI_LO
            report["midi_hi"] = repo_cfg.MIDI_HI
            report["num_pitches"] = repo_cfg.NUM_PITCHES
            report["config_path"] = str(cand / "config.py")
            midi_ok = True
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            sys.path = [p for p in sys.path if p != str(cand)]
    if not midi_ok:
        report["ok"] = False
        report["errors"].append(f"读 config.py 失败: {last_err}")

    # 4) PyTorch / GPU：无 CUDA 只警告
    try:
        import torch
        report["torch"] = torch.__version__
        report["cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
        else:
            report["warnings"].append("没有 CUDA，可以跑但会非常慢。")
    except Exception as e:  # noqa: BLE001
        report["ok"] = False
        report["errors"].append(f"未安装 torch: {e}")

    # 5) 真录独奏：没有则后面只能训 base，跳过伪标签微调
    real = repo_path(cfg["real_dir"])
    songs = []
    if real.is_dir():
        for ext in ("*.mp3", "*.wav", "*.flac", "*.m4a"):
            songs.extend(real.glob(ext))
    report["real_dir"] = str(real)
    report["real_songs"] = len(songs)
    report["has_pool_filters"] = (real / "pool_filters.json").is_file()
    if not songs:
        report["warnings"].append("没有 data/real 独奏，将只能训 base，跳过伪标签微调。")

    out = ensure_work() / "check.json"
    write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告 {out}")
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
