#!/usr/bin/env python3
"""把原始音频收进 work/raw/inbox：支持目录拷贝、zip 解压、可选 session CSV。

自定义乐器：把录音丢进一个文件夹，或写 examples/session_sheet.example.csv 那样的表。
Iowa 下载产物可直接指向 work/raw/iowa_marimba，不必再 ingest。
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import AUDIO_EXTS, ensure_work  # noqa: E402


def _copy_audio(src: Path, dest_dir: Path) -> int:
    n = 0
    if src.is_file() and src.suffix.lower() in AUDIO_EXTS | {".zip"}:
        files = [src]
    else:
        files = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS | {".zip"}]
    for p in files:
        dest = dest_dir / p.name
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                z.extractall(dest_dir / p.stem)
            n += 1
            print(f"  unzip {p.name} -> {dest_dir / p.stem}")
        else:
            shutil.copy2(p, dest)
            n += 1
    return n


def _ingest_sheet(sheet: Path, dest_dir: Path) -> int:
    n = 0
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"空表: {sheet}")
    for r in rows:
        src = Path(r["path"])
        if not src.is_file():
            src = sheet.parent / src
        if not src.is_file():
            raise SystemExit(f"找不到 {r['path']}")
        midi = int(r["midi"])
        dyn = r["dynamic"].strip().lower()
        timbre = r["timbre"].strip().lower()
        name = f"{_midi_note(midi)}.{timbre}.{dyn}{src.suffix.lower()}"
        shutil.copy2(src, dest_dir / name)
        n += 1
    return n


def _midi_note(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def main() -> None:
    ap = argparse.ArgumentParser(description="收集原始音频到 work/raw/inbox")
    ap.add_argument("--src", help="原始目录或单个 zip/音频")
    ap.add_argument("--sheet", help="CSV：path,midi,dynamic,timbre（自定义逐条单音）")
    ap.add_argument("--out", default=str(ensure_work("raw", "inbox")))
    args = ap.parse_args()
    if not args.src and not args.sheet:
        raise SystemExit("请提供 --src 或 --sheet")

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if args.src:
        n += _copy_audio(Path(args.src), dest)
    if args.sheet:
        n += _ingest_sheet(Path(args.sheet), dest)
    print(f"ingest {n} -> {dest}")


if __name__ == "__main__":
    main()
