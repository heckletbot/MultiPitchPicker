#!/usr/bin/env python3
"""step1 共用路径、音名解析、音频读写、jsonl。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

STEP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = STEP_DIR.parent
WORK = STEP_DIR / "work"
CONFIG_PATH = STEP_DIR / "config_prep.json"

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}

_NOTE_OFFSET = {
    "C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4,
    "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9,
    "A#": 10, "BB": 10, "B": 11,
}
_NOTE_RE = re.compile(r"^([A-Ga-g][#b]?)(-?\d+)$")
# Marimba.yarn.pp.C2B2.aif  /  Marimba.rubber.ff.C4.stereo.aif
_IOWA_RE = re.compile(
    r"^(?P<inst>[^.]+)\.(?P<timbre>[^.]+)\.(?P<dyn>pp|mf|ff)\."
    r"(?P<span>[A-Ga-g][#b]?\-?\d+(?:[A-Ga-g][#b]?\-?\d+)?)"
    r"(?:\.(?P<extra>stereo|mono))?$",
    re.I,
)
# C4.bright.mf.wav  （仓库 examples 的通用命名）
_GENERIC_RE = re.compile(
    r"^(?P<note>[A-Ga-g][#b]?\-?\d+)\.(?P<timbre>[^.]+)\.(?P<dyn>pp|mf|ff)$",
    re.I,
)


def load_config(path: Path | None = None) -> dict:
    p = path or CONFIG_PATH
    return json.loads(p.read_text(encoding="utf-8"))


def ensure_work(*parts: str) -> Path:
    d = WORK.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def note_to_midi(name: str) -> int:
    m = _NOTE_RE.match(name.replace(" ", ""))
    if not m:
        raise ValueError(f"无法解析音名: {name}")
    pc, octv = m.group(1).upper(), int(m.group(2))
    if pc not in _NOTE_OFFSET:
        raise ValueError(f"未知音级: {name}")
    return (octv + 1) * 12 + _NOTE_OFFSET[pc]


def midi_to_note(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"


def parse_span(span: str) -> list[int]:
    """C2B2 → [36..47]；C7 → [96]。"""
    m = re.match(
        r"^([A-Ga-g][#b]?)(-?\d+)(?:([A-Ga-g][#b]?)(-?\d+))?$",
        span.replace(" ", ""),
    )
    if not m:
        raise ValueError(f"无法解析音域片段: {span}")
    lo = note_to_midi(m.group(1) + m.group(2))
    if m.group(3) is None:
        return [lo]
    hi = note_to_midi(m.group(3) + m.group(4))
    if hi < lo:
        raise ValueError(f"音域上下界颠倒: {span}")
    return list(range(lo, hi + 1))


def parse_filename(path: Path) -> dict | None:
    """从 Iowa / 通用文件名抽出 {timbre, dynamic, midis}。失败返回 None。"""
    stem = path.stem
    m = _IOWA_RE.match(stem)
    if m:
        return {
            "instrument": m.group("inst"),
            "timbre": m.group("timbre").lower(),
            "dynamic": m.group("dyn").lower(),
            "midis": parse_span(m.group("span")),
            "source": "iowa",
        }
    m = _GENERIC_RE.match(stem)
    if m:
        return {
            "instrument": None,
            "timbre": m.group("timbre").lower(),
            "dynamic": m.group("dyn").lower(),
            "midis": [note_to_midi(m.group("note"))],
            "source": "generic",
        }
    return None


def iter_audio(root: Path) -> list[Path]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return files


def load_mono(path: Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    import librosa

    y, s = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), int(s)


def write_pcm16_wav(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(str(path), y, sr, subtype="PCM_16")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def python_bin() -> str:
    return sys.executable
