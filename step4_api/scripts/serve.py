#!/usr/bin/env python3
"""乐器转录 HTTP API：上传音频或给本地路径，返回音符事件列表。

解码调用根目录 infer_onnx.transcribe_file，不另写 onset 峰检测。

  python step4_api/scripts/serve.py
  python step4_api/scripts/serve.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import REPO_ROOT, default_assets, load_config, load_selected_thresh  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from infer_onnx import load_session, transcribe_file  # noqa: E402

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aif", ".aiff"}

app = FastAPI(
    title="MultiPitchPicker",
    description="单乐器自动转录：波形进，音符事件出（ONNX）。",
    version="1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATE: dict = {
    "sess": None,
    "meta": None,
    "onnx": None,
    "assets": None,
    "error": None,
}


def _assets_dir() -> Path:
    env = os.environ.get("IT_ASSETS")
    if env:
        return Path(env)
    return default_assets()


def try_load() -> None:
    assets = _assets_dir()
    try:
        sess, meta, onnx = load_session(assets)
    except Exception as e:  # noqa: BLE001
        _STATE.update(sess=None, meta=None, onnx=None, assets=assets, error=str(e))
        return
    _STATE.update(sess=sess, meta=meta, onnx=onnx, assets=assets, error=None)


@app.on_event("startup")
def _startup() -> None:
    try_load()


def _default_thresh() -> float | None:
    sel = load_selected_thresh()
    if sel is not None:
        return sel
    meta = _STATE.get("meta") or {}
    if meta.get("thresh_default") is not None:
        return float(meta["thresh_default"])
    return None


def _require_model() -> None:
    if _STATE.get("sess") is None:
        try_load()
    if _STATE.get("sess") is None:
        raise HTTPException(
            status_code=503,
            detail=_STATE.get("error") or "模型未加载。请先跑 step3_eval_export，确认 export/model.onnx 存在。",
        )


def _run_transcribe(path: Path, thresh: float | None) -> dict:
    _require_model()
    th = thresh if thresh is not None else _default_thresh()
    result = transcribe_file(
        path, thresh=th, sess=_STATE["sess"], meta=_STATE["meta"],
    )
    return {
        "duration": result["duration"],
        "thresh": result["thresh"],
        "n_notes": result["n_notes"],
        "notes": result["notes"],
    }


async def _save_upload(file) -> Path:
    suffix = Path(getattr(file, "filename", None) or "audio.wav").suffix.lower() or ".wav"
    if suffix not in AUDIO_EXTS:
        suffix = ".wav"
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="上传文件是空的")
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="it_upload_")
    os.close(fd)
    Path(tmp).write_bytes(data)
    return Path(tmp)


async def _resolve_audio(request: Request, path: str | None):
    """multipart 文件、query/form path、或 JSON {path}。不在签名里用 File()，以免挡住 JSON。"""
    tmp: Path | None = None
    audio: Path | None = Path(path) if path else None
    thresh_body: float | None = None
    ctype = (request.headers.get("content-type") or "").lower()

    if "multipart/form-data" in ctype:
        form = await request.form()
        up = form.get("file")
        if up is not None and hasattr(up, "read") and getattr(up, "filename", None):
            tmp = await _save_upload(up)
            audio = tmp
        if form.get("path"):
            audio = Path(str(form.get("path")))
        if form.get("thresh") not in (None, ""):
            thresh_body = float(str(form.get("thresh")))
    elif "application/json" in ctype:
        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"JSON 无法解析: {e}") from e
        if isinstance(body, dict):
            if body.get("path"):
                audio = Path(str(body["path"]))
            if body.get("thresh") is not None:
                thresh_body = float(body["thresh"])

    if audio is None:
        raise HTTPException(
            status_code=400,
            detail="请上传 file，或提供 path（query / form / JSON）。",
        )
    if tmp is None and not audio.is_file():
        raise HTTPException(status_code=400, detail=f"找不到音频: {audio}")
    return audio, tmp, thresh_body


@app.get("/")
def root():
    return {
        "service": "MultiPitchPicker",
        "docs": "/docs",
        "health": "/health",
        "transcribe": "POST /transcribe",
    }


@app.get("/health")
def health():
    if _STATE.get("sess") is None:
        try_load()
    meta = _STATE.get("meta") or {}
    onnx = _STATE.get("onnx")
    return {
        "ok": _STATE.get("sess") is not None,
        "loaded": _STATE.get("sess") is not None,
        "assets": str(_STATE.get("assets") or _assets_dir()),
        "onnx": str(onnx) if onnx is not None else None,
        "thresh_default": _default_thresh(),
        "midi_lo": meta.get("midi_lo"),
        "num_pitches": meta.get("num_pitches"),
        "error": _STATE.get("error"),
    }


@app.post(
    "/transcribe",
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "file": {"type": "string", "format": "binary"},
                            "path": {"type": "string"},
                            "thresh": {"type": "number"},
                        },
                    }
                },
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "thresh": {"type": "number"},
                        },
                    }
                },
            }
        }
    },
)
async def transcribe_endpoint(
    request: Request,
    path: str | None = Query(default=None, description="本机音频路径"),
    thresh: float | None = Query(default=None, description="onset 阈值，默认读 meta / selected.json"),
):
    tmp: Path | None = None
    try:
        audio, tmp, thresh_body = await _resolve_audio(request, path)
        th = thresh if thresh is not None else thresh_body
        return _run_transcribe(audio, th)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="转录 HTTP API")
    ap.add_argument("--host", default=cfg.get("host", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(cfg.get("port", 8000)))
    ap.add_argument("--assets", default=None, help="ONNX + meta.json 目录，默认 export/")
    args = ap.parse_args()
    if args.assets:
        os.environ["IT_ASSETS"] = str(Path(args.assets).resolve())
    import uvicorn

    print(f"assets={_assets_dir()}")
    print(f"docs  http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
