#!/usr/bin/env python3
"""从 University of Iowa MIS 马林巴页下载「单音/半音阶」原始音频。

默认只下 yarn / cord / rubber 的 16-bit/44.1k 半音阶 .aif（跳过 roll、deadstroke）。
这些文件是把一个八度的音按半音挨个敲进同一条音轨，后续由 03_split_chromatic.py 切开。

许可：MIS 声明可免费用于任何项目；使用时请遵守 https://theremin.music.uiowa.edu/MIS.html
并考虑向 Iowa EMS 捐助。本脚本不重新分发音频。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from common import ensure_work, load_config  # noqa: E402

UA = "MultiPitchPicker-step1/1.0 (research; +https://theremin.music.uiowa.edu/MIS.html)"


class _HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def _fetch(url: str, dest: Path | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    return data


def collect_urls(page_url: str, keep_mallets: list[str], skip: list[str]) -> list[str]:
    html = _fetch(page_url).decode("utf-8", errors="replace")
    p = _HrefParser()
    p.feed(html)
    keep = {m.lower() for m in keep_mallets}
    skip_set = {s.lower() for s in skip}
    out, seen = [], set()
    for href in p.hrefs:
        full = urljoin(page_url, href)
        name = unquote(Path(urlparse(full).path).name)
        low = name.lower()
        if not low.endswith(".aif") and not low.endswith(".aiff"):
            continue
        # 只要半音阶长文件，不下 24/96 zip（体积大；需要时可 --hires）
        parts = name.split(".")
        if len(parts) < 4:
            continue
        mallet = parts[1].lower()
        if mallet in skip_set or mallet not in keep:
            continue
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
    out.sort()
    return out


def collect_zip_urls(page_url: str, keep_mallets: list[str], skip: list[str]) -> list[str]:
    html = _fetch(page_url).decode("utf-8", errors="replace")
    p = _HrefParser()
    p.feed(html)
    keep = {m.lower() for m in keep_mallets}
    skip_set = {s.lower() for s in skip}
    out = []
    for href in p.hrefs:
        full = urljoin(page_url, href)
        name = unquote(Path(urlparse(full).path).name).lower()
        if not name.endswith(".zip"):
            continue
        m = re.match(r"marimba\.([a-z]+)\.(pp|mf|ff)\.zip$", name)
        if not m:
            continue
        mallet = m.group(1)
        if mallet in skip_set or mallet not in keep:
            continue
        out.append(full)
    return sorted(set(out))


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="下载 Iowa MIS 马林巴原始音频")
    ap.add_argument("--page", default=cfg["iowa_page"])
    ap.add_argument("--out", default=str(ensure_work("raw", "iowa_marimba")))
    ap.add_argument("--hires", action="store_true",
                    help="改下 24/96 单音 zip（更大；解压后已是单音，可跳过切分）")
    ap.add_argument("--sleep", type=float, default=0.4, help="请求间隔，避免打满对方服务器")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = cfg["iowa_keep_mallets"]
    skip = cfg["skip_articulations"]
    urls = collect_zip_urls(args.page, keep, skip) if args.hires else collect_urls(args.page, keep, skip)
    if not urls:
        raise SystemExit(f"页面 {args.page} 上没有匹配的音频链接，请检查网络或手动下载。")

    print(f"将下载 {len(urls)} 个文件 -> {out}", flush=True)
    ok, skip_n, fail = 0, 0, 0
    for i, url in enumerate(urls, 1):
        name = unquote(Path(urlparse(url).path).name)
        dest = out / name
        if dest.is_file() and dest.stat().st_size > 0 and not args.force:
            skip_n += 1
            print(f"  [{i}/{len(urls)}] 已存在 {name}", flush=True)
            continue
        try:
            _fetch(url, dest)
            ok += 1
            print(f"  [{i}/{len(urls)}] {name}  {dest.stat().st_size/1e6:.1f} MB", flush=True)
        except urllib.error.URLError as e:
            fail += 1
            print(f"  [{i}/{len(urls)}] 失败 {name}: {e}", flush=True)
        time.sleep(args.sleep)
    print(f"完成: 新下 {ok}，跳过 {skip_n}，失败 {fail}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
