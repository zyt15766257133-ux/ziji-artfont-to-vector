"""从官方公开来源下载字体和研究文档，记录提交、授权和 SHA256。"""
from __future__ import annotations

import hashlib
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
FAMILIES = ["zcoolxiaowei", "zcoolkuaile", "zcoolqingkehuangyou", "longcang",
            "mashanzheng", "zhimangxing", "liujianmaocao", "notosanssc"]
SOURCES = {
    "deepvecfont": "https://raw.githubusercontent.com/yizhiwang96/deepvecfont/master/README.md",
    "deepvecfont-v2": "https://raw.githubusercontent.com/yizhiwang96/DeepVecFont-v2/main/README.md",
    "starvector": "https://raw.githubusercontent.com/joanrod/star-vector/main/README.md",
    "svg-fonts": "https://huggingface.co/datasets/starvector/svg-fonts/raw/main/README.md",
    "live": "https://raw.githubusercontent.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization/master/README.md",
    "diffvg": "https://raw.githubusercontent.com/BachiLi/diffvg/master/README.md",
    "vtracer": "https://raw.githubusercontent.com/visioncortex/vtracer/master/README.md",
    "make-me-a-hanzi": "https://raw.githubusercontent.com/skishore/makemeahanzi/master/README.md",
}


def fetch(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "Ziji-local-research/1.0"})
    with urlopen(req, timeout=90) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", help="Google Fonts commit; default reuses local manifest if present")
    args = parser.parse_args()
    fonts = ROOT / "data" / "fonts"
    refs = ROOT / "research" / "sources"
    fonts.mkdir(parents=True, exist_ok=True)
    refs.mkdir(parents=True, exist_ok=True)
    existing = fonts / "manifest.json"
    commit = args.commit
    if not commit and existing.exists():
        commit = json.loads(existing.read_text(encoding="utf-8"))[0]["commit"]
    if not commit:
        commit = json.loads(fetch("https://api.github.com/repos/google/fonts/commits/main"))["sha"]

    def font(family: str) -> dict:
        directory = fonts / family
        directory.mkdir(exist_ok=True)
        entries = json.loads(fetch(f"https://api.github.com/repos/google/fonts/contents/ofl/{family}?ref={commit}"))
        selected = [e for e in entries if e["name"].lower().endswith((".ttf", ".otf")) or e["name"] == "OFL.txt"]
        if not any(e["name"] == "OFL.txt" for e in selected):
            raise RuntimeError(f"{family}: missing license")
        files = []
        for entry in selected:
            blob = fetch(entry["download_url"])
            (directory / entry["name"]).write_bytes(blob)
            files.append({"name": entry["name"], "url": entry["download_url"],
                          "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
        result = {"family": family, "license": "SIL Open Font License 1.1",
                  "commit": commit, "files": files}
        (directory / "source.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"font {family}: {sum(f['bytes'] for f in files):,} bytes", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = list(pool.map(font, FAMILIES))
    (fonts / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "research/font_sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, url in SOURCES.items():
        try:
            blob = fetch(url)
            (refs / f"{name}.md").write_bytes(blob)
            print(f"source {name}: {len(blob):,} bytes", flush=True)
        except Exception as exc:
            print(f"source {name}: {exc}", flush=True)
    (refs / "urls.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
