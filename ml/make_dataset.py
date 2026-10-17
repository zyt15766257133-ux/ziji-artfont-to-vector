"""OFL 字体轮廓 -> SVG -> resvg 真值 -> 退化 RGB，按汉字和字体隔离。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw

from app.pipeline.native_render import render_svg
from .model import features

ROOT = Path(__file__).resolve().parent
HELDOUT = {"longcang", "liujianmaocao"}
STYLES = ["clean", "noise", "paper", "uneven", "jpeg"]


def open_fonts():
    result = {}
    for directory in sorted((ROOT / "data/fonts").iterdir()):
        if not directory.is_dir():
            continue
        path = next(iter(sorted(directory.glob("*.ttf"))))
        font = TTFont(path)
        result[directory.name] = (font, font.getGlyphSet(), font.getBestCmap())
    return result


def glyph_svg(font, glyphs, cmap, char, rng, size=128):
    name = cmap[ord(char)]
    pen = SVGPathPen(glyphs)
    glyphs[name].draw(pen)
    bp = BoundsPen(glyphs)
    glyphs[name].draw(bp)
    x0, y0, x1, y1 = bp.bounds
    scale = rng.uniform(0.58, 0.84) * size / max(x1 - x0, y1 - y0)
    sx = scale * rng.uniform(0.83, 1.1)
    sy = scale * rng.uniform(0.88, 1.05)
    tx = (size - (x1 - x0) * sx) / 2 - x0 * sx + rng.uniform(-4, 4)
    ty = (size - (y1 - y0) * sy) / 2 + y1 * sy + rng.uniform(-4, 4)
    angle = rng.uniform(-6, 6)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<g transform="rotate({angle} {size/2} {size/2})"><path fill="#000000" '
            f'transform="matrix({sx} 0 0 {-sy} {tx} {ty})" d="{pen.getCommands()}"/></g></svg>')


def degrade(alpha, rng, style):
    size = alpha.shape[0]
    bg = rng.uniform(180, 250, 3)
    ink = rng.uniform(5, 120, 3)
    if rng.random() < 0.2:
        bg, ink = ink, bg
    paper = np.ones((size, size, 3), np.float32) * bg
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if style in {"paper", "uneven"}:
        # 低频纸纹、明暗变化和染色斑；所有退化只影响观测，不改标签。
        texture = cv2.resize(rng.normal(0, 1, (8, 8)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
        paper += texture[:, :, None] * rng.uniform(8, 20)
        shade = (xx * rng.uniform(-40, 40) + yy * rng.uniform(-40, 40))
        for _ in range(5):
            cx, cy = rng.random(2)
            shade -= rng.uniform(5, 35) * np.exp(-((xx - cx)**2 + (yy - cy)**2) / rng.uniform(.005, .035))
        paper += shade[:, :, None]
    a = alpha[:, :, None].astype(np.float32) / 255
    rgb = paper * (1 - a) + ink * a
    if style != "clean":
        rgb += rng.normal(0, rng.uniform(1, 9), rgb.shape)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if style == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 85))])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        rgb = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=2800)
    parser.add_argument("--eval-count", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20261017)
    args = parser.parse_args()
    output = ROOT / "data/paired"
    if (output / "manifest.json").exists():
        raise FileExistsError("Dataset already exists; preserve it before generating a replacement.")
    cv2.setNumThreads(1)
    fonts = open_fonts()
    common = set.intersection(*(set(v[2]) for v in fonts.values()))
    chars = [chr(c) for c in sorted(common) if 0x4e00 <= c <= 0x9fff]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(chars)
    chars = chars[:1200]
    n = len(chars)
    char_splits = {"train": chars[:int(n*.7)], "val": chars[int(n*.7):int(n*.85)], "test": chars[int(n*.85):]}
    if min(map(len, char_splits.values())) < 20:
        raise RuntimeError("Not enough shared Chinese characters")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "size": 128, "source": "Google Fonts OFL 1.1",
                "heldout_fonts": sorted(HELDOUT), "char_splits": char_splits, "splits": {}}
    for split, count in [("train", args.train), ("val", args.eval_count), ("test", args.eval_count), ("unseen_font", args.eval_count)]:
        pool = char_splits["test" if split == "unseen_font" else split]
        families = sorted(f for f in fonts if (f in HELDOUT) == (split == "unseen_font"))
        images, masks, records, xx = [], [], [], []
        truth_dir = output / split / "truth_svg"
        truth_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            family = families[int(rng.integers(len(families)))]
            char = pool[int(rng.integers(len(pool)))]
            style = STYLES[i % len(STYLES)]
            svg = glyph_svg(*fonts[family], char, rng)
            alpha = render_svg(svg)[:, :, 3]
            if (alpha > 127).sum() < 50:
                raise RuntimeError(f"Empty or missing glyph: {family}/{char}")
            rgb = degrade(alpha, rng, style)
            (truth_dir / f"{i:05}.svg").write_text(svg, encoding="utf-8")
            images.append(rgb)
            masks.append(alpha)
            xx.append(features(rgb))
            records.append({"id": i, "family": family, "char": char, "style": style,
                            "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
            if i % 500 == 0:
                print(f"{split} {i}/{count}", flush=True)
        np.savez_compressed(output / f"{split}.npz", images=np.array(images), masks=np.array(masks),
                            features=np.array(xx, dtype=np.float16))
        (output / f"{split}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["splits"][split] = {"count": count, "fonts": families,
                                    "styles": dict(Counter(r["style"] for r in records))}
        if split == "test":
            sheet = Image.new("RGB", (128*6, 150*4), "white")
            draw = ImageDraw.Draw(sheet)
            for j in range(12):
                col, row = j % 3, j // 3
                sheet.paste(Image.fromarray(images[j]), (col*256, row*150))
                sheet.paste(Image.fromarray(masks[j]).convert("RGB"), (col*256+128, row*150))
                draw.text((col*256+4, row*150+130), records[j]["style"], fill="black")
            sheet.save(output / "dataset_preview.png")
    all_hashes = []
    for split in manifest["splits"]:
        hashes = {r["image_sha256"] for r in json.loads((output / f"{split}.json").read_text(encoding="utf-8"))}
        if any(hashes & prev for prev in all_hashes):
            raise RuntimeError("Exact image leakage between splits")
        all_hashes.append(hashes)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Dataset complete; disjoint characters, held-out families, no exact image overlap.", flush=True)


if __name__ == "__main__":
    main()
