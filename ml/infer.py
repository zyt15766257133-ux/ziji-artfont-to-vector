"""实验入口：图片 -> 已训练字形掩膜 -> 项目现有轮廓引擎 -> 单色 SVG。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline.direct import fit_mask_paths
from app.pipeline.svgutil import SvgDocument, SvgLayer, assign_glyph_groups
from .model import load_model, predict

ROOT = Path(__file__).resolve().parent


def mask_to_svg(alpha: np.ndarray, glyph_count: int = 1):
    h, w = alpha.shape
    # 复用现有像素边界三次曲线拟合，直接保留整图坐标和内孔。
    paths, nodes = fit_mask_paths((alpha > .5).astype(np.uint8), tolerance=.55)
    doc = SvgDocument(w, h, title="experimental-learned-monochrome-glyph")
    doc.layers.append(SvgLayer("#000000", paths, label="learned-glyph"))
    doc.glyph_groups = assign_glyph_groups(doc, glyph_count)
    return doc.to_svg(), {"engine": "learned-mask+existing-cubic-fit", "node_total": nodes,
                          "glyphs": glyph_count, "node_budget_passed": 50 <= nodes/glyph_count <= 500}


def main():
    parser = argparse.ArgumentParser(description="实验性单色字形模型；不支持保真恢复阴影、渐变和飞白。")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/baseline_v1/best.pt")
    parser.add_argument("--glyph-count", type=int, default=1)
    args = parser.parse_args()
    if args.glyph_count < 1:
        parser.error("--glyph-count must be positive")
    rgba = np.array(Image.open(args.input).convert("RGBA"))
    rgb = rgba[:, :, :3]
    t0 = time.perf_counter()
    trusted_alpha = float((rgba[:, :, 3] < 250).mean()) > .01
    if trusted_alpha:
        alpha = rgba[:, :, 3].astype(np.float32)/255
    else:
        model = load_model(args.checkpoint)
        alpha = predict(model, rgb)
    svg, stats = mask_to_svg(alpha, args.glyph_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    Image.fromarray((alpha*255).astype(np.uint8)).save(args.output.with_suffix(".mask.png"))
    Image.fromarray(np.dstack([rgb, (alpha*255).astype(np.uint8)])).save(args.output.with_suffix(".foreground.png"))
    result = {"input": str(args.input), "output": str(args.output), "seconds": time.perf_counter()-t0,
              "checkpoint": str(args.checkpoint), "used_model": not trusted_alpha,
              "scope": "experimental monochrome glyph foreground", "stats": stats}
    args.output.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
