"""命令行入口：单张转换 / 批量转换 / 样本集生成 / 验收测试。

用法：
  python cli.py convert 图片.png -o out --preset balanced --format svg
  python cli.py batch   samples/images -o out/batch --format pdf
  python cli.py samples -o out/samples            # 生成样本集转换产物与对比图
  python cli.py accept  -o out/acceptance         # 跑验收指标，出报告
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.pipeline.export import svg_to_eps, svg_to_pdf  # noqa: E402
from app.pipeline.metrics import evaluate, format_check  # noqa: E402
from app.pipeline.service import convert_bytes  # noqa: E402
from app.pipeline.vectorize import VectorizeOptions  # noqa: E402

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
WINDOWS_BAD = set('<>:"|?*')


def safe_stem(name: str, fallback: str = "output") -> str:
    stem = Path(name).stem
    cleaned = "".join(ch for ch in stem if ch not in WINDOWS_BAD).strip().strip(".")
    return cleaned or fallback


def build_options(args) -> VectorizeOptions:
    return VectorizeOptions.from_preset(
        getattr(args, "preset", "balanced"),
        engine=getattr(args, "engine", "auto"),
        tolerance=getattr(args, "tolerance", None),
        max_nodes=getattr(args, "max_nodes", None),
        max_colors=getattr(args, "max_colors", None),
        gradient_levels=getattr(args, "gradient_levels", None),
        soft_effects=not getattr(args, "no_soft_effects", False),
        glyph_count=getattr(args, "glyphs", None),
        spot_matching=bool(getattr(args, "spot", None)),
        spot_colors=([s.strip() for s in args.spot.split(",") if s.strip()]
                     if getattr(args, "spot", None) else None),
    )


def write_outputs(res, outdir: Path, fmt: str, want_preview: bool) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(res.filename)
    svg_path = outdir / f"{stem}.svg"
    svg_path.write_text(res.svg, encoding="utf-8")
    written = [svg_path.name]
    if fmt in ("pdf", "ai"):
        (outdir / f"{stem}.pdf").write_bytes(svg_to_pdf(res.svg))
        written.append(f"{stem}.pdf")
    if fmt == "eps":
        (outdir / f"{stem}.eps").write_bytes(svg_to_eps(res.svg))
        written.append(f"{stem}.eps")
    if want_preview and res.preview_png:
        (outdir / f"{stem}_compare.png").write_bytes(res.preview_png)
        written.append(f"{stem}_compare.png")
    (outdir / f"{stem}.json").write_text(
        json.dumps({"filename": res.filename, "elapsed_ms": round(res.elapsed_ms, 1),
                    "size": res.size, "quality": res.quality, "metrics": res.metrics,
                    "stats": res.stats, "warnings": res.warnings}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    written.append(f"{stem}.json")
    return {"stem": stem, "files": written, "metrics": res.metrics}


def cmd_convert(args) -> int:
    opts = build_options(args)
    src = Path(args.input)
    if not src.exists():
        print(f"找不到文件：{src}")
        return 1
    res = convert_bytes(src.name, src.read_bytes(), opts,
                        with_preview=not args.no_preview, strict_input=args.strict,
                        glyph_count=args.glyphs)
    if res.error:
        print(f"失败：{res.error}")
        print("素材提示：" + "；".join(res.warnings or ["无"]))
        return 2
    info = write_outputs(res, Path(args.out), args.format, not args.no_preview)
    m = res.metrics
    print(f"[{info['stem']}] 引擎 {m.get('engine')} 图层次数 {m.get('layer_count')} 耗时 {m['elapsed_ms']:.0f}ms")
    if m.get('source_fidelity'):
        f = m['source_fidelity']
        print(f"  原图主体平均误差 {f['active_mae']:.4f}/255，最大误差 {f['max_error']:.4f}/255")
        print('  表示方式：' + res.stats['representation'] + '；不承诺恢复任意放大无失真的平滑字形')
        print('  产出：' + ', '.join(info['files']))
        return 0
    print(f"  轮廓还原度 {m['contour_fidelity'] * 100:.1f}%  特征保留 {m['feature_retention'] * 100:.1f}%  "
          f"单字节点 {m['node_per_glyph']:.0f}  色偏 {m['color_deviation'] * 100:.1f}%")
    print("  判定：" + ("全部达标" if m["checks"]["passed"] else
                       "未达标项 " + ", ".join(k for k, v in m["checks"].items() if not v and k != "passed")))
    print("  产出：" + ", ".join(info["files"]))
    for w in res.warnings:
        print("  提示：" + w)
    return 0


def cmd_batch(args) -> int:
    opts = build_options(args)
    src = Path(args.input)
    files = sorted(p for p in (src.iterdir() if src.is_dir() else [src])
                   if p.suffix.lower() in IMAGE_EXT)
    if not files:
        print("没有找到图片")
        return 1
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ok = 0
    rows = []
    for path in files:
        res = convert_bytes(path.name, path.read_bytes(), opts,
                            with_preview=not args.no_preview, strict_input=False)
        if res.error:
            rows.append({"file": path.name, "error": res.error})
            print(f"  [失败] {path.name}: {res.error}")
            continue
        info = write_outputs(res, outdir, args.format, not args.no_preview)
        ok += 1
        m = res.metrics
        rows.append({"file": path.name, "elapsed_ms": round(res.elapsed_ms, 1),
                     "fidelity": m.get("contour_fidelity"), "feature": m.get("feature_retention"),
                     "source_fidelity": m.get("source_fidelity"),
                     "node_per_glyph": m.get("node_per_glyph"), "color_deviation": m.get("color_deviation"),
                     "passed": m["checks"]["passed"]})
    total = (time.perf_counter() - t0) * 1000
    print(f"批量完成 {ok}/{len(files)}，总耗时 {total / 1000:.1f}s，平均 {total / max(1, ok):.0f}ms/张")
    (outdir / "_batch_summary.json").write_text(
        json.dumps({"count": len(files), "succeeded": ok, "total_ms": round(total, 1),
                    "avg_ms": round(total / max(1, ok), 1), "items": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_samples(args) -> int:
    src = Path(args.input) if args.input else ROOT / "samples" / "images"
    outdir = Path(args.out) if args.out else ROOT / "out" / "samples"
    opts = build_options(args)
    files = sorted(p for p in Path(src).glob("*.png"))
    if not files:
        print("样本目录为空，先运行 python samples/generate_samples.py")
        return 1
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    glyph_map: dict[str, int] = {}
    mf = ROOT / "samples" / "manifest.json"
    if mf.exists():
        for item in json.loads(mf.read_text(encoding="utf-8")):
            g = item.get("glyphs")
            if not isinstance(g, int):
                text = item.get("text") or ""
                g = max(1, sum(1 for ch in text if not ch.isspace() and ch != "/"))
            glyph_map[item["file"]] = g
    ok = 0
    for path in files:
        res = convert_bytes(path.name, path.read_bytes(), opts, with_preview=True,
                            strict_input=False, glyph_count=glyph_map.get(path.name))
        if res.error:
            print(f"  [失败] {path.name}: {res.error}")
            continue
        info = write_outputs(res, outdir, args.format, True)
        m = res.metrics
        ok += 1
        manifest.append({
            "sample": path.name, "svg": f"{info['stem']}.svg",
            "compare": f"{info['stem']}_compare.png", "engine": m.get("engine"),
            "fidelity": m["contour_fidelity"], "feature": m["feature_retention"],
            "node_per_glyph": m["node_per_glyph"], "color_deviation": m["color_deviation"],
            "color_levels": m["color_levels_out"], "elapsed_ms": m["elapsed_ms"],
            "passed": m["checks"]["passed"],
        })
        print(f"  {path.name:24s} 还原度 {m['contour_fidelity'] * 100:5.1f}%  "
              f"特征 {m['feature_retention'] * 100:5.1f}%  节点/字 {m['node_per_glyph']:5.0f}  "
              f"{'达标' if m['checks']['passed'] else '未达标'}")
    (outdir / "_samples_summary.json").write_text(
        json.dumps({"count": len(files), "succeeded": ok, "items": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"样本集转换完成 {ok}/{len(files)} → {outdir}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="艺术字转矢量图 命令行工具")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--preset", default="balanced", choices=list(VectorizeOptions.PRESETS))
    common.add_argument("--engine", default="auto", choices=["auto", "contour", "vtracer"])
    common.add_argument("--tolerance", type=float, default=None)
    common.add_argument("--max-nodes", type=int, default=None)
    common.add_argument("--max-colors", type=int, default=None)
    common.add_argument("--gradient-levels", type=int, default=None)
    common.add_argument("--glyphs", type=int, default=None,
                        help="真实字数（默认连通域估算，建议手填或用 manifest）")
    common.add_argument("--spot", default=None,
                        help="启用专色匹配：HEX 色板，英文逗号分隔；纯色层匹配最近色，渐变保持原样")
    common.add_argument("--format", default="svg", choices=["svg", "pdf", "eps", "ai"])
    common.add_argument("--no-preview", action="store_true")
    common.add_argument("--no-soft-effects", action="store_true", help="关闭保真档柔光补偿，便于同图对照")

    p1 = sub.add_parser("convert", parents=[common], help="单张转换")
    p1.add_argument("input")
    p1.add_argument("-o", "--out", default="out")
    p1.add_argument("--strict", action="store_true", help="严格校验输入素材")
    p1.set_defaults(func=cmd_convert)

    p2 = sub.add_parser("batch", parents=[common], help="批量转换")
    p2.add_argument("input")
    p2.add_argument("-o", "--out", default="out/batch")
    p2.set_defaults(func=cmd_batch)

    p3 = sub.add_parser("samples", parents=[common], help="转换整个样本集")
    p3.add_argument("input", nargs="?")
    p3.add_argument("-o", "--out")
    p3.set_defaults(func=cmd_samples)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
