"""转换服务层：输入校验、单张转换、批量转换与打包。"""
from __future__ import annotations

import base64
import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .assessment import assess_result
from .export import minify_svg, svg_to_eps, svg_to_pdf
from .metrics import evaluate, format_check
from .preprocess import preprocess
from .preview import build_comparison, encode_png
from .svgutil import SvgDocument
from .vectorize import VectorizeOptions, vectorize

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
MAX_BYTES = 5 * 1024 * 1024
MIN_SIDE = 500


@dataclass
class ConvertResult:
    filename: str
    stem: str
    svg: str
    metrics: dict
    quality: dict
    stats: dict
    size: dict
    elapsed_ms: float
    warnings: list[str]
    preview_png: bytes | None = None
    source_png: bytes | None = None   # 裁切对齐后的原图（与矢量同坐标系，供对比滑块）
    error: str | None = None
    timings: dict | None = None

    def to_dict(self, with_preview: bool = True) -> dict:
        data = {
            "filename": self.filename,
            "stem": self.stem,
            "svg": self.svg,
            "metrics": self.metrics,
            "quality": self.quality,
            "stats": self.stats,
            "size": self.size,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "timings": self.timings,
            "warnings": self.warnings,
            "error": self.error,
        }
        if with_preview and self.preview_png:
            data["preview_png"] = base64.b64encode(self.preview_png).decode("ascii")
        if with_preview and self.source_png:
            data["source_png"] = base64.b64encode(self.source_png).decode("ascii")
        return data


def validate_input(filename: str, data: bytes) -> list[str]:
    problems: list[str] = []
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        problems.append(f"格式不支持：{ext or '未知'}（支持 PNG/JPG/WEBP）")
    if len(data) > MAX_BYTES:
        problems.append(f"文件超过 5MB（当前 {len(data) / 1048576:.1f}MB）")
    return problems


def convert_bytes(filename: str, data: bytes, options: VectorizeOptions | None = None,
                  with_preview: bool = True, strict_input: bool = True,
                  glyph_count: int | None = None,
                  glyph_text: str | None = None,
                  truth_mask: "np.ndarray | None" = None,
                  progress: "Callable[[str], None] | None" = None) -> ConvertResult:
    """单张转换主入口。耗时统计覆盖“矢量化 + 指标 + 预览”全流程。

    glyph_count：真实字数（manifest/用户手填），参与节点预算与单字节点指标；
    glyph_text：原图文字（manifest），用于按字分组的语义命名；
    truth_mask：与输入原图同尺寸的真值字形掩膜，传入时指标额外输出对真值口径。
    """
    opts = (options or VectorizeOptions()).resolve_spot_options()
    overrides = {k: v for k, v in (("glyph_count", glyph_count), ("glyph_text", glyph_text))
                 if v is not None}
    if overrides:
        from dataclasses import replace
        opts = replace(opts, **overrides)
    if not np.isfinite(opts.tolerance) or opts.tolerance <= 0 or not 50 <= opts.max_nodes <= 500:
        raise ValueError('容差必须为正数，单字节点上限须在 50～500 之间')
    if not 1 <= opts.max_colors <= 32 or not 8 <= opts.gradient_levels <= 64:
        raise ValueError('颜色层须在 1～32 之间，渐变色阶须在 8～64 之间')
    if not 0 <= opts.filter_speckle <= 100 or not 1 <= opts.corner_angle <= 179:
        raise ValueError('杂点过滤须在 0～100 之间，角点角须在 1～179 之间')
    if opts.glyph_count is not None and (isinstance(opts.glyph_count, bool) or not isinstance(opts.glyph_count, int) or opts.glyph_count < 1):
        raise ValueError('真实字数必须为正整数')
    stem = Path(filename.replace('\\', '/')).stem or "image"
    t0 = time.perf_counter()

    def _tick(stage: str) -> None:
        if progress is not None:
            progress(stage)

    problems = validate_input(filename, data)
    if problems:
        return ConvertResult(filename, stem, "", {}, {}, {}, {}, 0.0, problems,
                             error="；".join(problems))

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED) if arr.size else None
    if img is None:
        return ConvertResult(filename, stem, "", {}, {}, {}, {}, 0.0,
                             ["无法解码图片"], error="无法解码图片，请确认文件完整")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    warnings: list[str] = []
    h, w = img.shape[:2]
    if strict_input and min(h, w) < MIN_SIDE:
        msg = f"分辨率 {w}×{h} 低于 500×500，请更换清晰素材"
        return ConvertResult(filename, stem, "", {}, {}, {}, {},
                             (time.perf_counter() - t0) * 1000, [msg], error=msg)

    _tick("preprocess")
    if opts.mode in {'fidelity', 'pixel_exact'}:
        from .fidelity import convert_source
        result = convert_source(filename, img, opts, with_preview, progress)
        elapsed = (time.perf_counter() - t0) * 1000
        result.timings['decode_validate_ms'] = elapsed - result.elapsed_ms
        result.timings['total_ms'] = elapsed
        result.timings['scope'] = 'decode validation trace render verification preview included'
        result.elapsed_ms = elapsed
        result.metrics['elapsed_ms'] = elapsed
        return result
    t_preprocess = time.perf_counter()
    prepared = preprocess(img)
    from .color_analysis import source_colors
    palette = source_colors(prepared.rgb_raw, np.minimum(prepared.source_rgba[:, :, 3], prepared.mask))
    preprocess_ms = (time.perf_counter() - t_preprocess) * 1000
    q = prepared.quality
    hard_reject = q.severity == "reject" or (strict_input and q.strict_reject)
    if hard_reject:
        msg = "；".join(q.warnings) or "素材不满足输入规范"
        return ConvertResult(filename, stem, "", {}, q.to_dict(), {}, {},
                             (time.perf_counter() - t0) * 1000,
                             q.warnings, error=f"素材异常：{msg}")
    if q.invalid and not hard_reject:
        # 需求 5.6：默认模式判出异常但不阻断，给出明确提示与结论
        reason_map = {
            "background_complex": "背景复杂杂乱",
            "blurry": "图像严重模糊",
            "glare": "大面积强反光/过曝",
            "no_foreground": "未识别到前景内容",
            "subject_small": "主体占画面低于 60%",
            "photo_like": "疑似照片/写实图，非艺术字素材",
        }
        names = "、".join(reason_map.get(r, r) for r in q.invalid_reasons)
        warnings.append(f"素材异常判定：{names}。默认模式继续转换，但建议更换更干净清晰的素材")
    _tick("vectorize")
    t_vectorize = time.perf_counter()
    doc, stats = vectorize(prepared, opts)
    doc.title = f"{stem} vectorized"
    svg = doc.to_svg()
    vectorize_ms = (time.perf_counter() - t_vectorize) * 1000.0

    # 栅格化只做一次，指标与对比图共用，避免重复渲染
    _tick("verify")
    t_verify = time.perf_counter()
    from .pathparse import SvgRaster
    raster = SvgRaster(svg).rasterize()
    metrics = evaluate(prepared, svg, raster=raster,
                       glyph_count=opts.glyph_count, truth_mask=truth_mask,
                       tolerance=opts.tolerance)
    metrics["format"] = format_check(svg)
    metrics['source_colors'] = palette
    metrics["input_invalid"] = q.invalid
    metrics["input_invalid_reasons"] = list(q.invalid_reasons)
    metrics["engine"] = stats.get("engine_used", stats.get("engine"))
    metrics["layer_count"] = len(doc.layers)

    metrics["checks_scope"] = "internal_diagnostics"
    metrics["source_transform"] = {
        "input_shape": prepared.input_shape,
        "canvas_shape": prepared.canvas_shape,
        "rotation": prepared.rot_matrix.tolist() if prepared.rot_matrix is not None else None,
        "crop_box": prepared.crop_box,
    }
    verify_ms = (time.perf_counter() - t_verify) * 1000
    t_preview = time.perf_counter()
    preview = build_comparison(prepared, svg, title=stem,
                               raster=raster) if with_preview else None

    # 对比滑块使用仅经过几何对齐的原始颜色和透明度。
    source_png = None
    if with_preview:
        ok, buf = cv2.imencode(".png", prepared.source_rgba[:, :, [2, 1, 0, 3]])
        if ok:
            source_png = buf.tobytes()

    preview_png = encode_png(preview) if preview is not None else None
    preview_ms = (time.perf_counter() - t_preview) * 1000
    warnings.extend(q.warnings)
    if stats.get('spot_matching', {}).get('enabled'):
        warnings.append(f"已启用专色匹配：改写 {stats['spot_matching']['matched_layers']} 个纯色图层；渐变保持原样。")
    metrics["assessment"] = assess_result(metrics, q.to_dict(), None, (w, h))
    elapsed = (time.perf_counter() - t0) * 1000
    metrics["elapsed_ms"] = round(elapsed, 1)
    assessment = metrics["assessment"]
    timing_item = assessment["items"]["single_time"]
    timing_item["value"] = elapsed
    if max(w, h) <= 1000:
        timing_item["status"] = "pass" if elapsed <= 5000 else "fail"
    statuses = [item["status"] for item in assessment["items"].values()]
    assessment["status"] = "fail" if "fail" in statuses else "untested" if "untested" in statuses else "pass"
    timings = {"preprocess_ms": round(preprocess_ms, 1), "vectorize_ms": round(vectorize_ms, 1),
               "verify_ms": round(verify_ms, 1), "preview_ms": round(preview_ms, 1),
               "total_ms": round(elapsed, 1),
               "scope": "decode_to_response_materials; excludes HTTP serialization and format download"}
    return ConvertResult(
        filename=filename, stem=stem, svg=svg, metrics=metrics, quality=q.to_dict(),
        stats=stats, size={"width": doc.width, "height": doc.height},
        elapsed_ms=elapsed, warnings=warnings,
        preview_png=preview_png,
        timings=timings,
        source_png=source_png,
    )


def inspect_bytes(filename: str, data: bytes) -> dict:
    """素材预检：不做矢量化，只给输入质量结论，供前端上传后立即展示。"""
    stem = Path(filename).stem or "image"
    out = {"filename": filename, "stem": stem, "problems": [], "warnings": [],
           "quality": {}, "size": None}
    problems = validate_input(filename, data)
    if problems:
        out["problems"] = problems
        return out
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED) if arr.size else None
    if img is None:
        out['problems'] = ["无法解码图片，请确认文件完整"]
        return out
    h, w = img.shape[:2]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    out["size"] = {"width": int(w), "height": int(h), "bytes": len(data)}
    if min(h, w) < MIN_SIDE:
        out["problems"].append(f"分辨率 {w}×{h} 低于 500×500，请更换清晰素材")
    from .background import remove_uniform_background
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA if img.shape[2] == 4 else cv2.COLOR_BGR2RGBA)
    foreground, out['background_removal'] = remove_uniform_background(rgba)
    # Inspect the same foreground used by the fidelity conversion. The legacy
    # border estimator otherwise reads screenshot frames as the background and
    # can report zero lettering despite successful spatial paper extraction.
    if out['background_removal']['applied'] or out['background_removal']['reason'] == 'existing_transparency':
        prepared = preprocess(cv2.cvtColor(foreground, cv2.COLOR_RGBA2BGRA),
                              deskew=False, denoise=False, enhance=False,
                              max_side=max(h, w))
        out['quality_basis'] = 'detected_foreground_native_resolution'
    else:
        prepared = preprocess(img)
        out['quality_basis'] = 'legacy_input_diagnostics'
    q = prepared.quality
    out["quality"] = q.to_dict()
    out["warnings"].extend(q.warnings)
    return out


def convert_to_format(res: ConvertResult, fmt: str) -> tuple[str, bytes]:
    """按目标格式导出，返回 (文件名, 内容)。"""
    fmt = (fmt or "svg").lower()
    if fmt == "svg":
        return f"{res.stem}.svg", res.svg.encode("utf-8")
    if fmt in ("svgz", "min"):
        return f"{res.stem}.min.svg", minify_svg(res.svg).encode("utf-8")
    if fmt == "pdf":
        return f"{res.stem}.pdf", svg_to_pdf(res.svg)
    if fmt == "eps":
        return f"{res.stem}.eps", svg_to_eps(res.svg)
    if fmt == "ai":
        # PDF 兼容的 .ai：Illustrator 可直接打开并编辑，另存即原生 .ai
        return f"{res.stem}.ai", svg_to_pdf(res.svg)
    raise ValueError(f"不支持的导出格式：{fmt}")


def batch_convert(files: list[tuple[str, bytes]], options: VectorizeOptions | None = None,
                  fmt: str = "svg", with_report: bool = True,
                  workers: int = 3, strict_input: bool = False) -> tuple[bytes, list[ConvertResult]]:
    """Convert a fixed batch and archive unique, traceable output names."""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    import json
    import re
    started = time.perf_counter()
    from copy import deepcopy
    if not 1 <= len(files) <= 100:
        raise ValueError('每批须有 1～100 张图片')
    if fmt not in {'svg','pdf','eps','ai'}:
        raise ValueError('不支持的导出格式')
    snapshot = deepcopy(options or VectorizeOptions())
    def one(name, data):
        try:
            return convert_bytes(name, data, snapshot, False, strict_input)
        except Exception as exc:
            return ConvertResult(name, Path(name).stem, '', {}, {}, {}, {}, 0, [], error=str(exc))
    with ThreadPoolExecutor(max_workers=max(1, min(3, int(workers)))) as pool:
        results = list(pool.map(lambda item: one(*item), files))
    buf = io.BytesIO()
    used = set()
    mapping = []
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        for index, res in enumerate(results, 1):
            stem = re.sub(r'[<>:"/\\\\|?*\x00-\x1f]', '_', res.stem).strip('. ') or 'image'
            if stem.casefold().startswith('_batch'):
                stem = 'image' + stem
            unique = stem
            suffix = 2
            while unique.casefold() in used:
                unique = f'{stem}__{suffix}'
                suffix += 1
            used.add(unique.casefold())
            entry = {'index':index,'source':res.filename,'output_stem':unique,'error':res.error}
            mapping.append(entry)
            if res.error:
                zf.writestr(f'{unique}.error.txt', res.error)
                continue
            try:
                out_name, payload = convert_to_format(res, fmt)
                extension = out_name[len(res.stem):]
                zf.writestr(unique + extension, payload)
                entry['output_file'] = unique + extension
            except Exception as exc:
                res.error = f'导出失败：{exc}'
                entry['error'] = res.error
                zf.writestr(f'{unique}.error.txt', res.error)
            if with_report:
                report = json.loads(_result_json(res))
                report['options'] = snapshot.to_dict()
                report['output_file'] = entry.get('output_file')
                zf.writestr(f'{unique}.json', json.dumps(report, ensure_ascii=False, indent=2))
        elapsed = (time.perf_counter() - started) * 1000
        summary = json.loads(_summary_json(results))
        summary.update(wall_ms=round(elapsed,1),options=snapshot.to_dict(),files=mapping)
        zf.writestr('_batch_summary.json', json.dumps(summary,ensure_ascii=False,indent=2))
    return buf.getvalue(), results


def _result_json(res: ConvertResult) -> str:
    import json
    return json.dumps({
        "filename": res.filename,
        "elapsed_ms": round(res.elapsed_ms, 1),
        "size": res.size,
        "quality": res.quality,
        "metrics": res.metrics,
        "warnings": res.warnings,
    }, ensure_ascii=False, indent=2)


def _summary_json(results: list[ConvertResult]) -> str:
    import json
    ok = [r for r in results if not r.error]
    def mean(key: str) -> float:
        vals = [r.metrics.get(key, 0) for r in ok if isinstance(r.metrics.get(key), (int, float))]
        return round(float(sum(vals)) / len(vals), 4) if vals else 0.0
    summary = {
        "count": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "total_elapsed_ms": round(sum(r.elapsed_ms for r in results), 1),
        "avg_elapsed_ms": round(sum(r.elapsed_ms for r in ok) / len(ok), 1) if ok else 0,
        "avg_contour_fidelity": mean("contour_fidelity"),
        "avg_feature_retention": mean("feature_retention"),
        "avg_node_per_glyph": mean("node_per_glyph"),
        "avg_color_deviation": mean("color_deviation"),
        "engines": sorted({r.metrics.get("engine", "") for r in ok if r.metrics}),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)
