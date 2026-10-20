"""艺术字转矢量图 Demo 服务（FastAPI）。

启动：python -m app.main   或   uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .jobs import JobManager, JobQueueFull

from .pipeline.export import svg_to_eps, svg_to_pdf
from .pipeline.service import batch_convert, convert_bytes, convert_to_format, inspect_bytes
from .pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
# 素材目录可配置（默认 samples/images）：ZIJI_SAMPLES_DIR 指向其他文件夹即可
BUILD_SHA = hashlib.sha256(b''.join(path.read_bytes() for path in sorted((ROOT/'app').rglob('*.py')))).hexdigest()[:12]
SAMPLES = Path(os.environ.get("ZIJI_SAMPLES_DIR") or (ROOT / "samples" / "images"))

@asynccontextmanager
async def lifespan(application):
    _warm_sample_cache()
    try:
        yield
    finally:
        await run_in_threadpool(_JOB_MANAGER.shutdown)
        from .pipeline.native_render import shutdown
        shutdown()


app = FastAPI(title="艺术字转矢量图 Demo", version="1.2.0-dev", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
EVIDENCE = Path(os.environ['ZIJI_EVIDENCE_DIR']) if os.environ.get('ZIJI_EVIDENCE_DIR') else None
if EVIDENCE is not None and EVIDENCE.is_dir():
    app.mount('/evidence', StaticFiles(directory=str(EVIDENCE)), name='evidence')
if SAMPLES.exists():
    app.mount("/samples", StaticFiles(directory=str(SAMPLES)), name="samples")
FIDELITY_RESULTS = ROOT / 'eval' / 'results' / 'full_folder_fidelity'
if FIDELITY_RESULTS.is_dir():
    app.mount('/fidelity-results', StaticFiles(directory=str(FIDELITY_RESULTS)), name='fidelity-results')
SOFT_EFFECT_RESULTS = ROOT / 'eval' / 'results' / 'soft_effects'
if SOFT_EFFECT_RESULTS.is_dir():
    app.mount('/soft-effects-results', StaticFiles(directory=str(SOFT_EFFECT_RESULTS)), name='soft-effects-results')

# 内置样本转换结果缓存：同一参数重复点示例时直接命中，避免每次等几秒
_SAMPLE_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_SAMPLE_CACHE_MAX = 48
_CACHE_LOCK = threading.RLock()


def _cache_get(key: str) -> dict | None:
    with _CACHE_LOCK:
        payload = _SAMPLE_CACHE.get(key)
        if payload is None:
            return None
        _SAMPLE_CACHE.move_to_end(key)
        return payload


def _cache_put(key: str, payload: dict) -> None:
    with _CACHE_LOCK:
        _SAMPLE_CACHE[key] = payload
        _SAMPLE_CACHE.move_to_end(key)
        while len(_SAMPLE_CACHE) > _SAMPLE_CACHE_MAX:
            _SAMPLE_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def build_options(
    preset: str = "balanced",
    engine: str = "auto",
    tolerance: float | None = None,
    max_nodes: int | None = None,
    max_colors: int | None = None,
    gradient_levels: int | None = None,
    filter_speckle: int | None = None,
    corner_angle: float | None = None,
    smooth: bool | None = None,
    auto_verify: bool | None = None,
    spot_matching: bool | None = None,
    spot_colors: str | list[str] | None = None,
    soft_effects: bool | None = None,
) -> VectorizeOptions:
    if preset not in VectorizeOptions.PRESETS or engine not in {'auto','contour','vtracer'}:
        raise HTTPException(status_code=422, detail='未知预设或引擎')
    if isinstance(spot_colors, str):
        spot_colors = [s.strip() for s in spot_colors.split(",") if s.strip()] or None
    if preset == 'pixel_exact':
        spot_matching = False
        spot_colors = None
    if not spot_matching:
        spot_colors = None
    opts = VectorizeOptions.from_preset(
        preset, engine=engine, tolerance=tolerance, max_nodes=max_nodes,
        max_colors=max_colors, gradient_levels=gradient_levels,
        filter_speckle=filter_speckle, corner_angle=corner_angle,
        smooth=smooth, auto_verify=auto_verify, spot_matching=bool(spot_matching),
        spot_colors=spot_colors, soft_effects=soft_effects,
    )
    opts.engine = engine or 'auto'
    try:
        opts = opts.resolve_spot_options()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    import math
    if not math.isfinite(opts.tolerance) or opts.tolerance <= 0 or not 50 <= opts.max_nodes <= 500:
        raise HTTPException(status_code=422, detail='容差须为正数，节点上限须在 50～500 之间')
    if not 1 <= opts.max_colors <= 32 or not 8 <= opts.gradient_levels <= 64:
        raise HTTPException(status_code=422, detail='颜色层须在 1～32，渐变色阶须在 8～64 之间')
    if not 0 <= opts.filter_speckle <= 100 or not 1 <= opts.corner_angle <= 179:
        raise HTTPException(status_code=422, detail='杂点过滤须在 0～100，角点角须在 1～179 之间')
    return opts


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get('/validation', response_class=HTMLResponse)
def validation_page():
    return HTMLResponse((STATIC/'validation.html').read_text(encoding='utf-8'))


@app.get('/api/validation')
def validation_report():
    path = EVIDENCE/'release_report.json' if EVIDENCE is not None else None
    if path is None or not path.is_file():
        return {'build':'验证正在进行','status':'pending'}
    return json.loads(path.read_text(encoding='utf-8'))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "time": time.time(), "version": app.version,
            "build": BUILD_SHA, "pid": os.getpid(), "sample_cache": len(_SAMPLE_CACHE)}


WARMUP_SAMPLES = [
    "01_hei_solid.png", "02_song_serif.png", "03_kai.png", "04_xingkai.png",
    "05_lishu.png", "06_youyuan.png", "07_shuti_hand.png", "08_thin_line.png",
    "09_stroke.png", "10_gradient.png", "11_3d.png", "12_shadow.png",
    "13_cartoon.png", "14_brush_feibai.png", "15_lianbi.png", "16_gradient_stroke.png",
    "17_inverse.png", "18_transparent.png", "19_multiline.png", "20_metal.png",
    "21_neon.png", "23_holes_dense.png", "24_standard_53.png",
    "25_radial.png",
]


def _manifest_meta(filename: str) -> tuple[int | None, str | None]:
    """内置样本文本字数与原文（manifest），无记录时返回 (None, None)。"""
    mf = ROOT / "samples" / "manifest.json"
    if not mf.exists():
        return None, None
    try:
        for item in json.loads(mf.read_text(encoding="utf-8")):
            if item.get("file") == filename:
                text = item.get("text") or ""
                g = item.get("glyphs")
                if not isinstance(g, int):
                    g = max(1, sum(1 for ch in text if not ch.isspace() and ch != "/")) or None
                return g, (text or None)
    except Exception:
        return None, None
    return None, None


def _manifest_glyphs(filename: str) -> int | None:
    return _manifest_meta(filename)[0]


@app.get("/api/presets")
def presets() -> dict:
    p = VectorizeOptions.PRESETS
    return {
        "presets": [
            {"key": "fidelity", "name": "保真曲线",
             "desc": "自动去除纯色及浅色纸底；复杂背景会提示去底失败，适用素材保留可编辑柔光"},
            {"key": "pixel_exact", "name": "原像素路径",
             "desc": "用矢量矩形保存原像素；文件可能很大，放大仍保留像素网格"},
            {"key": "quality", "name": "高质量",
             "desc": f"容差 {p['quality']['tolerance']}px，颜色层 {p['quality']['max_colors']}，节点更多、轮廓更准"},
            {"key": "balanced", "name": "均衡", "desc": "默认档，兼顾还原度与节点精简"},
            {"key": "compact", "name": "精简",
             "desc": f"容差 {p['compact']['tolerance']}px，颜色层 {p['compact']['max_colors']}，文件最小、便于印刷刻字"},
        ],
        "engines": [
            {"key": "auto", "name": "自动", "desc": "自研轮廓引擎为主，不达标自动与 vtracer 对比取优"},
            {"key": "contour", "name": "轮廓引擎", "desc": "自研：主色分层 + 角点保护曲线拟合，节点可控"},
            {"key": "vtracer", "name": "vtracer", "desc": "外部引擎：适合复杂多色与柔和过渡图像"},
        ],
        "formats": [
            {"key": "svg", "name": "SVG"},
            {"key": "pdf", "name": "PDF (Illustrator 可编辑)"},
            {"key": "eps", "name": "EPS"},
            {"key": "ai", "name": "AI (PDF 兼容，需 Illustrator 另存原生)"},
        ],
        "defaults": VectorizeOptions().to_dict(),
    }


@app.get("/api/samples")
def list_samples() -> dict:
    if not SAMPLES.exists():
        return {"samples": []}
    manifest = {}
    mf = ROOT / "samples" / "manifest.json"
    if mf.exists():
        for item in json.loads(mf.read_text(encoding="utf-8")):
            manifest[item["file"]] = item
    out = []
    for path in sorted(SAMPLES.glob("*.png")):
        meta = manifest.get(path.name, {})
        out.append({
            "file": path.name,
            "url": f"/samples/{path.name}",
            "style": meta.get("style", ""),
            "desc": meta.get("desc", ""),
            "text": meta.get("text", ""),
        })
    return {"samples": out}


async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    data = await file.read(5 * 1024 * 1024 + 1)
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='单张图片超过 5MB')
    return file.filename or "upload.png", data


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)) -> JSONResponse:
    """素材预检：只跑输入质量判定，不矢量化。"""
    name, data = await _read_upload(file)
    return JSONResponse(await run_in_threadpool(inspect_bytes, name, data))


_JOB_MANAGER = JobManager(workers=2, capacity=10)


def _sample_cache_key(name: str, opts: VectorizeOptions, strict_input: bool,
                      with_preview: bool = True, glyph_count: int | None = None) -> str:
    import hashlib
    signature = {'name':name, 'options':opts.to_dict(), 'strict_input':strict_input,
                 'with_preview':with_preview, 'glyph_count':glyph_count}
    path = SAMPLES / Path(name).name
    if path.exists():
        signature['content_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    preset: str = Form("balanced"),
    engine: str = Form("auto"),
    tolerance: float | None = Form(None),
    max_nodes: int | None = Form(None),
    max_colors: int | None = Form(None),
    gradient_levels: int | None = Form(None),
    filter_speckle: int | None = Form(None),
    corner_angle: float | None = Form(None),
    smooth: bool | None = Form(None),
    auto_verify: bool | None = Form(None),
    soft_effects: bool = Form(True),
    strict_input: bool = Form(False),
    with_preview: bool = Form(True),
    glyph_count: int | None = Form(None),
    spot_matching: bool = Form(False),
    spot_colors: str | None = Form(None),
) -> JSONResponse:
    name, data = await _read_upload(file)
    opts = build_options(preset, engine, tolerance, max_nodes, max_colors,
                         gradient_levels, filter_speckle, corner_angle, smooth,
                         auto_verify, spot_matching=spot_matching, spot_colors=spot_colors, soft_effects=soft_effects)
    m_glyphs, m_text = _manifest_meta(name)
    def operation(progress):
        result = convert_bytes(name, data, opts, with_preview=with_preview,
                               strict_input=strict_input,
                               glyph_count=glyph_count or m_glyphs,
                               glyph_text=m_text, progress=progress)
        payload = result.to_dict(with_preview=with_preview)
        payload['options'] = opts.to_dict()
        return payload

    try:
        job_id = _JOB_MANAGER.submit(operation)
    except JobQueueFull as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return JSONResponse({'job_id':job_id})


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    try:
        return _JOB_MANAGER.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='任务不存在') from exc


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    try:
        return _JOB_MANAGER.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='任务不存在') from exc


@app.post("/api/convert")
async def convert(
    file: UploadFile = File(...),
    preset: str = Form("balanced"),
    engine: str = Form("auto"),
    tolerance: float | None = Form(None),
    max_nodes: int | None = Form(None),
    max_colors: int | None = Form(None),
    gradient_levels: int | None = Form(None),
    filter_speckle: int | None = Form(None),
    corner_angle: float | None = Form(None),
    smooth: bool | None = Form(None),
    auto_verify: bool | None = Form(None),
    soft_effects: bool = Form(True),
    strict_input: bool = Form(False),
    with_preview: bool = Form(True),
    glyph_count: int | None = Form(None),
    spot_matching: bool = Form(False),
    spot_colors: str | None = Form(None),
) -> JSONResponse:
    name, data = await _read_upload(file)
    opts = build_options(preset, engine, tolerance, max_nodes, max_colors,
                         gradient_levels, filter_speckle, corner_angle, smooth,
                         auto_verify, spot_matching=spot_matching, spot_colors=spot_colors, soft_effects=soft_effects)
    m_glyphs, m_text = _manifest_meta(name)
    try:
        result = await run_in_threadpool(convert_bytes, name, data, opts, with_preview=with_preview,
                                        strict_input=strict_input,
                                        glyph_count=glyph_count or m_glyphs,
                                        glyph_text=m_text)
    except Exception as exc:  # 兜底：保证前端拿到结构化错误
        return JSONResponse({"error": f"转换失败：{exc}"}, status_code=500)
    payload = result.to_dict(with_preview=with_preview)
    payload["options"] = opts.to_dict()
    return JSONResponse(payload)


@app.post("/api/batch")
async def batch(
    files: list[UploadFile] = File(...),
    preset: str = Form("balanced"),
    engine: str = Form("auto"),
    tolerance: float | None = Form(None),
    max_nodes: int | None = Form(None),
    max_colors: int | None = Form(None),
    gradient_levels: int | None = Form(None),
    filter_speckle: int | None = Form(None),
    corner_angle: float | None = Form(None),
    fmt: str = Form('svg'),
    strict_input: bool = Form(False),
    smooth: bool | None = Form(None),
    auto_verify: bool | None = Form(None),
    soft_effects: bool = Form(True),
    spot_matching: bool = Form(False),
    spot_colors: str | None = Form(None),
    glyph_count: int | None = Form(None),
) -> Response:
    if fmt not in {'svg','pdf','eps','ai'}:
        raise HTTPException(status_code=422, detail='不支持的导出格式')
    if len(files) > 100:
        raise HTTPException(status_code=400, detail='每批最多 100 张图片')
    payloads: list[tuple[str, bytes]] = []
    for f in files:
        name, data = await _read_upload(f)
        payloads.append((name, data))
    if not payloads:
        raise HTTPException(status_code=400, detail="没有收到文件")
    opts = build_options(preset, engine, tolerance, max_nodes, max_colors,
                         gradient_levels, filter_speckle, corner_angle, smooth,
                         auto_verify, spot_matching=spot_matching, spot_colors=spot_colors, soft_effects=soft_effects)
    opts.glyph_count = glyph_count
    started = time.perf_counter()
    zip_bytes, results = await run_in_threadpool(batch_convert, payloads, opts, fmt=fmt, strict_input=strict_input)
    wall_ms = (time.perf_counter() - started) * 1000
    summary = {
        "count": len(results),
        "succeeded": sum(1 for r in results if not r.error),
        "elapsed_ms": round(wall_ms, 1),
        "sum_service_ms": round(sum(r.elapsed_ms for r in results), 1),
        "items": [{
            "filename": r.filename,
            "error": r.error,
            "elapsed_ms": round(r.elapsed_ms, 1),
            "metrics": r.metrics,
        } for r in results],
    }
    headers = {
        "Access-Control-Expose-Headers": "X-Batch-Summary",
    }
    headers['X-Batch-Summary'] = base64.b64encode(json.dumps({k:v for k,v in summary.items() if k!='items'}, ensure_ascii=False).encode()).decode()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Response(content=zip_bytes, media_type="application/zip",
                    headers={**headers,
                             "Content-Disposition": f'attachment; filename="vectorized-{stamp}.zip"'})


@app.get('/api/providers')
def providers():
    from .providers import provider_status
    return {'providers':[provider_status()]}


@app.post('/api/provider/vectorizer_ai/convert')
async def provider_convert(file: UploadFile = File(...)):
    name, data = await _read_upload(file)
    from .providers import vectorize_remote
    try:
        svg, details = await run_in_threadpool(vectorize_remote, name, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {'svg':svg,'filename':name,'provider':details}


@app.post('/api/export')
def export(
    svg: str = Form(...),
    fmt: str = Form("svg"),
    stem: str = Form("vectorized"),
) -> Response:
    fmt = fmt.lower()
    if fmt not in {'svg','pdf','eps','ai'}:
        raise HTTPException(status_code=422, detail='不支持的导出格式')
    import re
    from urllib.parse import quote
    stem = re.sub(r'[<>:"/\\\\|?*\r\n]', '_', stem).strip('. ') or 'vectorized'
    data = svg.encode('utf-8')
    media = "image/svg+xml"
    filename = f"{stem}.svg"
    try:
        if fmt in {'pdf','ai'}:
            data,media,filename=svg_to_pdf(svg),'application/pdf',f'{stem}.{fmt}'
        elif fmt=='eps':
            data,media,filename=svg_to_eps(svg),'application/postscript',f'{stem}.eps'
    except ValueError as exc:
        raise HTTPException(status_code=422,detail=str(exc)) from exc
    return Response(content=data, media_type=media,
                    headers={'Content-Disposition': f"attachment; filename=\"vectorized.{fmt}\"; filename*=UTF-8''{quote(filename)}"})


@app.post("/api/sample/{name}/convert")
def convert_sample(
    name: str,
    preset: str = Form("balanced"),
    engine: str = Form("auto"),
    tolerance: float | None = Form(None),
    max_nodes: int | None = Form(None),
    max_colors: int | None = Form(None),
    gradient_levels: int | None = Form(None),
    filter_speckle: int | None = Form(None),
    corner_angle: float | None = Form(None),
    smooth: bool | None = Form(None),
    auto_verify: bool | None = Form(None),
    soft_effects: bool = Form(True),
    strict_input: bool = Form(False),
    with_preview: bool = Form(True),
    glyph_count: int | None = Form(None),
    spot_matching: bool = Form(False),
    spot_colors: str | None = Form(None),
) -> JSONResponse:
    path = SAMPLES / Path(name).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="样本不存在")
    opts = build_options(preset, engine, tolerance, max_nodes, max_colors,
                         gradient_levels, filter_speckle, corner_angle, smooth,
                         auto_verify, spot_matching=spot_matching, spot_colors=spot_colors, soft_effects=soft_effects)
    cache_key = _sample_cache_key(path.name, opts, strict_input, with_preview, glyph_count)
    cached = _cache_get(cache_key)
    if cached is not None:
        payload = dict(cached)
        payload["cached"] = True
        return JSONResponse(payload)

    m_glyphs, m_text = _manifest_meta(path.name)
    try:
        result = convert_bytes(path.name, path.read_bytes(), opts,
                               with_preview=with_preview, strict_input=strict_input,
                               glyph_count=glyph_count or m_glyphs, glyph_text=m_text)
    except Exception as exc:
        return JSONResponse({"error": f"转换失败：{exc}"}, status_code=500)
    payload = result.to_dict(with_preview=with_preview)
    payload["options"] = opts.to_dict()
    payload["cached"] = False
    if not result.error:
        _cache_put(cache_key, payload)
    return JSONResponse(payload)


@app.get("/api/cache/stats")
def cache_stats() -> dict:
    return {"samples_cached": len(_SAMPLE_CACHE), "capacity": _SAMPLE_CACHE_MAX}


def _warm_sample_cache() -> None:
    """启动后在后台预热常用示例，让演示点击几乎瞬时出结果。"""
    if os.environ.get('ZIJI_WARMUP', '0') != '1':
        return
    import threading

    def work() -> None:
        # 慢素材放后面：先把常用示例热好，避免启动瞬间抢 CPU
        order = sorted(WARMUP_SAMPLES,
                       key=lambda n: n.startswith(("21", "12", "11", "14")))
        for name in order:
            path = SAMPLES / name
            if not path.exists():
                continue
            try:
                opts = build_options("balanced", "auto")
                m_glyphs, m_text = _manifest_meta(path.name)
                result = convert_bytes(path.name, path.read_bytes(), opts,
                                       with_preview=True, strict_input=False,
                                       glyph_count=m_glyphs, glyph_text=m_text)
                if not result.error:
                    payload = result.to_dict(with_preview=True)
                    payload["options"] = opts.to_dict()
                    payload["cached"] = False
                    _cache_put(_sample_cache_key(path.name, opts, False), payload)
            except Exception:      # 预热失败不影响服务
                pass
            time.sleep(0.25)       # 让出 CPU，避免影响真实请求

    threading.Thread(target=work, daemon=True, name="sample-warmup").start()


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
