"""Foreground tracing and an explicitly pixel-based source archive.

Curve output removes uniform backgrounds without deskew or denoise. Soft-effect
candidates pass source and regional gates before replacing color patches.
Pixel-exact remains an unmodified grid archive, not recovered font curves.
"""
from __future__ import annotations
import importlib.util
import io
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .native_render import render_svg
from .background import PAPER_BACKGROUNDS, background_canvas

_LOAD_LOCK = threading.Lock()
_NEW_ENGINE = None


def apply_spot_colors(svg: str, spot_colors: list[str]) -> tuple[str, dict]:
    """Map solid SVG fills to the nearest user-selected spot color.

    This is an explicit opt-in post-process. Gradient definitions and backdrop
    groups remain untouched so the branch changes only flat color layers.
    """
    import xml.etree.ElementTree as ET
    from .spot_colors import nearest_color, normalize_hex, normalize_palette

    palette = normalize_palette(spot_colors)

    root = ET.fromstring(svg)
    if root.tag.startswith('{http://www.w3.org/2000/svg}'):
        ET.register_namespace('', 'http://www.w3.org/2000/svg')
    changed = 0
    mapping_counts: dict[tuple[str, str], int] = {}

    def walk(node, excluded: bool = False):
        nonlocal changed
        marker = node.get('data-name', '')
        skip = excluded or marker in {'原背景', '原图背景'}
        if not skip:
            fill = normalize_hex(node.get('fill', ''))
            if fill:
                target = nearest_color(fill, palette)
                if target != fill:
                    node.set('fill', target)
                    changed += 1
                    key = (fill, target)
                    mapping_counts[key] = mapping_counts.get(key, 0) + 1
        for child in list(node):
            walk(child, skip)

    walk(root)
    return ET.tostring(root, encoding='unicode'), {
        'enabled': True,
        'matched_layers': changed,
        'palette': palette,
        'mapping_counts': [
            {'from': source, 'to': target, 'layers': count}
            for (source, target), count in sorted(mapping_counts.items())
        ],
        'gradients_unchanged': True,
    }


def load_engine():
    global _NEW_ENGINE
    with _LOAD_LOCK:
        if _NEW_ENGINE is None:
            # The 0.6 API remains installed for the existing three modes.
            init = Path(__file__).resolve().parents[2]/'.tools/vtracer1/vtracer/__init__.py'
            if not init.is_file():
                raise RuntimeError('保真曲线需要隔离的 vtracer==1.0.0a4；请运行安装说明中的命令')
            spec = importlib.util.spec_from_file_location('_ziji_fidelity_vtracer',init,
                                                         submodule_search_locations=[str(init.parent)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            if not hasattr(module,'Config'):
                raise RuntimeError('隔离的 VTracer 版本不支持共享边界曲线')
            _NEW_ENGINE = module
    return _NEW_ENGINE


def source_error(source, output):
    """Use actual original RGBA on black AND white, without silhouette scores.

    Ignore hidden RGB at alpha=0. Report all-pixel, active-region and worst-case
    errors separately so empty canvas cannot hide damaged strokes.
    """
    if source.shape != output.shape:
        raise ValueError('回渲尺寸与原图不一致')
    def over(a,b):
        a=a.astype(np.float64)
        return a[:,:,:3]*a[:,:,3:4]/255 + b*(1-a[:,:,3:4]/255)
    difference=np.maximum(np.abs(over(source,0)-over(output,0)),
                          np.abs(over(source,255)-over(output,255)))
    border=np.r_[source[0,:,:3],source[-1,:,:3],source[:,0,:3],source[:,-1,:3]]
    bg=np.median(border,axis=0)
    if np.any(source[:,:,3]<255):
        active=source[:,:,3]>0
    else:
        active=np.linalg.norm(source[:,:,:3].astype(float)-bg,axis=2)>5
    active=cv2.dilate(active.astype('uint8'),np.ones((5,5),'uint8'))>0
    if not active.any(): active=np.ones(source.shape[:2],bool)
    # Integer premultiplied samples describe what an 8-bit renderer stores.
    src_p=np.rint(source[:,:,:3].astype(float)*source[:,:,3:4]/255).astype('int16')
    out_p=np.rint(output[:,:,:3].astype(float)*output[:,:,3:4]/255).astype('int16')
    premul=np.concatenate([src_p-out_p,source[:,:,3:4].astype('int16')-output[:,:,3:4]],axis=2)
    # Adjacent small colour regions can introduce visible stripes despite a
    # lower mean colour error. Check newly introduced high-frequency variation
    # in reference regions that were smooth and well inside the artwork.
    reference_laplacian=cv2.Laplacian(over(source,255).astype('float32'),cv2.CV_32F)
    output_laplacian=cv2.Laplacian(over(output,255).astype('float32'),cv2.CV_32F)
    core=cv2.erode(active.astype('uint8'),np.ones((11,11),'uint8'))>0
    core &= np.max(np.abs(reference_laplacian),axis=2)<5
    high_frequency=float(np.abs(output_laplacian-reference_laplacian)[core].mean()) if core.any() else 0.
    return {'mae':float(difference.mean()),'active_mae':float(difference[active].mean()),
            'max_error':float(difference.max()),'p99_active':float(np.percentile(difference[active],99)),
            'pixels_over_8':int((difference.max(2)>8).sum()),
            'premultiplied_max_error':int(np.abs(premul).max()),
            'native_grid_exact':bool(not np.any(premul)),
            'scope':'original RGBA; black and white composites; native resolution',
            'active_pixels':int(active.sum()),'smooth_region_laplacian_mae':high_frequency}


def pixel_paths(source):
    """Merge equal-colour horizontal runs vertically into exact cell rectangles.

    All rectangles have integer edges. No raster payload, tracing, filtering,
    palette quantization or approximation is involved; every visible source
    RGBA sample is represented. Scaling reveals the original pixel grid.
    """
    h,w=source.shape[:2]
    packed=(source[:,:,0].astype('uint32')<<24 | source[:,:,1].astype('uint32')<<16 |
            source[:,:,2].astype('uint32')<<8 | source[:,:,3].astype('uint32'))
    # RGB of completely transparent pixels is unobservable and not archived.
    packed[source[:,:,3]==0]=0
    groups=defaultdict(list)
    active={}
    count=0
    def emit(key, start, end):
        nonlocal count
        x0,x1,color=key
        if not color & 255: return
        groups[color].append(f'M{x0} {start}H{x1}V{end}H{x0}Z')
        count+=1
    for y,row in enumerate(packed):
        boundaries=np.r_[0,np.flatnonzero(row[1:]!=row[:-1])+1,w]
        next_active={}
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            color=int(row[a])
            if not color & 255: continue
            key=(int(a),int(b),color)
            next_active[key]=active.pop(key,y)
        for key,start in active.items(): emit(key,start,y)
        active=next_active
    for key,start in active.items(): emit(key,start,h)
    body=[]
    for color,paths in groups.items():
        alpha=(color&255)/255
        attrs=f'fill="#{color>>8:06x}"'
        if alpha<1: attrs+=f' fill-opacity="{alpha:.12f}"'
        # Bound individual path lengths for SVG importers.
        for i in range(0,len(paths),2048):
            body.append(f'<path {attrs} d="{"".join(paths[i:i+2048])}"/>')
    svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}" shape-rendering="crispEdges">'
         '<title>原像素路径存档（放大保留像素网格）</title>'
         '<desc>Exact RGBA cell geometry; not smooth font outline recovery. '
         'No embedded bitmap. Hidden RGB at alpha=0 is omitted.</desc>'
         +'<g id="source-pixel-cells" data-name="原像素路径">'+''.join(body)+'</g></svg>')
    return svg,{'engine':'pixel_paths','rectangles':count,'node_total':count*4,
                'layers':len(groups),'representation':'pixel_grid','preserves_background':True}


def trace_curves(source, progress=None, *, soft_effects=True):
    started = time.perf_counter()
    # A textured raster can contain hundreds of thousands of tiny colour
    # regions. Preserve its samples explicitly instead of pretending that a
    # compact font outline represents the background faithfully.
    # A contiguous RGBA uint8 pixel is exactly one uint32 key. Sorting these
    # keys preserves the previous distinct-color decision without the costly
    # structured four-column sort (including RGB hidden under zero alpha).
    if len(np.unique(np.ascontiguousarray(source).view(np.uint32)))>65536:
        svg,stats=pixel_paths(source)
        stats['fallback_reason']='dense_source_colors; retained entire background as pixel paths'
        pixels=render_svg(svg)
        return svg,pixels,stats,source_error(source,pixels)
    engine=load_engine()
    h,w=source.shape[:2]
    opaque=bool(np.all(source[:,:,3]==255))
    border=np.r_[source[0,:,:3],source[-1,:,:3],source[:,0,:3],source[:,-1,:3]]
    uniform=opaque and bool(np.all(border==border[0]))
    x0=y0=0;region=source
    if uniform or not opaque:
        ink=np.any(source[:,:,:3]!=border[0],axis=2) if uniform else source[:,:,3]>0
        yy,xx=np.nonzero(ink)
        if len(xx):
            x0=max(0,int(xx.min())-16);y0=max(0,int(yy.min())-16)
            x1=min(w,int(xx.max())+17);y1=min(h,int(yy.max())+17)
            region=source[y0:y1,x0:x1]
    def restore_canvas(svg):
        import xml.etree.ElementTree as ET
        root=ET.fromstring(svg)
        backing=(f'<g data-name="原背景"><path fill="#{bytes(border[0]).hex()}" d="M0 0H{w}V{h}H0Z"/></g>' if uniform else '')
        inner=''.join(ET.tostring(child,encoding='unicode') for child in root)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
                f'viewBox="0 0 {w} {h}"><title>原图保真曲线（近似）</title>{backing}'
                f'<g data-name="保真曲线" transform="translate({x0},{y0})">'
                +inner+'</g></svg>')
    buf=io.BytesIO();Image.fromarray(region).save(buf,format='PNG')
    payload=buf.getvalue()
    candidates=[]
    # Watershed is a separate candidate: it is often better on gradients but
    # damages transparent inputs. Always compare against source before choosing.
    modes=['color-cluster']
    if np.all(source[:,:,3]==255): modes.append('watershed')
    for mode in modes:
        cfg=engine.Config(hierarchical='cutout',mode='spline',filter_speckle=0,
                          color_precision=8,layer_difference=1,path_precision=4,optimize=0)
        if mode=='watershed': cfg.clustering=mode;cfg.watershed_detail=256
        svg=restore_canvas(cfg.convert_bytes(payload))
        pixels=render_svg(svg)
        candidates.append((svg,pixels,source_error(source,pixels),mode))
    if np.any(source[:,:,3]<255):
        # Monochrome alpha coverage is better represented by a single boundary
        # than by a colour tracer that quantizes partially transparent pixels.
        from .preprocess import preprocess
        from .direct import monochrome_document
        from .vectorize import VectorizeOptions
        prepared=preprocess(cv2.cvtColor(source,cv2.COLOR_RGBA2BGRA),deskew=False,
                            denoise=False,enhance=False,max_side=max(source.shape[:2]))
        result=monochrome_document(prepared,VectorizeOptions(tolerance=.15,enforce_node_budget=False,
                                                           max_nodes=100000))
        if result:
            doc,_=result
            x,y,_,_=prepared.crop_box or (0,0,source.shape[1],source.shape[0])
            import xml.etree.ElementTree as ET
            root=ET.fromstring(doc.to_svg())
            inner=''.join(ET.tostring(e,encoding='unicode') for e in root)
            svg=(f'<svg xmlns="http://www.w3.org/2000/svg" width="{source.shape[1]}" '
                 f'height="{source.shape[0]}" viewBox="0 0 {source.shape[1]} {source.shape[0]}">'
                 f'<title>透明覆盖率曲线（近似）</title><g data-name="透明字形" transform="translate({x},{y})">{inner}</g></svg>')
            pixels=render_svg(svg)
            candidates.append((svg,pixels,source_error(source,pixels),'alpha-coverage'))
    # Texture-bearing artwork must not be replaced by a much less accurate
    # candidate just because it has a smoother interior. Use this term only
    # to choose between candidates within 0.8/255 of the best source MAE.
    lowest=min(item[2]['active_mae'] for item in candidates)
    eligible=[item for item in candidates if item[2]['active_mae']<=lowest+.8]
    best=min(eligible,key=lambda item:(item[2]['active_mae']+4*item[2]['smooth_region_laplacian_mae'],
                                         item[2]['pixels_over_8']))
    svg,pixels,report,mode=best
    import xml.etree.ElementTree as ET
    from .svgutil import count_nodes_from_d
    paths=[element for element in ET.fromstring(svg).iter() if element.tag.rsplit('}',1)[-1]=='path']
    stats={'engine':'fidelity_curves','version':getattr(engine,'__version__','1.0.0a4'),
           'algorithm':mode,'representation':'approximate_curves','preserves_background':True,
           'node_budget_applied':False,'candidate_errors':[{**r,'algorithm':m} for _,_,r,m in candidates],
           'node_total':sum(count_nodes_from_d(path.get('d','')) for path in paths),
           'layers':len(paths)}
    baseline_ms = (time.perf_counter()-started)*1000
    if soft_effects:
        from .soft_effects import soft_effect_candidate, compare_effect_candidate
        if progress: progress('effects')
        try:
            candidate, diagnostic = soft_effect_candidate(source)
        except (ValueError, ArithmeticError, cv2.error) as exc:
            # A failed optional fit must not discard an already verified trace.
            # Cancellation and programming errors are deliberately not swallowed.
            candidate = None
            diagnostic = {'attempted': True, 'selected': False,
                          'reason': 'numerical_fit_failed', 'detail': str(exc)}
        if candidate is not None:
            candidate_report = source_error(source, candidate[1])
            comparison = compare_effect_candidate(source, pixels, report, candidate, candidate_report)
            diagnostic['comparison'] = comparison
            diagnostic['candidate_source_error'] = candidate_report
            diagnostic['baseline_source_error'] = report
            diagnostic['baseline_nodes'] = stats['node_total']
            diagnostic['candidate_nodes'] = candidate[2]['node_total']
            if candidate[2]['node_total'] >= stats['node_total']:
                comparison['rejected'].append('no_complexity_reduction')
            if not comparison['rejected']:
                diagnostic['selected'] = True
                diagnostic['reason'] = 'source_and_local_gates_passed'
                old_stats = stats
                svg, pixels, stats = candidate[:3]
                stats['candidate_errors'] = old_stats['candidate_errors'] + [
                    {**candidate_report, 'algorithm': stats['algorithm']}]
                report = candidate_report
            else:
                diagnostic['reason'] = 'baseline_comparison_rejected'
        stats['effect_compensation'] = diagnostic
    stats['timings_ms'] = {'baseline_trace_and_verify': baseline_ms,
                           'effects_and_comparison': (time.perf_counter()-started)*1000-baseline_ms,
                           'trace_effects_and_verify': (time.perf_counter()-started)*1000}
    return svg,pixels,stats,report


def convert_source(filename, image, options, with_preview=True, progress=None):
    from .service import ConvertResult
    from .metrics import format_check
    options = options.resolve_spot_options()
    started=time.perf_counter()
    source=cv2.cvtColor(image,cv2.COLOR_BGRA2RGBA if image.shape[2]==4 else cv2.COLOR_BGR2RGBA)
    removal = {'applied': False, 'reason': 'pixel_archive'}
    if options.mode != 'pixel_exact' and options.transparent_background:
        from .background import remove_uniform_background
        foreground, removal = remove_uniform_background(source)
    from .color_analysis import source_colors
    foreground_known = removal['applied'] or removal['reason'] == 'existing_transparency'
    palette = source_colors(source[:, :, :3], foreground[:, :, 3] if foreground_known else source[:, :, 3],
                            scope='foreground' if foreground_known or np.any(source[:, :, 3] < 255) else 'visible_image')
    trace_source = source
    if removal.get('removed_artifact_pixels', 0):
        # Do not ask the tracer/soft-effect fitter to rediscover discarded
        # screenshot frames and badges. Preserve the input for its preview.
        trace_source = source.copy()
        blank = foreground[:, :, 3] == 0
        matte = np.rint(background_canvas(removal, source.shape[:2])).astype('uint8')
        trace_source[blank, :3] = matte[blank]
    if progress: progress('vectorize')
    if options.mode=='pixel_exact':
        svg,stats=pixel_paths(source)
        if progress: progress('verify')
        pixels=render_svg(svg)
        report=source_error(source,pixels)
        # No unverified "exact" label: fail if this renderer exceeds one 8-bit
        # premultiplication unit. This covers low-alpha unpremultiplication.
        if report['premultiplied_max_error']>1:
            raise ValueError('原像素路径回渲校验失败，不能标记为保真输出')
    elif removal['reason'] in PAPER_BACKGROUNDS:
        svg, stats = pixel_paths(foreground)
        stats.update(engine='fidelity_foreground_pixels', algorithm='native_ink_coverage',
                     fallback_reason='paper_grain_and_fine_tonal_texture; preserve native coverage')
        pixels = render_svg(svg)
    else:
        svg,pixels,stats,report=trace_curves(trace_source,progress,soft_effects=options.soft_effects)
        if options.transparent_background:
            if removal['applied']:
                if stats.get('effect_compensation', {}).get('selected'):
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(svg)
                    backing = [child for child in root if child.get('data-name') == '原背景']
                    if not backing:
                        raise ValueError('柔光去底失败：未找到独立背景层')
                    for child in backing:
                        root.remove(child)
                    svg = ET.tostring(root, encoding='unicode')
                    pixels = render_svg(svg)
                    stats['node_total'] -= 4
                    stats['layers'] -= 1
                    removal['method'] = 'verified_soft_effects_without_backdrop'
                else:
                    from .background import remove_traced_background
                    svg = remove_traced_background(svg, trace_source, foreground)
                    pixels = render_svg(svg)
                    import xml.etree.ElementTree as ET
                    from .svgutil import count_nodes_from_d
                    paths = [node for node in ET.fromstring(svg).iter() if node.tag.rsplit('}', 1)[-1]=='path']
                    stats['node_total'] = sum(count_nodes_from_d(node.get('d', '')) for node in paths)
                    stats['layers'] = len(paths)
            else:
                stats['background_removal'] = removal
                if removal['reason'] == 'existing_transparency':
                    stats['preserves_background'] = False
    if removal['applied']:
        from .background import verify_foreground_export
        effect_diagnostic = stats.get('effect_compensation')
        verified_effects = stats.get('effects') if (effect_diagnostic or {}).get('selected') else None
        verification = verify_foreground_export(foreground, pixels, verified_effects=verified_effects)
        # A palette-only SVG edit is not sufficient proof of removal: a tracer
        # can approximate the matte with a color absent from the input palette.
        # Retry from actual RGBA foreground, so spatial alpha drives tracing.
        if not verification['passed']:
            initial_verification = verification
            svg, pixels, stats, _ = trace_curves(foreground, progress, soft_effects=False)
            verification = verify_foreground_export(foreground, pixels)
            foreground_error = source_error(foreground, pixels)
            if not verification['passed'] or foreground_error['active_mae'] > 4:
                # Preserve native coverage when curves cannot pass the gate.
                # This is explicitly labelled pixel geometry, never smooth
                # outline recovery, and contains no embedded raster image.
                svg, stats = pixel_paths(foreground)
                pixels = render_svg(svg)
                stats.update(engine='fidelity_foreground_pixels',
                             fallback_reason='foreground_curve_verification_failed; preserve native coverage')
                verification = verify_foreground_export(foreground, pixels)
            stats['foreground_retry'] = {'initial_verification': initial_verification,
                                          'reason': 'traced_background_or_missing_foreground'}
            if effect_diagnostic is not None:
                stats['effect_compensation'] = {**effect_diagnostic, 'selected': False}
        if not verification['passed']:
            raise ValueError('SVG 透明背景回渲校验失败，不能交付为去底成功结果')
        removal['export_verification'] = verification
        stats['preserves_background'] = False
        removal['transparent_pixels'] = int((pixels[:, :, 3] == 0).sum())
        stats['background_removal'] = removal
        # Removing the matte intentionally changes alpha. Judge color and
        # geometry by compositing the result onto the estimated original bg.
        opacity = pixels[:, :, 3:4].astype(float)/255
        reconstructed = np.empty_like(pixels)
        reconstructed[:, :, :3] = np.rint(pixels[:, :, :3]*opacity+
            background_canvas(removal, source.shape[:2])*(1-opacity)).astype('uint8')
        reconstructed[:, :, 3] = 255
        report = source_error(trace_source, reconstructed)
        report['scope'] = 'output composited on original background; native resolution; alpha removal intentional'
        if removal.get('removed_artifact_pixels', 0):
            report['scope'] += '; detected background artifacts intentionally excluded'
            report['excluded_background_artifact_pixels'] = removal['removed_artifact_pixels']
    h,w=source.shape[:2]
    warnings=(['像素覆盖路径保留原分辨率纹理，放大仍有像素网格；不等于恢复平滑字形。']
              if stats['representation']=='pixel_grid' else
              ['曲线结果仍为近似，请检查放大细节；未承诺无失真或单字节点上限。'])
    if removal['reason'] in PAPER_BACKGROUNDS:
        warnings.append('已去除浅色纸底颗粒，按原分辨率保存文字灰度与柔边；未锐化或重绘，放大仍有像素网格，节点预算未验收。')
        if removal['reason'] == 'shaded_light_paper_background':
            warnings.append('已按空间底色模型去除浅色渐变纸底及连续截图黑边。')
    elif removal['applied']:
        warnings.append('已自动去除纯色背景，输出透明底；保留原画布，不执行校斜。')
    elif removal['reason'] == 'existing_transparency':
        warnings.append('保留输入的透明背景；不执行校斜或去噪。')
    elif removal['reason'] == 'nonuniform_border':
        warnings.append('边缘背景不均匀，未自动去底；请更换干净纯色背景素材。')
    else:
        warnings.append('原像素路径保留原背景和透明度；不执行去背景、校斜或去噪。' if options.mode=='pixel_exact'
                        else '未检测到可去除的纯色背景；保留原画布。')
    if removal.get('removed_artifact_pixels', 0):
        warnings.append('已清理与字身分离的截图边框和外围杂物；相连笔画、附近点画及光晕予以保留。')
    if options.spot_matching and options.spot_colors:
        svg, spot_info = apply_spot_colors(svg, options.spot_colors)
        pixels = render_svg(svg)
        stats['spot_matching'] = spot_info
        warnings.append(f"已启用专色匹配：改写 {spot_info['matched_layers']} 个纯色图层；渐变保持原样。")
        if removal['applied']:
            opacity = pixels[:, :, 3:4].astype(float) / 255
            reconstructed = np.empty_like(pixels)
            reconstructed[:, :, :3] = np.rint(
                pixels[:, :, :3] * opacity + background_canvas(removal, source.shape[:2]) * (1 - opacity)
            ).astype('uint8')
            reconstructed[:, :, 3] = 255
            report = source_error(trace_source, reconstructed)
            report['scope'] = 'output composited on original background; native resolution; alpha removal intentional'
        else:
            report = source_error(trace_source, pixels)
        report['intentional_recoloring'] = True
        if removal.get('removed_artifact_pixels', 0):
            report['scope'] += '; detected background artifacts intentionally excluded'
            report['excluded_background_artifact_pixels'] = removal['removed_artifact_pixels']
    trace_ms = (time.perf_counter()-started)*1000
    if stats.get('effect_compensation', {}).get('selected'):
        warnings.append('柔光采用可编辑 SVG 滤镜；PDF/AI 导出会近似展开，EPS 不支持这些透明效果。')
        if stats.get('effect_editing', {}).get('follows_body'):
            warnings.append('柔光附着在字身复合路径上，修改字身可联动；释放复合路径后的效果未验收。')
        else:
            warnings.append('多发光源独立保留，修改字身节点后需同步调整发光源；跨编辑器效果仍需核验。')
    background_failed = removal['reason'] in {'nonuniform_border', 'no_foreground'}
    if background_failed:
        warnings.insert(0, '自动去背景未成功，当前结果仍含原背景；请更换干净素材。')
    metrics={'source_colors':palette,'source_fidelity':report,'engine':stats['engine'],'node_total':stats['node_total'],
             'node_per_glyph':stats['node_total']/options.glyph_count if options.glyph_count else None,
             'glyph_count':options.glyph_count,'glyph_source':'provided' if options.glyph_count else 'unknown',
             'format':format_check(svg),'checks_scope':'original_source_native_grid',
             'checks':{'passed':False},
             'assessment':{'status':'untested','scope':'source_fidelity',
                'items':{'smooth_lossless':{'label':'任意放大无失真平滑字形','status':'untested','value':None}}},
             'background_removal': removal,
             'source_transform':{'input_shape':[h,w],'canvas_shape':[h,w],'rotation':None,'crop_box':None}}
    def encode(p):
        buf=io.BytesIO();Image.fromarray(p).save(buf,format='PNG');return buf.getvalue()
    source_png=encode(source) if with_preview else None
    preview_png=None
    if with_preview:
        from PIL import ImageDraw
        from .preview import over_checker, _font
        def premul(a):
            a=a.astype(float);return a[:,:,:3]*a[:,:,3:4]/255
        diff=np.max(np.abs(premul(source)-premul(pixels)),axis=2)
        heat=np.zeros((h,w,3),dtype=np.uint8)
        heat[:,:,0]=np.clip(diff*8,0,255).astype('uint8')
        cw=400;ch=max(1,round(h*cw/w))
        canvas=Image.new('RGB',(cw*3,ch+40),'white');draw=ImageDraw.Draw(canvas)
        for index,(panel,label) in enumerate(zip(
            [over_checker(source,False),over_checker(pixels,False),heat],
            ['原图（原始坐标）','SVG 回渲','差异放大 8 倍（红色）'])):
            draw.text((index*cw+8,8),label,font=_font(17),fill='black')
            canvas.paste(Image.fromarray(panel).resize((cw,ch)),(index*cw,40))
        buf=io.BytesIO();canvas.save(buf,format='PNG');preview_png=buf.getvalue()
    elapsed=(time.perf_counter()-started)*1000
    metrics['elapsed_ms']=elapsed
    return ConvertResult(filename,Path(filename.replace('\\','/')).stem,svg,metrics,
                          {'width':w,'height':h,'severity':'warning' if removal['reason'] in PAPER_BACKGROUNDS or background_failed else 'ok','invalid':background_failed,'passed':not background_failed,
                           'invalid_reasons':['background_removal_failed'] if background_failed else [],
                           'warnings':warnings,'scope':'uniform_background_removal' if removal['applied'] else 'source_preservation; no material rejection'},stats,
                         {'width':w,'height':h},elapsed,warnings,preview_png,source_png,
                         timings={'total_ms':elapsed, 'trace_verify_ms':trace_ms,
                                  'preview_and_metadata_ms':elapsed-trace_ms,
                                  'scope':'trace render verification preview included; caller adds decode'})
