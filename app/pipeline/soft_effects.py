"""Infer editable soft effects without quantizing the halo into colored paths.

The supported model is flat, opaque lettering over Gaussian source-over effects,
on an exactly uniform opaque or transparent backdrop. All estimates come from
source pixels, not filenames, OCR, font lookup or sample-generation parameters.
Unsupported gradients/textures keep the ordinary tracer through quality gates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import shift
from scipy.optimize import least_squares

from .direct import fit_mask_paths
from .native_render import render_svg
from .svgutil import rgb_to_hex
from .topology import compare_topology


@dataclass
class Effect:
    mask: np.ndarray
    sigma: float
    dx: float
    dy: float
    color: np.ndarray
    opacity: float
    cutoff: float = 0.


def _blur(mask, sigma, dx=0., dy=0.):
    field = cv2.GaussianBlur(mask.astype('float32'), (0, 0), max(.3, float(sigma)),
                             borderType=cv2.BORDER_CONSTANT)
    return shift(field, (dy, dx), order=1, prefilter=False, mode='constant')


def _premul(source):
    rgba = source.astype('float32') / 255
    rgba[:, :, :3] *= rgba[:, :, 3:4]
    return rgba


def _composite(mask, effects, background):
    result = np.broadcast_to(background, (*mask.shape, 4)).copy()
    for effect in effects:
        alpha = np.clip(_blur(effect.mask, effect.sigma, effect.dx, effect.dy)-effect.cutoff, 0, 1) * effect.opacity
        paint = np.dstack([alpha[:, :, None] * effect.color, alpha])
        result = paint + result * (1 - alpha[:, :, None])
    return result


def _fit_effect(mask, target, background, outside, *, scale=1.):
    """Recover geometry with a tail cutoff; compatible output is checked separately."""
    h, w = mask.shape
    target_delta = target - background
    opaque = background[3] == 1
    delta = target_delta[:, :, :3] if opaque else target_delta
    signal = np.max(np.abs(delta), axis=2)
    nearby = cv2.distanceTransform((mask < .5).astype('uint8'), cv2.DIST_L2, 3) < 100 * scale
    selected = outside & (nearby | (signal > .01))
    y, x = np.where(selected)
    if len(y) < 60 or np.count_nonzero(signal[outside] > .015) < 30:
        return None
    stride = max(1, len(y) // 6000)
    y, x = y[::stride], x[::stride]
    observed = delta[y, x].astype('float64')

    def solve(parameters, return_paint=False):
        dx, dy, sigma, cutoff = parameters
        basis = np.maximum(0, _blur(mask, sigma, dx, dy)[y, x].astype('float64')-cutoff)
        denominator = float(basis @ basis)
        beta = (basis[:, None] * observed).sum(0) / max(denominator, 1e-8)
        if opaque:
            beta = np.clip(beta, -background[:3], 1-background[:3])
        else:
            beta[3] = np.clip(beta[3], 0, 1)
            beta[:3] = np.clip(beta[:3], 0, beta[3])
        if return_paint:
            if opaque:
                opacity = max(float(np.max(np.divide(beta, 1-background[:3],
                                   out=np.zeros(3), where=beta > 0))),
                              float(np.max(np.divide(-beta, background[:3],
                                   out=np.zeros(3), where=beta < 0))), .001)
                color = np.clip(background[:3] + beta/opacity, 0, 1)
            else:
                opacity = float(beta[3])
                color = np.clip(beta[:3]/max(opacity, 1e-8), 0, 1)
            return color, opacity
        return (basis[:, None] * beta - observed).ravel()

    def jacobian(parameters):
        # Absolute steps remain meaningful at dx=dy=0 and with float32 OpenCV
        # fields. Relative finite differences at zero can freeze the offset fit.
        columns = []
        for index, step in enumerate((.08, .08, .08, .0002)):
            lo, hi = np.array(parameters), np.array(parameters)
            lo[index] -= step
            hi[index] += step
            columns.append((solve(hi)-solve(lo))/(2*step))
        return np.column_stack(columns)

    maximum = min(64 * scale, min(h, w)/3)
    if maximum < 1.1:
        return None
    best = None
    # Two seeds avoid treating a translated shadow as a very wide centered glow.
    for seed_sigma in (6., 20.):
        initial = [0., 0., np.clip(seed_sigma*scale, .8, maximum*.9), .001]
        fit = least_squares(solve, initial, bounds=([-maximum, -maximum, .7, 0.],
                            [maximum, maximum, maximum, .025]), jac=jacobian,
                            max_nfev=25, ftol=2e-3, xtol=2e-3, gtol=1e-5)
        if best is None or fit.cost < best.cost:
            best = fit
    color, opacity = solve(best.x, True)
    if opacity < .01:
        return None
    dx, dy, sigma, cutoff = map(float, best.x)
    return Effect(mask, sigma, dx, dy, color, min(1., opacity), cutoff)


def _document(width, height, paths, ink, effects, background, emitter_paths=None):
    """Emit recognized live effects; attach a union glow to its compound path.

    Multiple emitter colors retain separate, translated paths with pure blur.
    Do not apply a blur to each subpath: intersecting halos would accumulate.
    """
    defs, groups = [], []
    linked = len(effects) == 1 and emitter_paths is None
    for index, effect in enumerate(effects):
        fid = f'source-soft-effect-{index}'
        # One filter per entire emitter group: overlapping subpaths must not
        # receive separate blurs and accumulate alpha at their intersections.
        if effect.cutoff != 0:
            raise ValueError('兼容柔光不能包含透明度截止处理')
        # The explicit shadow chain is recognized by VectorCraft and avoids
        # the different flood-color conversion of our renderer's feDropShadow.
        primitive = (f'<feGaussianBlur in="SourceAlpha" stdDeviation="{effect.sigma:.5f}" result="blur"/>'
                     f'<feOffset in="blur" dx="{effect.dx:.5f}" dy="{effect.dy:.5f}" result="offset"/>'
                     f'<feFlood flood-color="{rgb_to_hex(effect.color*255)}" '
                     f'flood-opacity="{effect.opacity:.6f}" result="paint"/>'
                     f'<feComposite in="paint" in2="offset" operator="in" result="shadow"/>'
                     f'<feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge>') if linked else (
                     f'<feGaussianBlur stdDeviation="{effect.sigma:.5f}"/>')
        defs.append(f'<filter id="{fid}" filterUnits="userSpaceOnUse" '
                    f'x="0" y="0" width="{width}" height="{height}" '
                    f'color-interpolation-filters="sRGB">{primitive}</filter>')
        if linked:
            continue
        emission = paths if emitter_paths is None else emitter_paths[index]
        body = ''.join(f'<path d="{p}"/>' for p in emission)
        groups.append(f'<g id="soft-emitter-{index+1}" data-name="光晕或软阴影 {index+1}" '
                      f'fill="{rgb_to_hex(effect.color*255)}" fill-rule="evenodd" '
                      f'opacity="{effect.opacity:.6f}" filter="url(#{fid})">'
                      f'<g transform="translate({effect.dx:.5f} {effect.dy:.5f})">{body}</g></g>')
    backing = (f'<g data-name="原背景"><path fill="{rgb_to_hex(background[:3]*255)}" '
               f'd="M0 0H{width}V{height}H0Z"/></g>') if background[3] else ''
    if linked:
        # Keep the live effect on the compound path, NOT its ordinary wrapper:
        # ungrouping the wrapper then preserves the path's effect in VectorCraft.
        lettering = (f'<path id="lettering-compound" fill-rule="evenodd" d="{" ".join(paths)}" '
                     f'filter="url(#source-soft-effect-0)"/>')
    else:
        lettering = ''.join(f'<path d="{p}"/>' for p in paths)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}"><title>艺术字与兼容柔光效果（近似）</title>'
            f'<defs>{"".join(defs)}</defs>{backing}{"".join(groups)}'
            f'<g id="lettering" data-name="原图字身" fill="{rgb_to_hex(ink*255)}" fill-rule="evenodd">'
            f'{lettering}</g></svg>')


def _partition_effects(target, coverage, paths, background, crop, scale):
    """Fit separate light colors around disconnected emitters, with bounded work."""
    from .pathparse import parse_path
    count, labels, areas, _ = cv2.connectedComponentsWithStats((coverage > 0).astype('uint8'), 8)
    if not 2 <= count-1 <= 6 or np.any(areas[1:, cv2.CC_STAT_AREA] < 16):
        return None
    groups = [[] for _ in range(count-1)]
    for path in paths:
        points = np.concatenate(parse_path(path, resolution=3))
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, labels.shape[1]-1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, labels.shape[0]-1)
        hits = labels[y, x]
        if not np.any(hits):
            return None
        index = int(np.bincount(hits[hits > 0]).argmax())-1
        groups[index].append(path)
    if any(not group for group in groups):
        return None
    x0, y0, x1, y1 = crop
    size = (max(2, round((x1-x0)*scale)), max(2, round((y1-y0)*scale)))
    small_target = cv2.resize(target[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
    masks = [coverage*(labels == index) for index in range(1, count)]
    small_masks = [cv2.resize(mask[y0:y1, x0:x1].astype('float32'), size,
                             interpolation=cv2.INTER_AREA) for mask in masks]
    distances = [cv2.distanceTransform((mask < .25).astype('uint8'), cv2.DIST_L2, 3)
                 for mask in small_masks]
    owner = np.argmin(distances, axis=0)
    outside = cv2.dilate((np.maximum.reduce(small_masks) > .1).astype('uint8'),
                          np.ones((3, 3), 'uint8')) == 0
    effects = []
    for index, mask in enumerate(small_masks):
        effect = _fit_effect(mask, small_target, background, outside & (owner == index), scale=scale)
        if effect is None:
            return None
        effect.mask = masks[index].astype('float32')
        effect.sigma /= scale
        effect.dx /= scale
        effect.dy /= scale
        effect.cutoff = 0.
        effects.append(effect)
    return effects, groups


def _effect_gate(local):
    return (local['halo_pixels'] >= 100 and local['halo_mae'] <= 3 and local['halo_p95'] <= 10
            and abs(local['halo_luma_bias']) <= 1.5 and local['core_mae'] <= 3
            and all(region['relative_error'] <= .45 and region['mae'] <= 3
                    and region['p95'] <= 10 and abs(region['luma_bias']) <= 1.5
                    for region in local['regions']))


def _regions_preserved(before, after, allowance=.2):
    baseline = {region['id']: region for region in before['regions']}
    for region in after['regions']:
        original = baseline[region['id']]
        if (region['mae'] > original['mae']+allowance
                or region['weak_mae'] > original['weak_mae']+allowance
                or abs(region['luma_bias']) > abs(original['luma_bias'])+allowance):
            return False
    return True


def _reduce_effects(source, paths, ink, effects, emitter_paths, coverage, background):
    """Try removing low-contribution effects, protecting local and weak light.

    This is a bounded, region-protected simplification inspired by O&R, not a
    reimplementation of its DiffVG optimization or a glyph-node guarantee.
    """
    h, w = source.shape[:2]
    svg = _document(w, h, paths, ink, effects, background, emitter_paths)
    pixels = render_svg(svg)
    local = region_metrics(source, pixels, coverage, background)
    initial_count = len(effects)
    for effect in sorted(effects[:], key=lambda e: float(e.mask.sum()*e.opacity)):
        if len(effects) < 2:
            break
        index = next(i for i, e in enumerate(effects) if e is effect)
        trial_effects = effects[:index]+effects[index+1:]
        trial_paths = emitter_paths[:index]+emitter_paths[index+1:]
        trial_svg = _document(w, h, paths, ink, trial_effects, background, trial_paths)
        trial_pixels = render_svg(trial_svg)
        trial = region_metrics(source, trial_pixels, coverage, background)
        if (trial['halo_mae'] <= local['halo_mae']+.03
                and trial['weak_halo_mae'] <= local['weak_halo_mae']+.03
                and abs(trial['halo_luma_bias']) <= abs(local['halo_luma_bias'])+.03
                and trial['halo_p95'] <= local['halo_p95']+.25
                and trial['spill_over_4_pixels'] <= local['spill_over_4_pixels']
                and _regions_preserved(local, trial, .03)):
            effects, emitter_paths = trial_effects, trial_paths
            svg, pixels, local = trial_svg, trial_pixels, trial
    return svg, pixels, local, effects, emitter_paths, initial_count


def region_metrics(source, output, coverage, background):
    """Use source-defined local ROIs; never let empty canvas hide halo errors."""
    if background[3] == 0:
        support = (source[:, :, 3] > 0) | (output[:, :, 3] > 0)
    else:
        bg = np.rint(background[:3]*255).astype('uint8')
        support = np.any(source[:, :, :3] != bg, axis=2) | np.any(output[:, :, :3] != bg, axis=2)
    yy, xx = np.where(support | (coverage > 0))
    if len(xx):
        # Exclude only pixels identical to the background in BOTH images.
        # This preserves every spill pixel while avoiding full-canvas float work.
        y0, y1 = max(0, yy.min()-2), min(source.shape[0], yy.max()+3)
        x0, x1 = max(0, xx.min()-2), min(source.shape[1], xx.max()+3)
        source, output = source[y0:y1, x0:x1], output[y0:y1, x0:x1]
        coverage = coverage[y0:y1, x0:x1]
    src, dst = _premul(source), _premul(output)
    def over(p, bg):
        return p[:, :, :3] + bg*(1-p[:, :, 3:4])
    difference = np.maximum(np.abs(over(src, 0)-over(dst, 0)),
                            np.abs(over(src, 1)-over(dst, 1))) * 255
    signal = np.max(np.abs(src-background), axis=2)
    predicted = np.max(np.abs(dst-background), axis=2)
    outer = cv2.dilate((coverage > .1).astype('uint8'), np.ones((3, 3), 'uint8')) == 0
    halo = outer & (signal > 2/255)
    spill = outer & (signal <= 2/255) & (predicted > 2/255)
    unsupported = outer & (signal <= 2/255)
    substantial_spill = unsupported & (np.max(difference, axis=2) > 4)
    weak = halo & (signal < .08)
    core = coverage > .9
    edge = (coverage > .02) & (coverage < .98)
    luminosity = (over(dst, 0)-over(src, 0)) @ np.array([.2126, .7152, .0722]) * 255
    distance = cv2.distanceTransform((coverage < .5).astype('uint8'), cv2.DIST_L2, 3)
    profile = []
    for lo, hi in zip((0, 2, 4, 8, 16, 32, 64), (2, 4, 8, 16, 32, 64, 128)):
        ring = halo & (distance >= lo) & (distance < hi)
        if ring.any():
            profile.append({'distance_px': [lo, hi], 'pixels': int(ring.sum()),
                            'signed_srgb_luma_error': float(luminosity[ring].mean())})
    _, ownership = cv2.distanceTransformWithLabels((coverage < .5).astype('uint8'),
                                                   cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_CCOMP)
    regions = []
    for index in np.unique(ownership[halo]):
        part = halo & (ownership == index)
        if part.sum() < 40:
            continue
        faint = part & weak
        regions.append({'id': int(index), 'pixels': int(part.sum()),
                        'mae': float(difference[part].mean()),
                        'p95': float(np.percentile(difference[part], 95)),
                        'luma_bias': float(luminosity[part].mean()),
                        'weak_mae': float(difference[faint].mean()) if faint.any() else 0.,
                        'relative_error': float(np.max(difference, axis=2)[part].mean()/
                                                max(2., signal[part].mean()*255))})
    return {'scope': 'source-defined halo outside recovered body; black AND white composites',
            'halo_pixels': int(halo.sum()),
            'halo_mae': float(difference[halo].mean()) if halo.any() else 0.,
            'halo_p95': float(np.percentile(difference[halo], 95)) if halo.any() else 0.,
            'halo_luma_bias': float(luminosity[halo].mean()) if halo.any() else 0.,
            'weak_halo_mae': float(difference[weak].mean()) if weak.any() else 0.,
            'spill_pixels': int(spill.sum()),
            'spill_over_4_pixels': int(substantial_spill.sum()),
            'spill_mae': float(difference[spill].mean()) if spill.any() else 0.,
            'core_mae': float(difference[core].mean()) if core.any() else 0.,
            'edge_mae': float(difference[edge].mean()) if edge.any() else 0.,
            'decay_profile': profile, 'regions': regions}


def soft_effect_candidate(source):
    """Return a verified candidate and diagnostic, or (None, rejection report)."""
    started = time.perf_counter()
    timing = {}
    diagnostic = {'attempted': False, 'selected': False, 'timings_ms': timing}

    def reject(reason):
        diagnostic['reason'] = reason
        timing['total'] = (time.perf_counter()-started)*1000
        return None, diagnostic

    h, w = source.shape[:2]
    alpha = source[:, :, 3]
    opaque = bool(np.all(alpha == 255))
    border = np.concatenate([source[0], source[-1], source[:, 0], source[:, -1]])
    if opaque and np.all(border[:, :3] == border[0, :3]):
        background = np.r_[border[0, :3]/255., 1.].astype('float32')
    elif not opaque and np.all(border[:, 3] == 0):
        background = np.zeros(4, dtype='float32')
    else:
        return reject('nonuniform_or_nontransparent_border')
    target = _premul(source)
    signal = np.max(np.abs(target-background), axis=2)
    active = signal > 2/255
    if active.sum() < 200:
        return reject('insufficient_foreground')
    # Dominant solid paint is only a seed. Soft pixels are explained by a model,
    # not relabeled as noise; final original-pixel gates reject textured inputs.
    flat = alpha >= 253
    rgb = source[:, :, :3]
    for channel in range(3):
        plane = rgb[:, :, channel]
        flat &= cv2.dilate(plane, np.ones((3, 3), 'uint8')) - cv2.erode(plane, np.ones((3, 3), 'uint8')) <= 1
    contrast = np.max(np.abs(rgb.astype('float32')/255-background[:3]), axis=2)
    solid = flat & (contrast > .25)
    if solid.sum() < 80:
        return reject('no_flat_opaque_body')
    packed = rgb[solid].astype('int32') >> 3
    codes = (packed[:, 0] << 10) | (packed[:, 1] << 5) | packed[:, 2]
    dominant = np.bincount(codes, minlength=32768).argmax()
    colors = rgb[solid][codes == dominant]
    ink = np.median(colors, axis=0).astype('float32')/255
    exact = np.max(np.abs(rgb.astype('float32')/255-ink), axis=2) < 3/255
    seed = exact & (alpha >= 250)
    if seed.sum() < 100:
        return reject('insufficient_body_support')
    outside = cv2.dilate(seed.astype('uint8'), np.ones((7, 7), 'uint8')) == 0
    soft = outside & (signal > .015) & (signal < .8)
    if soft.sum() < max(100, seed.sum()*.025):
        return reject('no_broad_soft_effect')
    # Sharp secondary paint, fine texture and flat decorations cannot be replaced
    # by a blur. The residual gates below provide an additional full-color check.
    lap = np.max(np.abs(cv2.Laplacian(target[:, :, :3], cv2.CV_32F)), axis=2)
    if np.mean(lap[soft] > .06) > .08:
        return reject('textured_or_sharp_exterior')
    sharp = outside & (signal > .015) & (lap > .06)
    _, _, sharp_regions, _ = cv2.connectedComponentsWithStats(sharp.astype('uint8'), 8)
    if np.any(sharp_regions[1:, cv2.CC_STAT_AREA] >= 4):
        # A small colored serif or decoration must not disappear merely because
        # it occupies little area compared with the smooth halo.
        return reject('sharp_exterior_features')
    diagnostic['attempted'] = True
    timing['analysis'] = (time.perf_counter()-started)*1000
    y, x = np.where(active)
    margin = 64
    x0, x1 = max(0, int(x.min())-margin), min(w, int(x.max())+margin+1)
    y0, y1 = max(0, int(y.min())-margin), min(h, int(y.max())+margin+1)
    cropped = target[y0:y1, x0:x1]
    size = cropped.shape[:2]
    scale = min(1., 420/max(size))
    sw, sh = max(2, round(size[1]*scale)), max(2, round(size[0]*scale))
    small_target = cv2.resize(cropped, (sw, sh), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(seed[y0:y1, x0:x1].astype('float32'), (sw, sh), interpolation=cv2.INTER_AREA)
    small_outside = cv2.dilate((small_mask > .1).astype('uint8'), np.ones((3, 3), 'uint8')) == 0
    fit_started = time.perf_counter()
    initial = _fit_effect(small_mask, small_target, background, small_outside, scale=scale)
    if initial is None:
        return reject('effect_fit_failed')
    under = _composite(small_mask, [initial], background)
    full_under = np.broadcast_to(background, (h, w, 4)).copy()
    full_under[y0:y1, x0:x1] = cv2.resize(under, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    paint = np.r_[ink, 1.]
    direction = paint-full_under
    coverage = np.clip(np.sum((target-full_under)*direction, axis=2)/
                       np.maximum(np.sum(direction*direction, axis=2), 1e-6), 0, 1)
    # Only recover coverage close to the solid body. Broad diffuse regions belong
    # exclusively to the effect model, never to a second lettering silhouette.
    distance = cv2.distanceTransform((~seed).astype('uint8'), cv2.DIST_L2, 3)
    coverage[distance > 3] = 0
    coverage[coverage < .02] = 0
    timing['initial_fit'] = (time.perf_counter()-fit_started)*1000
    geometry_started = time.perf_counter()
    paths, body_nodes = fit_mask_paths(np.rint(coverage*255), .1, 55, geometric=True)
    if not paths or body_nodes > 6000:
        return reject('body_geometry_too_complex')
    body_svg = _document(w, h, paths, ink, [], np.zeros(4))
    body_pixels = render_svg(body_svg)
    rendered_coverage = body_pixels[:, :, 3]/255.
    # Combining independent paths changes evenodd behavior if they overlap.
    # Validate the merged silhouette before attaching any live effect to it.
    compound_svg = _document(w, h, [' '.join(paths)], ink, [], np.zeros(4))
    compound_coverage = render_svg(compound_svg)[:, :, 3]/255.
    if np.max(np.abs(compound_coverage-rendered_coverage)) > 1/255 + 1e-6:
        return reject('compound_geometry_changes_fill')
    topology = compare_topology(coverage > .5, rendered_coverage > .5)
    reference = coverage > .5
    actual = rendered_coverage > .5
    iou = float(np.logical_and(reference, actual).sum()/max(1, np.logical_or(reference, actual).sum()))
    edge = (cv2.dilate(reference.astype('uint8'), np.ones((3, 3), 'uint8')) !=
            cv2.erode(reference.astype('uint8'), np.ones((3, 3), 'uint8')))
    edge_error = float(np.abs(rendered_coverage-coverage)[edge].mean()) if edge.any() else 0.
    if not topology['passed'] or iou < .98 or edge_error > .10:
        diagnostic.update(body_topology=topology, body_iou=iou, body_edge_coverage_mae=edge_error)
        return reject('body_geometry_gate')
    timing['body_geometry_and_render'] = (time.perf_counter()-geometry_started)*1000
    fit_started = time.perf_counter()
    small_mask = cv2.resize(rendered_coverage[y0:y1, x0:x1].astype('float32'), (sw, sh), interpolation=cv2.INTER_AREA)
    effect = _fit_effect(small_mask, small_target, background, small_outside, scale=scale)
    if effect is None:
        return reject('refined_effect_fit_failed')
    effect.mask = rendered_coverage.astype('float32')
    effect.dx /= scale
    effect.dy /= scale
    effect.sigma /= scale
    effect.cutoff = 0.
    effects = [effect]
    emitter_paths = None
    svg = _document(w, h, paths, ink, effects, background)
    timing['refined_fit'] = (time.perf_counter()-fit_started)*1000
    verify_started = time.perf_counter()
    pixels = render_svg(svg)
    local = region_metrics(source, pixels, coverage, background)
    complexity = {'strategy': 'fit_one_union_emitter_before_splitting',
                  'initial_effect_layers': 1, 'final_effect_layers': 1,
                  'protect_weak_halo': True}
    if not _effect_gate(local):
        partition_started = time.perf_counter()
        partition = _partition_effects(target, rendered_coverage, paths, background,
                                      (x0, y0, x1, y1), scale)
        if partition is not None:
            effects, emitter_paths = partition
            svg, pixels, local, effects, emitter_paths, count = _reduce_effects(
                source, paths, ink, effects, emitter_paths, coverage, background)
            complexity.update(strategy='partition_then_region_protected_reduce',
                              initial_effect_layers=count, final_effect_layers=len(effects))
        timing['partition_and_reduce'] = (time.perf_counter()-partition_started)*1000
    diagnostic.update(body_iou=iou, body_topology=topology, body_edge_coverage_mae=edge_error,
                      local=local, body_nodes=body_nodes)
    if not _effect_gate(local):
        return reject('local_effect_residual')
    timing['final_render_and_local_check'] = (time.perf_counter()-verify_started)*1000
    timing['total'] = (time.perf_counter()-started)*1000
    diagnostic['reason'] = 'candidate_verified'
    params = [{'sigma': e.sigma, 'dx': e.dx, 'dy': e.dy, 'opacity': e.opacity, 'alpha_cutoff': e.cutoff,
               'color': rgb_to_hex(e.color*255)} for e in effects]
    from .svgutil import count_nodes_from_d
    all_effect_paths = paths if emitter_paths is None else [p for group in emitter_paths for p in group]
    linked = len(effects) == 1 and emitter_paths is None
    effect_nodes = 0 if linked else sum(count_nodes_from_d(p) for p in all_effect_paths)
    stats = {'engine': 'fidelity_curves', 'version': 'source-soft-effects-compatible-v2',
             'algorithm': 'body-and-soft-effects', 'representation': 'approximate_curves',
             'preserves_background': True, 'node_budget_applied': False,
             'node_total': body_nodes+effect_nodes+(4 if opaque else 0), 'body_nodes': body_nodes,
             'layers': 1+len(effects)+(1 if opaque else 0), 'effects': params, 'complexity': complexity,
             'effect_editing': {'target': 'vectorcraft', 'representation':
                 'linked_compound' if linked else 'independent_emitters',
                 'follows_body': linked, 'embeds_raster': False},
             'effect_compensation': diagnostic}
    return (svg, pixels, stats, coverage, background), diagnostic


def compare_effect_candidate(source, baseline_pixels, baseline_report, candidate, candidate_report):
    """Regional protection during complexity reduction, including weak light."""
    _, pixels, stats, coverage, background = candidate
    baseline = region_metrics(source, baseline_pixels, coverage, background)
    proposed = stats['effect_compensation']['local']
    rejected = []
    if candidate_report['active_mae'] > baseline_report['active_mae'] + .2:
        rejected.append('source_active_error')
    if proposed['core_mae'] > baseline['core_mae'] + .5:
        rejected.append('body_color')
    if proposed['edge_mae'] > baseline['edge_mae'] + 1.:
        rejected.append('body_edge_color')
    if proposed['halo_mae'] > baseline['halo_mae']*.90:
        rejected.append('halo_not_improved')
    if proposed['halo_p95'] > baseline['halo_p95'] + .5:
        rejected.append('halo_error_tail')
    if proposed['weak_halo_mae'] > baseline['weak_halo_mae'] + .2:
        rejected.append('weak_halo')
    if abs(proposed['halo_luma_bias']) > abs(baseline['halo_luma_bias']) + .2:
        rejected.append('halo_brightness')
    if not _regions_preserved(baseline, proposed):
        rejected.append('local_emitter_region')
    if (proposed['spill_over_4_pixels'] > baseline['spill_over_4_pixels'] + max(20, proposed['halo_pixels']*.01)
            or proposed['spill_mae'] > max(2., baseline['spill_mae']+.5)):
        rejected.append('unsupported_spill')
    return {'baseline_local': baseline, 'candidate_local': proposed, 'rejected': rejected}
