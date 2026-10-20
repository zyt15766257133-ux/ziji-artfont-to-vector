"""Uniform-background matting without resampling or changing solid lettering."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import xml.etree.ElementTree as ET


PAPER_BACKGROUNDS = {'light_paper_background', 'shaded_light_paper_background'}


def background_canvas(status, shape):
    """Rebuild the estimated matte for verification, including illumination."""
    model = status.get('background_model')
    if not model:
        return np.broadcast_to(np.array(status['color']), (*shape, 3))
    h, w = shape
    y, x = np.mgrid[:h, :w].astype('float64')
    x /= max(w-1, 1)
    y /= max(h-1, 1)
    c = model['coefficients']
    field = np.clip(c[0]+c[1]*x+c[2]*y+c[3]*x*y+c[4]*x*x+c[5]*y*y, 0, 255)
    return np.repeat(field[:, :, None], 3, axis=2)


def _remove_shaded_paper(source):
    """Fit gentle neutral paper illumination from many independent edge blocks.

    This is a restricted dark-ink model, not semantic image segmentation. All
    decisions depend on pixels, never a filename or the sample manifest.
    """
    rejected = {'applied': False, 'reason': 'nonuniform_border'}
    rgb = source[:, :, :3].astype('float64')
    h, w = rgb.shape[:2]
    if min(h, w) < 64 or np.percentile(np.ptp(rgb, axis=2), 99) > 4:
        return source.copy(), rejected
    gray = rgb.mean(2)
    # Exclude only thin, continuous dark bands touching the canvas boundary.
    frame = np.zeros((h, w), bool)
    for reverse in (False, True):
        for i in range(min(12, max(1, h//40))):
            row = h-1-i if reverse else i
            if np.mean(gray[row] < 180) < .95:
                break
            frame[row] = True
        for i in range(min(12, max(1, w//40))):
            col = w-1-i if reverse else i
            if np.mean(gray[:, col] < 180) < .95:
                break
            frame[:, col] = True
    # Use inset strips so screenshot frames cannot dominate the paper estimate.
    band = min(20, min(h, w)//6)
    inset = 4
    if frame.any():
        rows = np.flatnonzero(frame.mean(1) > .95)
        cols = np.flatnonzero(frame.mean(0) > .95)
        depths = [min(int(i)+1, h-int(i)) for i in rows]
        depths += [min(int(i)+1, w-int(i)) for i in cols]
        inset = max(inset, max(depths, default=0)+1)
    band = max(band, inset+4)
    samples = []
    def sample(y0, y1, x0, x1):
        samples.append(((x0+x1-1)/2/(w-1), (y0+y1-1)/2/(h-1),
                        float(np.median(gray[y0:y1, x0:x1]))))
    for x in range(inset, w-inset, 24):
        for lo, hi in ((inset, band), (h-band, h-inset)):
            sample(lo, hi, x, min(x+24, w-inset))
    for y in range(band, h-band, 24):
        for lo, hi in ((inset, band), (w-band, w-inset)):
            sample(y, min(y+24, h-band), lo, hi)
    x, y, values = np.array(samples).T
    design = np.array([np.ones_like(x), x, y, x*y, x*x, y*y]).T
    keep = np.ones(len(values), bool)
    for _ in range(5):
        if keep.mean() < .85:
            return source.copy(), rejected
        coefficients = np.linalg.lstsq(design[keep], values[keep], rcond=None)[0]
        residual = values-design@coefficients
        scale = max(1., 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        keep = np.abs(residual) < max(4., 3*scale)
    if keep.mean() < .85 or np.percentile(np.abs(residual), 95) > 4:
        return source.copy(), rejected
    status = {'applied': True, 'reason': 'shaded_light_paper_background',
              'method': 'neutral_ink_spatial_matte_native_coverage',
              'background_model': {'type': 'quadratic_neutral_paper',
                                   'coefficients': coefficients.tolist()},
              'removed_frame_pixels': int(frame.sum())}
    field = background_canvas(status, (h, w))[:, :, 0]
    # Bounded illumination only; reject a dark/strongly shaded backdrop and
    # abrupt edges that a smooth paper field does not explain.
    if field.min() < 215 or np.ptp(field) > 40 or np.mean(gray-field > 6) > .005:
        return source.copy(), rejected
    strips = np.zeros((h, w), bool)
    strips[inset:band, inset:-inset] = strips[-band:-inset, inset:-inset] = True
    strips[inset:-inset, inset:band] = strips[inset:-inset, -band:-inset] = True
    differences = (field-gray)[strips]
    if np.percentile(np.abs(differences), 95) > 12:
        return source.copy(), rejected
    noise = max(2., float(np.percentile(np.abs(differences), 99)))
    signal = np.maximum(field-gray, 0)
    core = (signal > max(32., noise*2)) & ~frame
    if core.mean() < .005 or core.mean() > .75:
        return source.copy(), rejected
    support = cv2.dilate(core.astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    alpha = np.clip(signal/np.maximum(field, 1), 0, 1)
    alpha[~support | frame | (signal <= noise)] = 0
    result = np.zeros_like(source)
    result[:, :, 3] = np.rint(alpha*255).astype('uint8')
    status.update(color=[float(np.median(field))]*3, paper_noise_level=noise,
                  transparent_pixels=int((result[:, :, 3] == 0).sum()))
    return result, status


def _remove_light_paper(source):
    """Extract neutral ink from a nearly constant light paper field.

    Grain is measured outside the lettering. Coverage inside the lettering is
    retained at the native pixel grid, without smoothing, sharpening or CLAHE.
    """
    rgb = source[:, :, :3].astype('float32')
    h, w = rgb.shape[:2]
    if min(h, w) < 48 or np.percentile(np.ptp(rgb, axis=2), 99) > 4:
        return source.copy(), {'applied': False, 'reason': 'nonuniform_border'}
    band = min(20, min(h, w)//6)
    regions = [rgb[4:band, 4:-4], rgb[-band:-4, 4:-4],
               rgb[4:-4, 4:band], rgb[4:-4, -band:-4]]
    samples = np.concatenate([region.mean(2).ravel() for region in regions])
    low, paper, high = np.percentile(samples, [5, 50, 95])
    medians = [float(np.median(region)) for region in regions]
    if low < 220 or high-low > 18 or max(medians)-min(medians) > 6:
        return source.copy(), {'applied': False, 'reason': 'nonuniform_border'}
    gray = rgb.mean(2)
    signal = np.maximum(paper-gray, 0)
    noise = max(6., float(paper-np.percentile(samples, 1)))
    core = signal > max(32, noise*2)
    frame = np.zeros(core.shape, bool)
    if core[0].mean() > .9:
        frame[0] = True
    if core[-1].mean() > .9:
        frame[-1] = True
    if core[:, 0].mean() > .9:
        frame[:, 0] = True
    if core[:, -1].mean() > .9:
        frame[:, -1] = True
    core[frame] = False
    if core.mean() < .005:
        return source.copy(), {'applied': False, 'reason': 'nonuniform_border'}
    support = cv2.dilate(core.astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    alpha = np.clip(signal/paper, 0, 1)
    alpha[~support | frame | (signal <= 2)] = 0
    result = np.zeros_like(source)
    result[:, :, 3] = np.rint(alpha*255).astype('uint8')
    return result, {'applied': True, 'reason': 'light_paper_background',
                    'color': [float(paper)]*3,
                    'method': 'neutral_ink_native_coverage',
                    'paper_noise_level': noise,
                    'transparent_pixels': int((result[:, :, 3] == 0).sum())}


def _solid_border_model(source):
    """Consensus across edge strips; lettering may intersect any canvas side.

    A majority color on several sides is evidence for a matte. One corner or
    one edge is insufficient (e.g. a half black / half white image).
    """
    h, w = source.shape[:2]
    band = max(1, min(8, min(h, w)//80))
    regions = [source[:band], source[-band:], source[:, :band], source[:, -band:]]
    sides = [region[:, :, :3][region[:, :, 3] >= 250].astype('float32') for region in regions]
    samples = np.concatenate(sides)
    if len(samples) < 8:
        return None
    color = np.median(samples, axis=0)
    for _ in range(3):
        distances = np.max(np.abs(samples-color), axis=1)
        close = distances <= 8
        if close.mean() < .6:
            return None
        color = np.median(samples[close], axis=0)
    distances = np.max(np.abs(samples-color), axis=1)
    close = distances <= 8
    fractions = [float(np.mean(np.max(np.abs(side-color), axis=1) <= 8))
                 if len(side) else 0. for side in sides]
    if close.mean() < .6 or sum(value >= .3 for value in fractions) < 3:
        return None
    noise = float(np.percentile(distances[close], 99))
    # JPEG/WebP ringing can exceed the flat-strip noise near DCT block edges.
    # Bound the allowance to 8/255; an exact-color matte retains tolerance 2.
    tolerance = 2. if noise == 0 else min(8., np.ceil(noise*2)+4)
    return color, tolerance, fractions, noise


def _extract_uniform_background(source):
    """Estimate edge coverage and undo the old matte in antialiased pixels.

    A single flattened image cannot identify arbitrary foreground/background
    layers. A consistent dominant edge color is required. Existing coverage is
    multiplied by the extracted coverage, never replaced with opaque pixels.
    """
    result = source.copy()
    border_alpha = np.r_[source[0, :, 3], source[-1, :, 3],
                         source[:, 0, 3], source[:, -1, 3]]
    if np.mean(border_alpha < 250) > .5:
        return result, {'applied': False, 'reason': 'existing_transparency'}
    rgb = source[:, :, :3].astype('float32')
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    spread = np.max(np.abs(border-background), axis=1)
    tolerance, consensus = 2., None
    if np.percentile(spread, 98) > 6 or np.median(spread) > 1.5:
        consensus = _solid_border_model(source)
        # Paper grain/illumination needs its own model. A truly flat dominant
        # color takes priority so cropped black lettering does not become gray.
        if consensus is None or consensus[3] > 2.5:
            foreground, status = _remove_light_paper(source)
            if not status['applied']:
                foreground, status = _remove_shaded_paper(source)
            if status['applied']:
                foreground[:, :, 3] = np.rint(foreground[:, :, 3].astype(float)*source[:, :, 3]/255)
                return foreground, status
        if consensus is None:
            return result, {'applied': False, 'reason': 'nonuniform_border'}
        background, tolerance, _, _ = consensus
    elif spread.max() > 0:
        consensus = _solid_border_model(source)
        if consensus is not None:
            background, tolerance, _, _ = consensus
    delta = rgb-background
    difference = np.max(np.abs(delta), axis=2)
    support = (difference > tolerance) & (source[:, :, 3] > 0)
    if tolerance > 2:
        # Compression can create tiny isolated blocks in otherwise blank paper.
        # Remove only small weak islands, retaining long one-pixel strokes.
        count, labels, sizes, _ = cv2.connectedComponentsWithStats(support.astype('uint8'), 8)
        maxima = np.zeros(count, dtype='float32')
        np.maximum.at(maxima, labels.ravel(), difference.ravel())
        weak = (sizes[:, cv2.CC_STAT_AREA] < 8) & (maxima < tolerance*3)
        weak[0] = False
        support[weak[labels]] = False
    if not support.any():
        return result, {'applied': False, 'reason': 'no_foreground'}
    alpha = support.astype('float32')
    core = cv2.erode(support.astype('uint8'), np.ones((3, 3), 'uint8'), iterations=2) > 0
    if tolerance > 2:
        # A JPEG ringing block is not an opaque lettering sample. Using it as
        # the reference leaves colored flecks after the large matte is gone.
        core &= difference > max(16., tolerance*2)
    if core.any():
        distance, nearest = distance_transform_edt(~core, return_indices=True)
        reference = rgb[nearest[0], nearest[1]]-background
        denominator = np.sum(reference*reference, axis=2)
        coverage = np.clip(np.sum(delta*reference, axis=2)/np.maximum(denominator, 1), 0, 1)
        # Correct only the narrow antialias band. Interior colors, fine strokes
        # without a solid core, and decorative texture are never eroded away.
        edge = support & ~core & (distance <= 4) & (denominator > 400)
        alpha[edge] = coverage[edge]
        if tolerance > 2:
            weak = support & ~core & (difference <= max(24., tolerance*3)) & (denominator > 400)
            alpha[weak] = coverage[weak]
            # Beyond the normal antialias fringe, tiny coverage inferred from
            # lossy compression is background ringing, not a new opaque speck.
            alpha[weak & (distance > 3) & (coverage < .15)] = 0
    restored = background+delta/np.maximum(alpha[:, :, None], 1/255)
    result[:, :, :3] = np.rint(np.clip(restored, 0, 255)).astype('uint8')
    result[:, :, 3] = np.rint(alpha*source[:, :, 3]).astype('uint8')
    result[result[:, :, 3] == 0, :3] = 0
    status = {'applied': True, 'reason': 'uniform_background',
                    'color': np.rint(background).astype(int).tolist(),
                    'method': 'border_consensus_with_local_edge_unmixing',
                    'color_tolerance': float(tolerance),
                    'transparent_pixels': int((result[:, :, 3] == 0).sum())}
    if consensus is not None:
        status['border_consensus'] = consensus[2]
    return result, status


def _clean_background_artifacts(foreground):
    """Remove detached peripheral frame lines and debris, protecting lettering.

    No largest-component-only crop: detached glyphs, dots and nearby thin
    strokes must remain. Long neutral lines need both a margin position and
    separation from substantial lettering. Small components need separation
    AND a peripheral position; texture/halos connected to lettering stay.
    """
    h, w = foreground.shape[:2]
    short = min(h, w)
    mask = (foreground[:, :, 3] > 0).astype('uint8')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    anchors = []
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        skinny = max(cw, ch) > 20*min(cw, ch) and min(cw, ch) < short*.04
        if (area >= max(32, h*w*.0015) and min(cw, ch) >= 4
                and area/(cw*ch) > .10 and not skinny):
            anchors.append(label)
    if not anchors:
        return foreground, {'removed_artifact_pixels': 0, 'removed_artifact_components': 0}
    largest = max(stats[label, cv2.CC_STAT_AREA] for label in anchors)
    anchors = [label for label in anchors if stats[label, cv2.CC_STAT_AREA] >= largest*.08]
    lettering_area = sum(int(stats[label, cv2.CC_STAT_AREA]) for label in anchors)
    protected = np.isin(labels, anchors)
    distance = distance_transform_edt(~protected)
    ys, xs = np.where(protected)
    bounds = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    removed = np.zeros((h, w), bool)
    # A dark rectangular screenshot border can be one connected component.
    # Detect its long runs individually, not via component aspect ratio.
    rgb = foreground[:, :, :3].astype('int16')
    neutral = ((np.ptp(rgb, axis=2) <= 24) & (rgb.max(2) < 160)
               & (foreground[:, :, 3] >= 32)).astype('uint8')
    margin = max(3, round(short*.06))
    gap = max(5, short*.015)
    line_mask = np.zeros((h, w), bool)
    horizontal = cv2.morphologyEx(neutral, cv2.MORPH_OPEN,
                                 np.ones((1, max(20, round(w*.65))), 'uint8')) > 0
    vertical = cv2.morphologyEx(neutral, cv2.MORPH_OPEN,
                               np.ones((max(20, round(h*.65)), 1), 'uint8')) > 0
    line_mask[:margin] |= horizontal[:margin]
    line_mask[-margin:] |= horizontal[-margin:]
    line_mask[:, :margin] |= vertical[:, :margin]
    line_mask[:, -margin:] |= vertical[:, -margin:]
    # Remove a run only when all of it is detached; never nibble an actual
    # long stroke where it approaches the lettering.
    nlines, ll, line_stats, _ = cv2.connectedComponentsWithStats(line_mask.astype('uint8'), 8)
    frame_components = 0
    for label in range(1, nlines):
        region = ll == label
        if distance[region].min() > gap:
            removed |= region
            frame_components += 1
    # Recompute after stripping frames, so an inset frame cannot dominate
    # the area threshold for isolated badges/scan flecks.
    remaining = mask.copy()
    remaining[removed] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(remaining, 8)
    debris_components = 0
    separation = max(12, short*.06)
    closest = np.full(count, np.inf)
    np.minimum.at(closest, labels.ravel(), distance.ravel())
    discard = np.zeros(count, bool)
    for label in range(1, count):
        x, y, cw, ch, area = stats[label]
        if area > max(8, lettering_area*.025):
            continue
        # An isolated fine stroke may be a glyph radical or calligraphic
        # accent. Only the separately verified margin-line rule may delete it.
        if max(cw, ch) > short*.08 and max(cw, ch) > 5*min(cw, ch):
            continue
        peripheral = (x+cw <= bounds[0] or x > bounds[2]
                      or y+ch <= bounds[1] or y > bounds[3])
        tiny_speck = area <= max(4, round(short*.004)**2) and max(cw, ch) <= max(3, short*.012)
        if (peripheral or tiny_speck) and closest[label] > separation:
            discard[label] = True
            debris_components += 1
    removed |= discard[labels]
    output = foreground.copy()
    output[removed] = 0
    return output, {'removed_artifact_pixels': int(removed.sum()),
                    'removed_artifact_components': frame_components+debris_components,
                    'removed_frame_components': frame_components,
                    'removed_peripheral_components': debris_components,
                    'cleanup_method': 'detached_margin_lines_and_peripheral_components',
                    'cleanup_scope': 'geometric background debris; nearby lettering and connected effects protected'}


def remove_uniform_background(source):
    foreground, status = _extract_uniform_background(source)
    if status['applied']:
        foreground, cleanup = _clean_background_artifacts(foreground)
        status.update(cleanup)
        status['transparent_pixels'] = int((foreground[:, :, 3] == 0).sum())
    return foreground, status


def verify_foreground_export(foreground, pixels, *, verified_effects=None):
    """Check transparency in space, not just resemblance on the old background.

    Permit a two-pixel curve rasterization fringe. Large opaque canvases, even
    if they look perfect when composited on white, must fail this gate.
    """
    blank = cv2.erode((foreground[:, :, 3] == 0).astype('uint8'),
                      np.ones((5, 5), 'uint8')) > 0
    visible = cv2.erode((foreground[:, :, 3] >= 250).astype('uint8'),
                        np.ones((5, 5), 'uint8')) > 0
    if verified_effects:
        # A source-verified Gaussian halo legitimately extends beyond a hard
        # color threshold. Its body already passed topology/coverage gates in
        # soft_effects.py. Check blank space beyond that bounded effect fringe.
        margin = max(2*effect['sigma']+abs(effect['dx'])+abs(effect['dy'])+2
                     for effect in verified_effects)
        blank &= distance_transform_edt(foreground[:, :, 3] == 0) > margin
        visible[:] = False
    alpha_error = np.abs(pixels[:, :, 3].astype(float)-foreground[:, :, 3])
    leak = float(np.mean(pixels[:, :, 3][blank] > 2)) if blank.any() else 0.
    missing = float(np.mean(pixels[:, :, 3][visible] < 64)) if visible.any() else 0.
    return {'passed': bool(blank.any()) and leak <= .0001 and missing <= .005,
            'background_leak_fraction': leak, 'missing_foreground_fraction': missing,
            'alpha_mae': float(alpha_error.mean()), 'blank_pixels_checked': int(blank.sum()),
            'scope': ('native SVG render; blank space beyond verified Gaussian effects; body verified separately'
                      if verified_effects else
                      'native SVG render; spatial blank mask and missing lettering; 2px curve fringe')}


def remove_traced_background(svg, source, foreground):
    """Keep verified curve geometry while unmatting its original color layers."""
    rgb = source[:, :, :3].reshape(-1, 3)
    rgba = foreground.reshape(-1, 4)
    packed = (rgb[:, 0].astype('uint32') << 16) | (rgb[:, 1].astype('uint32') << 8) | rgb[:, 2]
    # A color shared by an opaque fill and its edge stays opaque. Otherwise use
    # the recovered edge coverage for that color, without moving any path.
    order = np.lexsort((-rgba[:, 3].astype('int16'), packed))
    _, first = np.unique(packed[order], return_index=True)
    selected = order[first]
    palette, paints = rgb[selected], rgba[selected]
    tree = cKDTree(palette)
    root = ET.fromstring(svg)
    mapping = {}
    for parent in root.iter():
        for child in list(parent):
            if child.get('data-name') == '原背景':
                parent.remove(child)
                continue
            color = child.get('fill', '')
            if not color.startswith('#') or len(color) != 7:
                continue
            if color not in mapping:
                value = np.array(list(bytes.fromhex(color[1:])))
                distance, index = tree.query(value)
                mapping[color] = paints[index] if distance <= 6 else None
            paint = mapping[color]
            if paint is None:
                continue
            if not paint[3]:
                parent.remove(child)
            else:
                child.set('fill', '#'+bytes(paint[:3]).hex())
                opacity = float(child.get('fill-opacity', '1'))*int(paint[3])/255
                if opacity < 1:
                    child.set('fill-opacity', f'{opacity:.8f}')
    return ET.tostring(root, encoding='unicode')
