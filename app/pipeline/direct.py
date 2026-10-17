"""Source-pixel silhouettes with bounded cubic approximation for clean lettering."""
from __future__ import annotations
import cv2
import numpy as np
from .curvefit import fit_ring
from .svgutil import SvgDocument, SvgLayer, FittedContour, contour_to_path_d, polygon_area, rgb_to_hex


def fit_mask_paths(mask, tolerance=.6, corner_angle=62, smooth=True, geometric=False, snap_deg=0):
    """Fit boundaries, using coverage values when available instead of binarizing first."""
    values = np.asarray(mask)
    if values.dtype == bool or (values.size and values.max() <= 1):
        values = values.astype(np.uint8) * 255
    else:
        values = values.astype(np.uint8)
    coverage = bool(((values > 0) & (values < 255)).any())
    # Keep source antialiasing: interpolation locates the 50% coverage contour at
    # subpixel positions. Nearest-neighbor enlargement only magnifies pixel stairs.
    up = cv2.resize(values, None, fx=4, fy=4,
                    interpolation=cv2.INTER_LINEAR if coverage else cv2.INTER_NEAREST)
    if geometric and coverage:
        # findContours follows foreground pixel centres, which systematically
        # thins strokes. Project those samples back to the actual 50% level set.
        field = up.astype(float)
        gy, gx = np.gradient(field)
    if coverage:
        up = (up > 127).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(up, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return [], 0
    pieces = {}
    nodes = 0
    for index, contour in enumerate(contours):
        points = (contour.reshape(-1, 2).astype(float) + .5) / 4
        if geometric and coverage:
            xy = contour.reshape(-1,2)
            x,y = xy[:,0],xy[:,1]
            gradient = np.column_stack([gx[y,x],gy[y,x]])
            norm2 = np.sum(gradient*gradient,axis=1)
            step = np.divide(127.5-field[y,x],norm2,out=np.zeros(len(x)),where=norm2>1e-8)
            displacement = gradient*step[:,None]
            displacement *= np.minimum(1,1.5/np.maximum(np.linalg.norm(displacement,axis=1),1e-8))[:,None]
            points += displacement/4
        if len(points) < 3:
            continue
        if coverage:
            # Suppress quarter-pixel extraction steps before estimating tangents;
            # the deviation is smaller than one input pixel and retains real corners.
            points = cv2.approxPolyDP(points.astype(np.float32), .18, True).reshape(-1, 2)
        if geometric:
            from .geometry import fit_geometric_ring
            def fitter(p, t, c):
                return fit_geometric_ring(p, t, c, snap_deg)
        else:
            fitter = fit_ring
        segments = fitter(points, tolerance, corner_angle) if smooth else [
            (points[i], None, None, points[(i + 1) % len(points)]) for i in range(len(points))]
        if not segments:
            continue
        nodes += len(segments) + 1
        pieces[index] = contour_to_path_d(FittedContour(segments, points, abs(polygon_area(points))))
    paths = []
    for index, entry in enumerate(hierarchy[0]):
        depth, parent = 0, int(entry[3])
        while parent >= 0:
            depth += 1
            parent = int(hierarchy[0][parent][3])
        if depth % 2 or index not in pieces:
            continue
        rings = [pieces[index]]
        child = int(entry[2])
        while child >= 0:
            if child in pieces:
                rings.append(pieces[child])
            child = int(hierarchy[0][child][0])
        paths.append(' '.join(rings))
    return paths, nodes


def fit_mask(mask, tolerance=.6):
    paths, nodes = fit_mask_paths(mask, tolerance)
    return ' '.join(paths), nodes


def monochrome_document(prepared, options):
    transparent = prepared.source_rgba is not None and bool(np.any(prepared.source_rgba[:, :, 3] < 250))
    if (not prepared.quality.background_uniform or prepared.quality.textured_background
            or options.spot_colors or (prepared.bg_color is None and not transparent)):
        return None
    raw = prepared.rgb_raw.astype(float)
    bg = np.asarray(prepared.bg_color) if prepared.bg_color is not None else None
    if transparent:
        support = prepared.source_rgba[:, :, 3] > 127
    else:
        support = np.linalg.norm(raw - bg, axis=2) > 20
    core = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    if core.sum() < 100:
        return None
    colors = raw[core]
    ink = np.median(colors, axis=0)
    if transparent:
        # Transparent pixels have no background colour. Their RGB can equal the
        # ink (or be arbitrary); geometry comes from source alpha, not RGB contrast.
        source_alpha = prepared.source_rgba[:, :, 3]
        # A solid silhouette can preserve antialiasing, but not a translucent fill
        # or a wide soft shadow. Keep those in the existing layered-effect engine.
        if np.percentile(source_alpha[core], 5) < 250:
            return None
        partial = (source_alpha > 8) & (source_alpha < 247)
        inside = cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
        outside = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
        if partial.any() and np.max((inside + outside)[partial]) > 2.5:
            return None
        visible = source_alpha > 16
        if np.percentile(np.linalg.norm(raw[visible]-ink, axis=1), 95) > 3:
            return None
        boundary = source_alpha
    else:
        direction = ink-bg
        denom = float(direction@direction)
        if denom < 1000:
            return None
        alpha = np.clip((raw-bg)@direction/denom, 0, 1)
        if np.abs(raw[core]-(bg+alpha[:, :, None]*direction)[core]).mean() > 2:
            return None
        boundary = support
    # Broad tonal modulation needs a separate texture representation.
    if np.percentile(np.linalg.norm(colors-ink, axis=1), 95) > 10:
        return None
    glyphs = options.glyph_count or 1
    from .native_render import render_svg
    from .topology import compare_topology
    # A binary overlap score alone accepts visible edge errors. Transparent
    # lettering gets a tighter bound against its original coverage samples.
    start_tolerance = max(.12, min(.3, float(options.tolerance)))
    tolerances = [start_tolerance, start_tolerance*.7, start_tolerance*.45] if transparent else [.75, .55, .35]
    for tol in tolerances:
        paths, nodes = fit_mask_paths(boundary, tol, options.corner_angle, options.smooth)
        if not paths or nodes > options.max_nodes*glyphs or (options.glyph_count and nodes < 50*glyphs):
            continue
        doc = SvgDocument(support.shape[1], support.shape[0])
        doc.layers.append(SvgLayer(rgb_to_hex(ink), paths, label='lettering'))
        pixels = render_svg(doc.to_svg())
        output = pixels[:, :, 3] > 127
        iou = np.logical_and(support, output).sum()/max(1, np.logical_or(support, output).sum())
        topology = compare_topology(support, output)
        if iou >= .97 and topology['passed']:
            return doc, {'engine':'contour', 'source_silhouette':True, 'node_total':nodes,
                         'glyphs':glyphs, 'layers':1, 'source_iou':round(float(iou),4),
                         'source_coverage':transparent, 'tolerance_used':tol,
                         'source_topology_passed':True}
    return None
