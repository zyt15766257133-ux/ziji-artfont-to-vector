"""Recover crisp flat lettering and a separately fitted, soft offset shadow.

This is a conservative candidate for uniform ink on an opaque, uniform backdrop.
It infers geometry from pixels, never from a font or the sample's text/name.
Unexplained colours, texture, topology changes and poor reconstructions fall back.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import shift
from .direct import fit_mask_paths
from .native_render import render_svg
from .svgutil import SvgDocument, SvgLayer, rgb_to_hex
from .topology import compare_topology


def _shadow(mask, parameters):
    dx,dy,sigma,opacity = parameters
    blurred = cv2.GaussianBlur(mask,(0,0),float(sigma),borderType=cv2.BORDER_CONSTANT)
    return shift(blurred,(dy,dx),order=1,prefilter=False,mode='constant')*opacity


def shadow_document(prepared, options):
    if (options.shadow_layers == 'off' or options.shadow_style != 'soft'
            or options.spot_colors or options.contour_grow or not options.smooth
            or prepared.bg_color is None or not prepared.quality.background_uniform
            or prepared.quality.textured_background or prepared.rgb_raw is None):
        return None
    if prepared.source_rgba is not None and np.any(prepared.source_rgba[:,:,3] < 250):
        return None
    raw = prepared.rgb_raw.astype(float)
    bg = np.asarray(prepared.bg_color,dtype=float)
    distance = np.linalg.norm(raw-bg,axis=2)
    visible = distance > 20
    if visible.sum() < 200:
        return None
    far = np.percentile(distance[visible],90)
    ink = np.median(raw[distance >= far],axis=0)
    direction = ink-bg
    norm = float(direction@direction)
    if norm < 10000:
        return None
    density = np.clip((raw-bg)@direction/norm,0,1)
    residual = np.linalg.norm(raw-(bg+density[:,:,None]*direction),axis=2)
    if np.percentile(residual[visible],95) > 4:
        return None
    body = density > .65
    solid = density > .9
    if solid.sum() < 100 or np.percentile(np.linalg.norm(raw[solid]-ink,axis=1),90) > 4:
        return None
    outside = cv2.dilate(body.astype(np.uint8),np.ones((5,5),np.uint8)) == 0
    soft = outside & (density > .025) & (density < .6)
    if soft.sum() < max(150,body.sum()*.08):
        return None
    # Fit both visible shadow and empty pixels. Including the latter prevents a
    # model from extending an unsupported shadow beyond the observed footprint.
    y,x = np.where(outside)
    stride = max(1,len(y)//6000)
    y,x = y[::stride],x[::stride]
    max_shift = min(64., min(body.shape)/3.)
    max_sigma = min(32., min(body.shape)/6.)
    if max_sigma <= 1:
        return None
    seed = body.astype(float)
    fit = least_squares(lambda p:_shadow(seed,p)[y,x]-density[y,x],
                        [0,0,min(5.,max_sigma),.4],
                        bounds=([-max_shift,-max_shift,1,.04],
                                [max_shift,max_shift,max_sigma,.65]),
                        diff_step=.02,max_nfev=40)
    if not fit.success:
        return None
    predicted = _shadow(seed,fit.x)
    shadow_error = np.abs(predicted-density)[outside & (density > .015)]
    if not len(shadow_error) or shadow_error.mean() > .015 or np.percentile(shadow_error,95) > .045:
        return None
    # Undo foreground-over-shadow compositing before extracting the 50% edge.
    # Thresholding the whole image, or eroding the body, loses stroke thickness.
    coverage = np.clip((density-predicted)/np.maximum(1-predicted,.1),0,1)
    coverage[coverage < .03] = 0
    reference = coverage > .5
    edge = (cv2.dilate(reference.astype(np.uint8),np.ones((3,3),np.uint8))
            != cv2.erode(reference.astype(np.uint8),np.ones((3,3),np.uint8)))
    from .vectorize import estimate_glyph_count
    glyphs = options.glyph_count or estimate_glyph_count(reference.astype(np.uint8)*255)
    cap = options.max_nodes*max(1,glyphs)
    tolerance = float(np.clip(options.tolerance*.6,.18,.48))
    dx,dy,sigma,opacity = map(float,fit.x)
    for tol in (tolerance,tolerance*.75):
        paths,nodes = fit_mask_paths(np.rint(coverage*255).astype(np.uint8),tol,
                                    options.corner_angle,geometric=True,snap_deg=options.snap_deg)
        if not paths or nodes*2 > cap:
            continue
        doc = SvgDocument(body.shape[1],body.shape[0])
        color = rgb_to_hex(ink)
        # Group the entire shadow as one filter input: per-path blurs would
        # accumulate opacity at overlaps and change the original appearance.
        doc.defs = [f'<filter id="inferred-shadow" filterUnits="userSpaceOnUse" '
                    f'x="0" y="0" width="{doc.width}" height="{doc.height}" '
                    f'color-interpolation-filters="sRGB">'
                    f'<feGaussianBlur stdDeviation="{sigma:.4f}"/>'
                    f'<feOffset dx="{dx:.4f}" dy="{dy:.4f}"/></filter>']
        doc.layers = [SvgLayer(color,paths,label='shadow',opacity=opacity,filter_id='inferred-shadow'),
                      SvgLayer(color,paths,label='lettering')]
        body_doc = SvgDocument(doc.width,doc.height,layers=[doc.layers[1]])
        output_coverage = render_svg(body_doc.to_svg())[:,:,3]/255.
        output_body = output_coverage > .5
        iou = np.logical_and(reference,output_body).sum()/max(1,np.logical_or(reference,output_body).sum())
        area_ratio = output_coverage.sum()/max(1,coverage.sum())
        edge_error = float(np.abs(output_coverage-coverage)[edge].mean())
        if iou < .975 or not .985 <= area_ratio <= 1.015 or edge_error > .12:
            continue
        if not compare_topology(reference,output_body)['passed']:
            continue
        pixels = render_svg(doc.to_svg())
        alpha = pixels[:,:,3:4]/255.
        painted = pixels[:,:,:3]*alpha+bg*(1-alpha)
        error = np.abs(painted-raw).mean(2)/255.
        if error[visible].mean() > .018 or np.percentile(error[visible],95) > .09:
            continue
        return doc, {'engine':'contour','source_shadow_model':True,'glyphs':int(glyphs),
                     'layers':2,'shadow_layers':1,'node_total':nodes*2,'body_nodes':nodes,
                     'tolerance_used':tol,'body_iou':round(float(iou),5),
                     'body_area_ratio':round(float(area_ratio),5),
                     'body_edge_coverage_mae':round(edge_error,5),
                     'composite_mae':round(float(error[visible].mean()),5),
                     'shadow':{'style':'soft','model':'offset-gaussian','dx':dx,'dy':dy,
                               'sigma':sigma,'opacity':opacity,
                               'fit_mae':float(shadow_error.mean())}}
    return None
