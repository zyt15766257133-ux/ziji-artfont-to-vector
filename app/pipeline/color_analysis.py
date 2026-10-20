"""Read-only representative colors from source pixels, before any recoloring."""
from __future__ import annotations

import numpy as np
from PIL import Image


def source_colors(rgb: np.ndarray, alpha: np.ndarray, *, scope: str = 'foreground') -> dict:
    """Summarize at most eight colors; ignore hidden RGB and weight soft edges.

    The palette describes representative source colors, not an exact count of
    all colors in a gradient. Sampling and median cut bound work on large images.
    """
    result = {'colors': [], 'scope': scope, 'approximate': True, 'source': 'original_image'}
    visible = alpha > 0
    pixels = rgb[visible]
    if not len(pixels):
        return result
    weights = alpha[visible].astype(np.float64) / 255
    if len(pixels) > 65536:
        indices = np.linspace(0, len(pixels) - 1, 65536, dtype=np.int64)
        pixels, weights = pixels[indices], weights[indices]
    # Quantize only a temporary palette image. Source pixels and SVG are untouched.
    indexed = Image.fromarray(pixels.reshape(1, -1, 3)).quantize(
        colors=16, method=Image.Quantize.MEDIANCUT)
    labels = np.asarray(indexed).reshape(-1)
    totals = np.bincount(labels, weights=weights, minlength=16)
    sums = np.array([np.bincount(labels, weights=weights * pixels[:, i], minlength=16)
                     for i in range(3)]).T
    groups = []
    for index in np.argsort(-totals, kind='stable'):
        weight = totals[index]
        if weight <= 0:
            continue
        color = sums[index] / weight
        for group in groups:
            if np.linalg.norm(color - group[0] / group[1]) <= 24:
                group[0] += sums[index]
                group[1] += weight
                break
        else:
            groups.append([sums[index].copy(), float(weight)])
    groups.sort(key=lambda group: -group[1])
    for total, weight in groups:
        share = weight / weights.sum()
        if share < .01 and result['colors']:
            continue
        color = np.rint(total / weight).astype(int).tolist()
        result['colors'].append({'hex': '#%02X%02X%02X' % tuple(color),
                                 'rgb': color, 'share': round(float(share), 4)})
        if len(result['colors']) == 8:
            break
    return result
