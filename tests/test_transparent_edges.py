"""Regressions for source coverage, crisp corners and transparency semantics."""
from pathlib import Path
import cv2
import numpy as np

from app.pipeline.direct import monochrome_document
from app.pipeline.native_render import render_svg
from app.pipeline.preprocess import preprocess
from app.pipeline.service import convert_bytes
from app.pipeline.svgutil import fit_contour, contour_to_path_d
from app.pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parents[1]


def test_transparent_lettering_retains_antialiasing_in_real_conversion():
    result = convert_bytes('18_transparent.png', (ROOT/'samples/images/18_transparent.png').read_bytes(),
                           VectorizeOptions(glyph_count=3), strict_input=False, with_preview=True)
    assert not result.error
    assert result.stats.get('source_coverage')
    source = cv2.imdecode(np.frombuffer(result.source_png, np.uint8), cv2.IMREAD_UNCHANGED)[:, :, 3]/255.
    output = render_svg(result.svg)[:, :, 3]/255.
    # Compare the coverage transition, not only the filled area: binary overlap
    # hid the original visible staircase even when global IoU exceeded 98%.
    kernel = np.ones((3, 3), np.uint8)
    band = (cv2.dilate((source > 0).astype(np.uint8), kernel) > 0)
    band &= cv2.erode((source > .99).astype(np.uint8), kernel) == 0
    assert np.abs(source-output)[band].mean() < .06
    assert result.metrics['shape_iou'] >= .98
    assert 50 <= result.metrics['node_per_glyph'] <= 500
    assert result.metrics['holes_preserved']
    assert '<image' not in result.svg and 'filter=' not in result.svg


def _prepared_rgba(alpha, rgb=(0, 122, 255)):
    rgba = np.empty((*alpha.shape, 4), np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = alpha
    return preprocess(rgba[:, :, [2, 1, 0, 3]])


def test_invisible_rgb_does_not_determine_transparent_lettering_geometry():
    original = ('<svg xmlns="http://www.w3.org/2000/svg" width="180" height="140">'
                '<path fill="#007aff" fill-rule="evenodd" d="M12 18H168V122H12Z M50 48H130V92H50Z"/></svg>')
    pixels = render_svg(original)
    a = pixels.copy()
    b = pixels.copy()
    a[a[:, :, 3] == 0, :3] = (0, 122, 255)
    b[b[:, :, 3] == 0, :3] = (170, 36, 212)
    docs = []
    for image in (a, b):
        prepared = preprocess(image[:, :, [2, 1, 0, 3]])
        result = monochrome_document(prepared, VectorizeOptions())
        assert result is not None
        docs.append(result[0].to_svg())
    assert docs[0] == docs[1]
    alpha = render_svg(docs[0])[:, :, 3]
    assert alpha[70, 90] == 0 and alpha[30, 90] == 255


def test_translucent_fill_and_soft_halo_are_not_replaced_by_solid_silhouette():
    alpha = np.zeros((160, 200), np.uint8)
    alpha[20:140, 25:175] = 190
    assert monochrome_document(_prepared_rgba(alpha), VectorizeOptions()) is None
    alpha[20:140, 25:175] = 255
    soft = cv2.GaussianBlur(alpha, (0, 0), 4)
    assert monochrome_document(_prepared_rgba(soft), VectorizeOptions()) is None


def test_all_corner_contour_remains_a_rectangle_after_smoothing():
    contour = np.array([[20, 20], [180, 20], [180, 120], [20, 120]], dtype=float)
    fit = fit_contour(contour, tolerance=.5, corner_angle=62, smooth=True)
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="140">'
           f'<path d="{contour_to_path_d(fit)}"/></svg>')
    alpha = render_svg(svg)[:, :, 3] > 127
    reference = np.zeros((140, 200), bool)
    reference[20:120, 20:180] = True
    assert np.logical_xor(alpha, reference).sum() <= 4
