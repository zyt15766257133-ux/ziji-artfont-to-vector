"""Independent debris masks, plus the user's original screenshot regression."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.background import _extract_uniform_background, remove_uniform_background
from app.pipeline.native_render import render_svg
from app.pipeline.service import convert_bytes, inspect_bytes
from app.pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parents[1]


def debris_scene(background, *, inset_frame=False, grain=False):
    # Dot, detached one-pixel stroke and a cropped glyph are real artwork.
    truth = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="512">'
                       '<path fill="#a42d63" fill-rule="evenodd" '
                       'd="M120 90H255V370H120Z M150 150H225V280H150Z"/>'
                       '<path fill="#2378aa" d="M350 130H465V380H350Z"/>'
                       '<path fill="#a42d63" d="M160 63H172V75H160Z"/>'
                       '<path fill="#432574" d="M100 135H101V330H100Z"/></svg>')
    a = truth[:, :, 3:4]/255.
    rgb = np.rint(truth[:, :, :3]*a+np.array(background)*(1-a)).astype('uint8')
    if grain:
        rng = np.random.default_rng(20261018)
        blank = truth[:, :, 3] == 0
        rgb[blank] = np.clip(rgb[blank].astype(int)+rng.integers(-2, 3, (blank.sum(), 1)), 0, 255)
    debris = np.zeros(truth.shape[:2], bool)
    if inset_frame:
        debris[5:7, 7:633] = debris[503:506, 7:633] = True
        debris[5:506, 7:9] = debris[5:506, 631:633] = True
    else:
        debris[:2, :] = debris[-6:, :] = True
    debris[450:470, 290:345] = True  # Independent footer badge.
    debris[28:31, 42:45] = debris[440:442, 560:562] = True  # Scan flecks.
    debris[250:252, 302:304] = True  # Dirt between two glyphs, within their shared bbox.
    rgb[debris] = 24
    return rgb, truth, debris


@pytest.mark.parametrize('background,grain', [((255, 255, 255), False),
                                            ((244, 242, 235), True),
                                            ((35, 170, 91), False)])
@pytest.mark.parametrize('inset_frame', [False, True])
def test_removes_detached_frames_badges_and_grain_without_erasing_dot_or_thin_stroke(background, grain, inset_frame):
    source, truth, debris = debris_scene(background, inset_frame=inset_frame, grain=grain)
    rgba = np.array(Image.fromarray(source).convert('RGBA'))
    snapshot = rgba.copy()
    foreground, status = remove_uniform_background(rgba)
    assert status['applied']
    assert np.all(foreground[:, :, 3][debris] == 0)
    assert status['removed_artifact_pixels'] >= int(debris.sum())
    np.testing.assert_array_equal(foreground[truth[:, :, 3] > 0], truth[truth[:, :, 3] > 0])
    np.testing.assert_array_equal(rgba, snapshot)


def test_actual_reported_screenshot_removes_both_black_edges_and_footer_preserving_lettering():
    path = ROOT/'samples/images/屏幕截图 2026-10-18 200055.png'
    source = np.array(Image.open(path).convert('RGBA'))
    snapshot = path.read_bytes()
    before, _ = _extract_uniform_background(source)
    after, status = remove_uniform_background(source)
    assert after.shape == source.shape
    assert not after[:2, :, 3].any()
    assert not after[580:, :, 3].any()
    assert not after[518:550, 385:476, 3].any()
    np.testing.assert_array_equal(after[95:480, 45:815], before[95:480, 45:815])
    assert status['removed_frame_components'] == 2
    assert status['removed_peripheral_components'] == 1
    assert path.read_bytes() == snapshot


def test_nearby_underline_and_connected_decoration_are_not_mistaken_for_frame():
    source, truth, _ = debris_scene((255, 255, 255))
    # A long real underline is close to the lettering, outside the margin.
    source[389:391, 85:520] = [30, 30, 30]
    output, _ = remove_uniform_background(np.array(Image.fromarray(source).convert('RGBA')))
    assert np.all(output[389:391, 85:520, 3] == 255)
    assert np.all(output[truth[:, :, 3] == 255, 3] == 255)


def test_new_upload_inspection_and_final_svg_share_artifact_cleanup():
    source, truth, debris = debris_scene((244, 242, 235), inset_frame=True, grain=True)
    stream = io.BytesIO()
    Image.fromarray(source).save(stream, format='PNG')
    data = stream.getvalue()
    inspection = inspect_bytes('new-unlisted-grain-and-frame.png', data)
    result = convert_bytes('new-unlisted-grain-and-frame.png', data,
                           VectorizeOptions.from_preset('fidelity'), True, False)
    assert not result.error
    assert inspection['background_removal']['removed_artifact_pixels'] > 0
    output = render_svg(result.svg)
    assert not output[:, :, 3][debris].any()
    assert np.mean(output[:, :, 3][truth[:, :, 3] == 255] < 240) < .005
    assert result.metrics['background_removal']['export_verification']['passed']
    assert result.metrics['source_fidelity']['excluded_background_artifact_pixels'] > 0
    np.testing.assert_array_equal(np.array(Image.open(io.BytesIO(result.source_png)))[:, :, :3], source)
    assert '<image' not in result.svg
