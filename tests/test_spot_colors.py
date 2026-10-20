"""Exercise the optional color branch through real engines and API validation."""
import io
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, build_options
from app.pipeline.fidelity import apply_spot_colors, pixel_paths
from app.pipeline.native_render import render_svg
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions, _match_spot


def image_bytes():
    source = np.full((160, 160, 4), 255, dtype=np.uint8)
    source[14:146, 14:78, :3] = [220, 30, 50]
    source[14:146, 82:146, :3] = [25, 65, 205]
    buffer = io.BytesIO()
    Image.fromarray(source).save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.mark.parametrize('preset,engine', [('fidelity', 'auto'), ('balanced', 'contour'), ('balanced', 'vtracer')])
def test_real_engines_only_apply_palette_when_enabled(preset, engine):
    opts = VectorizeOptions.from_preset(preset, engine=engine, soft_effects=False, auto_verify=False)
    data = image_bytes()
    baseline = convert_bytes('color.png', data, opts, False, False)
    opts.spot_colors = ['#00ff00']
    disabled = convert_bytes('color.png', data, opts, False, False)
    assert not baseline.error and not disabled.error
    assert baseline.svg == disabled.svg
    assert 'spot_matching' not in disabled.stats
    assert opts.spot_colors == ['#00ff00']  # The caller's settings are preserved.
    opts.spot_matching = True
    enabled = convert_bytes('color.png', data, opts, False, False)
    assert not enabled.error
    assert baseline.metrics['source_colors'] == disabled.metrics['source_colors'] == enabled.metrics['source_colors']
    palette = baseline.metrics['source_colors']
    assert palette['scope'] == 'foreground'
    assert {color['hex'] for color in palette['colors']} == {'#DC1E32', '#1941CD'}
    assert enabled.stats['spot_matching']['matched_layers'] > 0
    output = render_svg(enabled.svg)
    assert output[80, 45, :3].tolist() == [0, 255, 0]
    assert output[80, 115, :3].tolist() == [0, 255, 0]
    assert output[0, 0, 3] == 0


def test_fidelity_pixel_coverage_fallback_can_be_recolored(monkeypatch):
    from app.pipeline import background
    source = np.zeros((25, 30, 4), dtype=np.uint8)
    source[3:22, 3:27] = [200, 20, 20, 255]
    # Force the real paper branch to exercise its native pixel-coverage SVG.
    monkeypatch.setattr(background, 'remove_uniform_background', lambda _: (
        source.copy(), {'applied': True, 'color': [255, 255, 255],
                        'reason': next(iter(background.PAPER_BACKGROUNDS))}))
    buffer = io.BytesIO()
    Image.fromarray(source).save(buffer, format='PNG')
    opts = VectorizeOptions.from_preset('fidelity', spot_matching=True, spot_colors=['#0f0'])
    result = convert_bytes('paper.png', buffer.getvalue(), opts, False, False)
    assert result.stats['representation'] == 'pixel_grid'
    assert result.stats['spot_matching']['matched_layers'] > 0
    output = render_svg(result.svg)
    assert output[10, 10, :3].tolist() == [0, 255, 0]
    np.testing.assert_array_equal(output[:, :, 3], source[:, :, 3])


def test_pixel_exact_ignores_enabled_matching_in_library_and_api():
    opts = VectorizeOptions.from_preset('pixel_exact', spot_matching=True, spot_colors=['invalid'])
    result = convert_bytes('archive.png', image_bytes(), opts, False, False)
    assert 'spot_matching' not in result.stats
    assert result.metrics['source_fidelity']['native_grid_exact']
    palette = result.metrics['source_colors']
    assert palette['scope'] == 'visible_image'
    assert {color['hex'] for color in palette['colors']} == {'#FFFFFF', '#DC1E32', '#1941CD'}
    api_opts = build_options('pixel_exact', spot_matching=True, spot_colors='invalid')
    assert not api_opts.spot_matching and api_opts.spot_colors is None


def test_svg_gradient_definitions_survive_palette_matching():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
           '<defs><linearGradient id="g"><stop stop-color="#f00"/>'
           '<stop offset="1" stop-color="#00f"/></linearGradient></defs>'
           '<path fill="#123456" fill-opacity="0.4" d="M0 0H20V20H0Z"/>'
           '<path fill="url(#g)" d="M20 0H40V20H20Z"/></svg>')
    mapped, stats = apply_spot_colors(svg, ['#0f0'])
    before, after = render_svg(svg), render_svg(mapped)
    np.testing.assert_array_equal(after[:, 20:], before[:, 20:])
    np.testing.assert_array_equal(after[:, :, 3], before[:, :, 3])
    assert stats['matched_layers'] == 1
    assert stats['palette'] == ['#00ff00']
    assert any(e.get('fill') == 'url(#g)' for e in ET.fromstring(mapped).iter())


def test_nearest_palette_rule_is_shared_by_both_pipelines():
    palette = ['#800000', '#606000']
    # Squared RGB distance selects the first; max-channel distance selects the second.
    assert _match_spot('#000000', palette, force=True)[0] == '#800000'
    source = np.full((3, 3, 4), [0, 0, 0, 255], dtype=np.uint8)
    svg, _ = pixel_paths(source)
    mapped, _ = apply_spot_colors(svg, palette)
    assert render_svg(mapped)[1, 1, :3].tolist() == [128, 0, 0]


@pytest.mark.parametrize('route,field', [('/api/convert', 'file'), ('/api/jobs', 'file'), ('/api/batch', 'files')])
@pytest.mark.parametrize('palette', ['', '#xyzxyz', '#12345', '#123456,bad'])
def test_invalid_enabled_palette_returns_422_before_conversion(route, field, palette):
    client = TestClient(app)
    response = client.post(route, files={field: ('color.png', image_bytes(), 'image/png')},
                           data={'preset': 'fidelity', 'spot_matching': 'true', 'spot_colors': palette})
    assert response.status_code == 422, response.text
    assert 'HEX' in response.json()['detail'] or '色值格式' in response.json()['detail']


def test_palette_normalization_and_disabled_palette_validation():
    opts = build_options('fidelity', spot_matching=True, spot_colors='#F00, #ff0000, #00F')
    assert opts.spot_colors == ['#ff0000', '#0000ff']
    assert build_options('fidelity', spot_matching=False, spot_colors='bad').spot_colors is None
