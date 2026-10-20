"""Independent SVG references for effect inference and local quality gates."""
from pathlib import Path
import io
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import pytest

from app.pipeline.native_render import render_svg
from app.pipeline.soft_effects import soft_effect_candidate, region_metrics
from app.pipeline.fidelity import source_error
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parents[1]
RING = 'M45 45H150V190H45Z M65 65V170H130V65Z'
CURVE = 'M245 48C355 48 355 188 245 188L245 167C322 167 322 69 245 69Z'


def reference(*, transparent=False, multicolor=False, dx=0, dy=0, second_opacity=.6):
    groups = [(RING, '#25b8f2', 7), (CURVE, '#ed288d' if multicolor else '#25b8f2', 7)]
    if not multicolor:
        groups = [(RING+' '+CURVE, '#25b8f2', 7)]
    defs, paint = [], []
    for i, (path, color, sigma) in enumerate(groups):
        defs.append(f'<filter id="f{i}" filterUnits="userSpaceOnUse" x="0" y="0" width="400" height="240" '
                    f'color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="{sigma}"/>'
                    f'<feOffset dx="{dx}" dy="{dy}"/></filter>')
        opacity = second_opacity if i == 1 else .6
        paint.append(f'<g filter="url(#f{i})" fill="{color}" fill-rule="evenodd" opacity="{opacity}"><path d="{path}"/></g>')
    background = '' if transparent else '<path fill="#0b1019" d="M0 0H400V240H0Z"/>'
    body = f'<g fill="#f4f4f4" fill-rule="evenodd"><path d="{RING} {CURVE}"/></g>'
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">'
           f'<defs>{"".join(defs)}</defs>{background}{"".join(paint)}{body}</svg>')
    body_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="400" height="240">{body}</svg>'
    return render_svg(svg), render_svg(body_svg)[:, :, 3]/255.


@pytest.mark.parametrize('transparent,dx,dy', [(False, 0, 0), (False, 8, -5), (True, -5, 6)])
def test_infers_source_effects_preserving_holes_and_alpha(transparent, dx, dy):
    source, truth = reference(transparent=transparent, dx=dx, dy=dy)
    snapshot = source.copy()
    candidate, diagnostic = soft_effect_candidate(source)
    assert candidate is not None, diagnostic
    svg, pixels, stats, coverage, background = candidate
    assert diagnostic['body_topology']['passed']
    assert diagnostic['body_iou'] > .98
    assert stats['node_total'] < 1000
    assert '<image' not in svg and '<text' not in svg
    assert 'feGaussianBlur' in svg
    root = ET.fromstring(svg)
    linked = next(e for e in root.iter() if e.get('id') == 'lettering-compound')
    assert linked.get('filter')
    assert 'feComponentTransfer' not in svg
    assert stats['effect_editing']['follows_body']
    assert all(e['alpha_cutoff'] == 0 for e in stats['effects'])
    local = region_metrics(source, pixels, truth, background)
    assert local['halo_mae'] < 1.5, local
    assert local['halo_p95'] < 4, local
    assert abs(local['halo_luma_bias']) < 1, local
    assert abs(stats['effects'][0]['dx']-dx) < 1
    assert abs(stats['effects'][0]['dy']-dy) < 1
    assert source_error(source, pixels)['active_mae'] < 2
    np.testing.assert_array_equal(snapshot, source)


def test_separates_different_glow_colors_instead_of_one_global_gradient():
    source, truth = reference(multicolor=True)
    candidate, diagnostic = soft_effect_candidate(source)
    assert candidate is not None, diagnostic
    _, pixels, stats, _, background = candidate
    assert len(stats['effects']) == 2
    assert stats['complexity']['protect_weak_halo']
    colors = [e['color'] for e in stats['effects']]
    assert len(set(colors)) == 2
    assert stats['effect_editing']['representation'] == 'independent_emitters'
    assert stats['effect_editing']['follows_body'] is False
    root = ET.fromstring(candidate[0])
    for f in root.findall('.//{*}filter'):
        assert [e.tag.rsplit('}', 1)[-1] for e in f] == ['feGaussianBlur']
    assert region_metrics(source, pixels, truth, background)['halo_mae'] < 1.5


def test_low_brightness_second_color_is_not_deleted_during_reduction():
    source, truth = reference(multicolor=True, second_opacity=.06)
    candidate, diagnostic = soft_effect_candidate(source)
    assert candidate is not None, diagnostic
    assert len(candidate[2]['effects']) == 2
    local = region_metrics(source, candidate[1], truth, candidate[4])
    assert all(region['relative_error'] < .3 for region in local['regions']), local


def test_small_sharp_decoration_is_not_sacrificed_to_a_large_smooth_halo():
    source, _ = reference()
    source[90:93, 181:203] = [244, 30, 50, 255]
    candidate, diagnostic = soft_effect_candidate(source)
    assert candidate is None, diagnostic


def test_transparent_hidden_rgb_does_not_change_inferred_effects():
    source, _ = reference(transparent=True)
    alternate = source.copy()
    alternate[source[:, :, 3] == 0, :3] = [221, 17, 124]
    a, da = soft_effect_candidate(source)
    b, db = soft_effect_candidate(alternate)
    assert a is not None and b is not None, (da, db)
    assert a[0] == b[0]


def test_neon_service_replaces_halo_and_records_original_candidate_comparison():
    p = ROOT/'samples/images/21_neon.png'
    result = convert_bytes(p.name, p.read_bytes(), VectorizeOptions.from_preset('fidelity'),
                           strict_input=False, with_preview=False)
    diagnostic = result.stats['effect_compensation']
    assert diagnostic['selected'], diagnostic
    compare = diagnostic['comparison']
    assert not compare['rejected']
    assert compare['candidate_local']['halo_mae'] < compare['baseline_local']['halo_mae']*.5
    assert abs(compare['candidate_local']['halo_luma_bias']) < 1
    assert result.stats['node_total'] < diagnostic['baseline_nodes']*.02
    assert result.metrics['source_transform']['crop_box'] is None
    assert result.timings['total_ms'] == result.elapsed_ms
    assert result.timings['decode_validate_ms'] >= 0
    assert result.stats['node_total'] < 1000
    assert result.stats['effect_editing']['follows_body']
    assert result.stats['background_removal']['applied']
    assert render_svg(result.svg)[0, 0, 3] == 0
    assert '原背景' not in result.svg


def test_linked_shadow_export_retains_body_holes_and_distinct_halo_color():
    from app.pipeline.soft_effects import _document, Effect
    from app.pipeline.filter_export import expand_filters
    from app.pipeline.export import svg_to_pdf
    body = 'M40 40H160V160H40Z M70 70V130H130V70Z'
    effect = Effect(np.ones((200, 200)), 6, 3, -2, np.array([0., .65, 1.]), .6)
    svg = _document(200, 200, [body], np.ones(3), [effect], np.zeros(4))
    expanded = expand_filters(svg)
    output = render_svg(expanded)
    assert 'filter=' not in expanded and '<image' not in expanded
    np.testing.assert_array_equal(output[55, 55], [255, 255, 255, 255])
    assert output[100, 100, 3] == 0
    assert output[35, 100, 3] > 0
    assert output[35, 100, 2] > output[35, 100, 0] + 100
    pdf = svg_to_pdf(svg)
    assert pdf.startswith(b'%PDF-')
    assert b'/Subtype /Image' not in pdf


def test_http_upload_and_svg_download_use_the_same_linked_result():
    from fastapi.testclient import TestClient
    from app.main import app
    source, _ = reference(transparent=True, dx=5, dy=-3)
    buf = io.BytesIO()
    Image.fromarray(source).save(buf, format='PNG')
    # Deliberately avoid a built-in filename: the route must infer from pixels.
    client = TestClient(app)
    response = client.post('/api/convert', files={'file': ('uploaded-glow.png', buf.getvalue(), 'image/png')},
                           data={'preset': 'fidelity', 'with_preview': 'false'})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['stats']['effect_compensation']['selected'], result['stats']
    assert result['stats']['effect_editing']['follows_body']
    download = client.post('/api/export', data={'svg': result['svg'], 'fmt': 'svg', 'stem': result['stem']})
    assert download.status_code == 200, download.text
    assert download.content.decode('utf-8') == result['svg']
    assert render_svg(download.content.decode('utf-8'))[0, 0, 3] == 0


def test_bypass_and_pixel_archive_never_call_effect_inference(monkeypatch):
    import app.pipeline.soft_effects as effects
    def unexpected(*args, **kwargs):
        raise AssertionError('effect inference must be bypassed')
    monkeypatch.setattr(effects, 'soft_effect_candidate', unexpected)
    source, _ = reference()
    buf = io.BytesIO()
    Image.fromarray(source).save(buf, format='PNG')
    for preset in ['fidelity', 'pixel_exact']:
        result = convert_bytes('reference.png', buf.getvalue(),
                               VectorizeOptions.from_preset(preset, soft_effects=False),
                               strict_input=False, with_preview=False)
        assert not result.error
        assert 'effect_compensation' not in result.stats


def test_api_bypass_is_honored_instead_of_silently_using_default(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    import app.pipeline.soft_effects as effects
    def unexpected(*args, **kwargs):
        raise AssertionError('API soft_effects=false must bypass inference')
    monkeypatch.setattr(effects, 'soft_effect_candidate', unexpected)
    source, _ = reference()
    buf = io.BytesIO()
    Image.fromarray(source).save(buf, format='PNG')
    client = TestClient(app)
    response = client.post('/api/convert', files={'file': ('reference.png', buf.getvalue(), 'image/png')},
                           data={'preset': 'fidelity', 'soft_effects': 'false', 'with_preview': 'false'})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['options']['soft_effects'] is False
    assert 'effect_compensation' not in result['stats']


def test_optional_numerical_failure_keeps_verified_tracer_output(monkeypatch):
    import app.pipeline.soft_effects as effects
    source, _ = reference()
    buf = io.BytesIO()
    Image.fromarray(source).save(buf, format='PNG')
    baseline = convert_bytes('source.png', buf.getvalue(),
                             VectorizeOptions.from_preset('fidelity', soft_effects=False), False, False)
    def fail(*args):
        raise ValueError('fit did not converge')
    monkeypatch.setattr(effects, 'soft_effect_candidate', fail)
    result = convert_bytes('source.png', buf.getvalue(), VectorizeOptions.from_preset('fidelity'), False, False)
    assert result.svg == baseline.svg
    assert result.stats['effect_compensation']['reason'] == 'numerical_fit_failed'


@pytest.mark.parametrize('name', ['10_gradient.png', '14_brush_feibai.png', '18_transparent.png'])
def test_unsupported_material_is_not_replaced_by_a_soft_effect(name):
    source = np.asarray(Image.open(ROOT/'samples/images'/name).convert('RGBA'))
    candidate, diagnostic = soft_effect_candidate(source)
    assert candidate is None, diagnostic
