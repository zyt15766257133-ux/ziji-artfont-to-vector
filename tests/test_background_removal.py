"""Independent alpha and original-image checks for automatic plain-background removal."""
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.background import background_canvas, remove_uniform_background
from app.pipeline.fidelity import source_error, trace_curves
from app.pipeline.native_render import render_svg
from app.pipeline.service import convert_bytes, inspect_bytes
from app.pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parents[1]


def unseen_solid_scene(background):
    """Known foreground, including a letter crossing the top canvas boundary."""
    truth = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="192" height="160">'
                       '<path fill="#a42d63" fill-rule="evenodd" '
                       'd="M25 0H85V130H25Z M42 40V92H68V40Z"/>'
                       '<path fill="#238baa" d="M110 25H165V135H110Z"/>'
                       '<path fill="#432574" d="M95 45H96V118H95Z"/></svg>')
    alpha = truth[:, :, 3:4]/255.
    source = np.full_like(truth, 255)
    source[:, :, :3] = np.rint(truth[:, :, :3]*alpha+np.array(background)*(1-alpha))
    return source, truth


@pytest.mark.parametrize('background', [(255, 255, 255), (9, 12, 18), (35, 170, 91), (230, 184, 52)])
@pytest.mark.parametrize('format', ['PNG', 'JPEG', 'WEBP'])
def test_unseen_solid_colors_with_touching_letters_and_compression(background, format):
    import cv2
    source, truth = unseen_solid_scene(background)
    stream = io.BytesIO()
    Image.fromarray(source[:, :, :3]).save(stream, format=format, quality=85)
    decoded = np.asarray(Image.open(io.BytesIO(stream.getvalue())).convert('RGBA'))
    foreground, status = remove_uniform_background(decoded)
    assert status['applied'], status
    # Independently known blank region, excluding compression/antialias fringe.
    blank = cv2.erode((truth[:, :, 3] == 0).astype('uint8'), np.ones((19, 19), 'uint8')) > 0
    assert (foreground[:, :, 3][blank] == 0).mean() > .995
    # Also inspect close to glyphs: distant whitespace misses JPEG ringing.
    near_blank = cv2.erode((truth[:, :, 3] == 0).astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    assert (foreground[:, :, 3][near_blank] > 32).mean() < .0005
    assert foreground[65, 55, 3] == 0  # Enclosed counter.
    assert foreground[10, 45, 3] == 255  # Cropped letter remains opaque.
    assert np.abs(foreground[60, 130, :3].astype(int)-truth[60, 130, :3]).max() <= 12
    if format == 'PNG':
        np.testing.assert_array_equal(foreground, truth)


def test_stray_transparent_pixel_does_not_disable_white_background_removal():
    source, truth = unseen_solid_scene((255, 255, 255))
    source[0, 0] = [0, 0, 0, 0]
    source[70, 130, 3] = 170
    truth[70, 130, 3] = 170
    output, status = remove_uniform_background(source)
    assert status['applied']
    np.testing.assert_array_equal(output, truth)


@pytest.mark.parametrize('background', [(255, 255, 255), (35, 170, 91)])
def test_unseen_upload_svg_remains_transparent_after_tracing(background):
    source, truth = unseen_solid_scene(background)
    stream = io.BytesIO()
    Image.fromarray(source[:, :, :3]).save(stream, format='JPEG', quality=85)
    data = stream.getvalue()
    inspection = inspect_bytes('never-in-sample-manifest.jpg', data)
    result = convert_bytes('never-in-sample-manifest.jpg', data,
                           VectorizeOptions.from_preset('fidelity'), False, False)
    assert not result.error
    assert inspection['background_removal']['applied']
    output = render_svg(result.svg)
    assert result.stats['preserves_background'] is False
    assert output[65, 55, 3] == output[145, 180, 3] == 0
    assert result.metrics['background_removal']['export_verification']['passed']
    assert source_error(truth, output)['active_mae'] < 12


def shaded_paper_scene():
    """Independent ink alpha and paper field; no product-generated truth."""
    h, w = 180, 240
    y, x = np.mgrid[:h, :w].astype(float)
    x /= w-1
    y /= h-1
    paper = 254-22*y*y-4*x*y
    alpha = np.zeros((h, w))
    alpha[40:145, 35:90] = 1
    alpha[40:145, 135:190] = .65
    alpha[65:110, 50:75] = 0  # Interior hole must lose the same gradient.
    alpha[45:130, 110] = .8  # Isolated single-pixel line.
    alpha[120:132:2, 140:185] = .3  # Fine tonal texture.
    source = np.full((h, w, 4), 255, dtype='uint8')
    source[:, :, :3] = np.rint(paper*(1-alpha))[:, :, None]
    source[:3, :, :3] = 20
    source[:, :2, :3] = 20
    return source, alpha, paper


def test_shaded_paper_and_thin_frame_preserve_independent_ink_coverage():
    source, alpha, paper = shaded_paper_scene()
    snapshot = source.copy()
    output, status = remove_uniform_background(source)
    assert status['reason'] == 'shaded_light_paper_background'
    assert status['removed_frame_pixels'] == 3*240+2*(180-3)
    expected = np.zeros_like(source)
    expected[:, :, 3] = np.rint(alpha*255)
    assert np.abs(output[:, :, 3].astype(int)-expected[:, :, 3]).max() <= 1
    assert np.abs(background_canvas(status, source.shape[:2])[:, :, 0]-paper).max() < 1
    assert np.all(output[:30, :, 3] == 0)
    assert np.all(output[65:110, 50:75, 3] == 0)
    np.testing.assert_array_equal(source, snapshot)


@pytest.mark.parametrize('format,extension', [('PNG', 'png'), ('JPEG', 'jpg'), ('WEBP', 'webp')])
def test_new_upload_formats_share_background_detection_and_export(format, extension):
    source, _, _ = shaded_paper_scene()
    stream = io.BytesIO()
    Image.fromarray(source[:, :, :3]).save(stream, format=format, quality=95)
    data = stream.getvalue()
    decoded = np.array(Image.open(io.BytesIO(data)).convert('RGBA'))
    expected, _ = remove_uniform_background(decoded)
    inspection = inspect_bytes('unseen-upload.'+extension, data)
    result = convert_bytes('unseen-upload.'+extension, data,
                           VectorizeOptions.from_preset('fidelity'), False, False)
    assert not result.error
    assert inspection['background_removal']['reason'] == 'shaded_light_paper_background'
    assert inspection['quality_basis'] == 'detected_foreground_native_resolution'
    assert 'no_foreground' not in inspection['quality']['invalid_reasons']
    assert 'blurry' not in inspection['quality']['invalid_reasons']
    assert inspection['quality']['orig_bbox_ratio'] > .3
    assert result.metrics['background_removal']['reason'] == inspection['background_removal']['reason']
    assert result.stats['preserves_background'] is False
    actual = render_svg(result.svg)
    assert source_error(expected, actual)['premultiplied_max_error'] <= 1
    assert actual[85, 60, 3] == actual[5, 5, 3] == 0
    assert '<image' not in result.svg and 'data:image' not in result.svg


def test_unrecognized_background_is_reported_as_not_removed():
    source = np.full((80, 100, 3), 255, dtype='uint8')
    source[:, :50] = 0
    stream = io.BytesIO()
    Image.fromarray(source).save(stream, format='PNG')
    result = convert_bytes('unseen-complex.png', stream.getvalue(),
                           VectorizeOptions.from_preset('fidelity'), False, False)
    assert not result.error
    assert result.metrics['background_removal']['reason'] == 'nonuniform_border'
    assert result.stats['preserves_background'] is True
    assert result.quality['severity'] == 'warning'
    assert result.quality['passed'] is False
    assert 'background_removal_failed' in result.quality['invalid_reasons']
    assert '自动去背景未成功' in result.warnings[0]


@pytest.mark.parametrize('kind', ['abrupt', 'colored', 'grain'])
def test_shaded_paper_model_rejects_backgrounds_it_cannot_explain(kind):
    source, _, _ = shaded_paper_scene()
    if kind == 'abrupt':
        source[:, 120:, :3] = np.minimum(source[:, 120:, :3], 220)
    elif kind == 'colored':
        source[:, :, 0] = np.maximum(source[:, :, 0].astype(int)-20, 0)
    else:
        rng = np.random.default_rng(18342)
        source[:, :, :3] = np.clip(source[:, :, :3].astype(int)+
            rng.integers(-30, 31, (*source.shape[:2], 1)), 0, 255)
    actual, status = remove_uniform_background(source)
    assert status['reason'] == 'nonuniform_border'
    np.testing.assert_array_equal(actual, source)


@pytest.mark.parametrize('background', [(12, 12, 16), (255, 255, 255)])
def test_unmattes_edges_without_filling_holes_or_changing_solid_color(background):
    truth = render_svg('<svg xmlns="http://www.w3.org/2000/svg" width="180" height="150">'
                       '<path fill="#318cc7" fill-rule="evenodd" '
                       'd="M20.3 20.3H159.7V129.7H20.3Z M60 50V100H120V50Z"/>'
                       '<path fill="#318cc7" d="M5 40H6V110H5Z"/></svg>')
    alpha = truth[:, :, 3:4]/255.
    source = np.empty_like(truth)
    source[:, :, :3] = np.rint(truth[:, :, :3]*alpha+np.array(background)*(1-alpha))
    source[:, :, 3] = 255
    snapshot = source.copy()
    foreground, status = remove_uniform_background(source)
    assert status['applied']
    assert foreground[0, 0, 3] == foreground[75, 90, 3] == 0
    assert foreground[60, 5, 3] == 255  # Isolated one-pixel stroke.
    np.testing.assert_array_equal(foreground[30, 80], truth[30, 80])
    assert source_error(truth, foreground)['active_mae'] < 1.5
    np.testing.assert_array_equal(source, snapshot)


def test_existing_alpha_and_nonuniform_background_are_not_guessed():
    source = np.full((80, 100, 4), 255, dtype='uint8')
    source[:, :, 3] = 0
    source[20:60, 20:80] = [20, 80, 140, 170]
    output, status = remove_uniform_background(source)
    assert status['reason'] == 'existing_transparency'
    np.testing.assert_array_equal(output, source)
    source[:, :, 3] = 255
    source[:, :50, :3] = 0
    output, status = remove_uniform_background(source)
    assert status['reason'] == 'nonuniform_border'
    np.testing.assert_array_equal(output, source)


def test_light_paper_grain_is_removed_without_smoothing_tonal_strokes():
    rng = np.random.default_rng(18026)
    source = np.full((180, 240, 4), 255, dtype='uint8')
    paper = np.clip(np.rint(rng.normal(239, 3, (180, 240))), 0, 255).astype('uint8')
    source[:, :, :3] = paper[:, :, None]
    # Independently specified one-pixel dark/light stripes in two letter bars.
    source[45:135, 45:80, :3] = np.where(np.arange(90)[:, None, None] % 2, 25, 125)
    source[45:135, 140:175, :3] = np.where(np.arange(90)[:, None, None] % 2, 25, 125)
    source[0, :, :3] = 26  # Scan frame must not determine the paper color.
    snapshot = source.copy()
    foreground, status = remove_uniform_background(source)
    assert status['reason'] == 'light_paper_background'
    assert status['applied']
    assert np.all(foreground[:30, :, 3] == 0)
    opacity = foreground[:, :, 3:4]/255.
    restored = foreground[:, :, :3]*opacity+np.array(status['color'])*(1-opacity)
    assert np.max(np.abs(restored[45:135, 45:80]-source[45:135, 45:80, :3])) < 1
    np.testing.assert_array_equal(snapshot, source)


def test_paper_route_preserves_native_coverage_in_export():
    source = np.full((100, 140, 4), 255, dtype='uint8')
    rng = np.random.default_rng(34)
    source[:, :, :3] = np.rint(rng.normal(239, 3, (100, 140, 1)))
    source[30:75, 40:100, :3] = 40
    source[40:65, 60:80, :3] = 239  # Letter hole.
    stream = io.BytesIO()
    Image.fromarray(source).save(stream, format='PNG')
    expected, _ = remove_uniform_background(source)
    result = convert_bytes('paper.png', stream.getvalue(), VectorizeOptions.from_preset('fidelity'), False, False)
    actual = render_svg(result.svg)
    assert not result.error
    assert result.stats['representation'] == 'pixel_grid'
    assert result.stats['preserves_background'] is False
    assert result.metrics['background_removal']['reason'] == 'light_paper_background'
    assert actual[50, 70, 3] == actual[5, 5, 3] == 0
    assert source_error(expected, actual)['premultiplied_max_error'] <= 1
    assert 'data:image' not in result.svg


def test_reported_gradient_sample_exports_transparency_and_keeps_original_preview():
    path = ROOT/'samples/images/16_gradient_stroke.png'
    original = np.asarray(Image.open(path).convert('RGBA'))
    result = convert_bytes(path.name, path.read_bytes(), VectorizeOptions.from_preset('fidelity'),
                           with_preview=True, strict_input=False)
    assert not result.error
    output = render_svg(result.svg)
    assert result.stats['background_removal']['applied']
    assert result.stats['preserves_background'] is False
    assert output[0, 0, 3] == output[-1, -1, 3] == 0
    assert (output[:, :, 3] == 0).mean() > .8
    assert '原背景' not in result.svg
    baseline = trace_curves(original, soft_effects=False)[3]
    assert result.metrics['source_fidelity']['active_mae'] <= baseline['active_mae']+.05
    assert result.metrics['source_fidelity']['pixels_over_8'] <= baseline['pixels_over_8']+100
    assert 'composited on original background' in result.metrics['source_fidelity']['scope']
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(result.source_png))), original)
    assert not any(token in result.svg for token in ['<image', 'data:image'])


def test_white_background_is_removed_and_pixel_archive_stays_exact():
    source = np.full((80, 100, 4), 255, dtype='uint8')
    source[20:60, 20:80] = [18, 50, 80, 255]
    stream = io.BytesIO()
    Image.fromarray(source).save(stream, format='PNG')
    for preset in ['fidelity', 'pixel_exact']:
        result = convert_bytes('white.png', stream.getvalue(), VectorizeOptions.from_preset(preset),
                               with_preview=False, strict_input=False)
        output = render_svg(result.svg)
        if preset == 'fidelity':
            assert output[0, 0, 3] == 0
            np.testing.assert_array_equal(output[40, 50], source[40, 50])
        else:
            np.testing.assert_array_equal(output, source)
