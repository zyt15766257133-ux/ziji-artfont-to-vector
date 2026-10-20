"""Recognize source colors without changing source pixels or recolored output."""
import numpy as np

from app.pipeline.color_analysis import source_colors


def test_known_colors_and_transparent_pixels_are_preserved():
    rgb = np.array([[[255, 220, 0]] * 7 + [[0, 0, 0]] * 3 + [[0, 255, 0]]], dtype=np.uint8)
    alpha = np.array([[255] * 10 + [0]], dtype=np.uint8)
    original_rgb, original_alpha = rgb.copy(), alpha.copy()
    palette = source_colors(rgb, alpha)
    assert palette['colors'] == [
        {'hex': '#FFDC00', 'rgb': [255, 220, 0], 'share': .7},
        {'hex': '#000000', 'rgb': [0, 0, 0], 'share': .3},
    ]
    assert palette['source'] == 'original_image'
    np.testing.assert_array_equal(rgb, original_rgb)
    np.testing.assert_array_equal(alpha, original_alpha)


def test_soft_edge_alpha_weights_color_area():
    rgb = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    palette = source_colors(rgb, np.array([[255, 85]], dtype=np.uint8))
    assert [(c['hex'], c['share']) for c in palette['colors']] == [('#FF0000', .75), ('#0000FF', .25)]


def test_transparent_input_has_no_colors():
    assert source_colors(np.full((3, 4, 3), 255, dtype=np.uint8),
                         np.zeros((3, 4), dtype=np.uint8))['colors'] == []


def test_gradient_has_representative_colors_instead_of_exact_color_count():
    gradient = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
    palette = source_colors(gradient, np.full((1, 256), 255, dtype=np.uint8))
    assert palette['approximate']
    assert 2 <= len(palette['colors']) <= 8
    assert all(len(set(c['rgb'])) == 1 for c in palette['colors'])
