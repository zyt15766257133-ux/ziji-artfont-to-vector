"""学习模块的坐标、分块和数据隔离契约。"""
import json

import numpy as np
import torch

from app.pipeline.native_render import render_svg
from .infer import mask_to_svg
from .model import GlyphUNet, features, predict
from .make_dataset import ROOT


def test_split_isolation():
    manifest = json.loads((ROOT / "data/paired/manifest.json").read_text(encoding="utf-8"))
    chars = [set(v) for v in manifest["char_splits"].values()]
    assert not chars[0] & chars[1]
    assert not chars[0] & chars[2]
    assert not chars[1] & chars[2]
    assert not set(manifest["splits"]["train"]["fonts"]) & set(manifest["splits"]["unseen_font"]["fonts"])
    hashes = []
    for name in manifest["splits"]:
        rows = json.loads((ROOT / f"data/paired/{name}.json").read_text(encoding="utf-8"))
        current = {r["image_sha256"] for r in rows}
        assert all(not current & h for h in hashes)
        hashes.append(current)


def test_small_prediction_matches_training_forward():
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    rgb = rng.integers(0, 256, (127, 113, 3), dtype=np.uint8)
    model = GlyphUNet().eval()
    actual = predict(model, rgb)
    assert actual.shape == (127, 113)
    assert np.isfinite(actual).all()
    assert actual.min() >= 0 and actual.max() <= 1
    rgb = rgb[:124, :112]
    with torch.inference_mode():
        expected = model(torch.from_numpy(features(rgb))[None]).sigmoid()[0, 0].numpy()
    np.testing.assert_allclose(predict(model, rgb), expected)


def test_tiling_preserves_shape_and_prior_without_seams():
    torch.set_num_threads(2)
    rng = np.random.default_rng(8)
    rgb = rng.integers(0, 256, (173, 291, 3), dtype=np.uint8)
    model = GlyphUNet().eval()  # 初始网络 head=0，因此应与整图固定 prior 完全一致。
    actual = predict(model, rgb)
    with torch.inference_mode():
        expected = torch.sigmoid((torch.from_numpy(features(rgb)[4])-.1)*12).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_svg_keeps_original_canvas_and_hole():
    alpha = np.zeros((128, 160), np.float32)
    alpha[31:99, 52:120] = 1
    alpha[48:82, 69:103] = 0
    svg, _ = mask_to_svg(alpha)
    rendered = render_svg(svg)[:, :, 3] > 127
    assert rendered.shape == alpha.shape
    truth = alpha > .5
    assert (rendered & truth).sum() / (rendered | truth).sum() > .97
    assert not rendered[64, 85]
