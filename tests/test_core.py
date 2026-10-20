"""核心口径的自动化测试：指标规则、XML 合法性、节点统计、分组解析、导出校验。

运行：python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.export import validate_export, svg_to_eps, svg_to_pdf  # noqa: E402
from app.pipeline.metrics import (_checks, _line_single_color, _redundant_anchor_ratio,  # noqa: E402
                                  count_holes, format_check)
from app.pipeline.pathparse import SvgRaster, parse_svg_layers  # noqa: E402
from app.pipeline.svgutil import SvgDocument, SvgLayer  # noqa: E402
from app.pipeline.vectorize import _match_spot  # noqa: E402


def make_doc(grouped: bool = False) -> SvgDocument:
    doc = SvgDocument(width=100, height=100)
    doc.layers.append(SvgLayer(color="#ff0000", paths=["M10 10H50V50H10Z", "M20 20H30V30H20Z"]))
    doc.layers.append(SvgLayer(color="#00ff00", paths=["M60 60H90V90H60Z"]))
    if grouped:
        doc.glyph_groups = [[(0, 0)], [(0, 1), (1, 0)]]
        doc.glyph_labels = ["甲", "乙"]
    return doc


def test_svg_well_formed_and_escaped():
    doc = make_doc()
    doc.title = 'a<b&c"d> test'
    svg = doc.to_svg()
    assert format_check(svg)["passed"]


def test_title_with_ampersand_parses():
    doc = SvgDocument(width=10, height=10)
    doc.title = "x&y<z"
    doc.layers.append(SvgLayer(color="#000000", paths=["M0 0H10V10H0Z"]))
    import xml.etree.ElementTree as ET
    ET.fromstring(doc.to_svg())   # 非法字符已转义，可解析


def test_glyph_grouped_parse_matches_flat():
    flat = make_doc(False).to_svg()
    grouped = make_doc(True).to_svg()
    _, lf = parse_svg_layers(flat)
    _, lg = parse_svg_layers(grouped)
    # 分组会把同色层拆到不同字组，比较总路径数与渲染结果即可
    assert sum(len(l["d"]) for l in lf) == sum(len(l["d"]) for l in lg)
    assert {l["fill"] for l in lf} == {l["fill"] for l in lg}
    rf, rg = SvgRaster(flat), SvgRaster(grouped)
    a = rf.rasterize(scale=1)[:, :, 3] > 0
    b = rg.rasterize(scale=1)[:, :, 3] > 0
    assert (a ^ b).sum() == 0      # 分组结构不改变渲染结果


def test_path_d_not_confused_with_id():
    # id="p1" 里含 d=" 子串，解析必须取真正的 d 属性
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" '
           'viewBox="0 0 100 100"><g id="glyph-1"><g fill="#ff0000">'
           '<path id="p1" d="M10 10H50V50H10Z"/></g></g></svg>')
    _, layers = parse_svg_layers(svg)
    assert layers[0]["d"][0][0].startswith("M10 10")


def test_node_count():
    from app.pipeline.svgutil import count_nodes_from_d
    assert count_nodes_from_d("M0 0C1 2 3 4 5 6C7 8 9 10 11 12Z") == 3
    assert count_nodes_from_d("M0 0L10 0L10 10Z") == 3


def test_checks_single_rule_and_holes():
    base = {"contour_fidelity": 0.95, "feature_retention": 0.9,
            "node_per_glyph": 300, "holes_preserved": True,
            "color_mode": "single", "color_deviation": 0.01,
            "speck_count": 0, "stroke_breaks": 0, "spur_ratio": 0.0}
    assert _checks(dict(base, passed=False))["passed"] is True
    # 节点下限
    assert not _checks(dict(base, node_per_glyph=30))["node_50_500"]
    # 节点上限唯一规则：单层口径不再放宽
    assert not _checks(dict(base, node_per_glyph=600, node_per_layer_glyph=100))["node_50_500"]
    # 内孔不允许少
    assert not _checks(dict(base, holes_preserved=False))["holes_preserved"]
    # 颜色口径
    assert "color_dev<=0.03" in _checks(dict(base))
    assert "color_dev<=0.08" in _checks(dict(base, color_mode="multi"))


def test_line_single_color():
    bg = (255.0, 255.0, 255.0)
    ink = ((10.0, 10.0, 10.0), 0.5)
    mix = ((132.0, 132.0, 132.0), 0.3)      # 墨色与背景的混色
    off = ((255.0, 214.0, 0.0), 0.4)        # 离线的黄色
    assert _line_single_color([ink, mix], bg) is True
    assert _line_single_color([ink, mix, off], bg) is False


def test_spot_match():
    assert _match_spot("#1a1a1a", ["#191919"]) == ("#191919", "#191919")
    color, hit = _match_spot("#ff0000", ["#191919"])
    assert hit is None and color == "#ff0000"
    assert _match_spot("#ff0000", ["#191919"], force=True) == ("#191919", "#191919")


def test_redundant_anchor_ratio():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
           '<g fill="#000"><path d="M0 0C0.1 0 9.9 0 10 0C10 0.1 10 9.9 10 10'
           'C10 10 0 10 0 10C0 10 0 0 0 0Z"/></g></svg>')
    assert _redundant_anchor_ratio(svg) == 1.0   # 全部近直线


def test_count_holes_min_area():
    mask = np.zeros((60, 60), np.uint8)
    mask[5:55, 5:55] = 255
    mask[20:24, 20:24] = 0        # 16px² 小孔，低于 24px² 不计
    mask[30:40, 30:42] = 0        # 120px² 大孔，计入
    assert count_holes(mask.astype(bool)) == 1


def test_export_validation():
    svg = make_doc(False).to_svg()
    pdf = svg_to_pdf(svg)
    eps = svg_to_eps(svg)
    assert validate_export(pdf, "pdf")["checks"]["passed"]
    assert validate_export(eps, "eps")["checks"]["passed"]
    assert validate_export(b"not a pdf", "pdf")["checks"]["passed"] is False


def test_radial_gradient_raster():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" viewBox="0 0 40 40">'
           '<radialGradient id="g" gradientUnits="userSpaceOnUse" cx="20" cy="20" r="20">'
           '<stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#000000"/>'
           '</radialGradient>'
           '<g fill="url(#g)"><path d="M0 0H40V40H0Z"/></g></svg>')
    out = SvgRaster(svg).rasterize(scale=1)
    center = out[20, 20, :3]
    corner = out[1, 1, :3]
    assert center.mean() > corner.mean() + 60   # 中心亮、边缘暗的径向插值生效
