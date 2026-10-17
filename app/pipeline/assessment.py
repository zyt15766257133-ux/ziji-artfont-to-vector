"""按单张需求记录通过、失败、未测和不适用；诊断分数保留各自参考口径。"""
from __future__ import annotations
import math


def assess_result(metrics: dict, quality: dict, elapsed_ms: float | None,
                  input_size: tuple[int, int], *, gradient_required: bool | None = None) -> dict:
    items = {}

    def record(key, label, value, ok=None, *, applicable=True, note=''):
        status = 'not_applicable' if not applicable else 'untested' if ok is None else 'pass' if ok else 'fail'
        items[key] = {'label': label, 'status': status, 'value': value, 'note': note}

    def numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    w, h = input_size
    record('resolution', '输入分辨率 ≥500×500', [w, h], min(w, h) >= 500)
    invalid = quality.get('invalid')
    record('input_quality', '输入背景、清晰度和主体资格', quality.get('invalid_reasons'),
           not invalid if isinstance(invalid, bool) else None,
           note='依据当前预处理器判定，仍须人工核对样本资格。')
    ratio = quality.get('orig_bbox_ratio')
    record('subject_ratio', '裁切前主体占比 ≥60%', ratio,
           ratio >= .60 if numeric(ratio) else None,
           note='当前按提取掩膜外接矩形计算，独立标注及正式占比定义待核定。')
    raw = metrics.get('shape_iou')
    composite_basis = metrics.get('comparison_basis') == 'source_background_composite_contrast_20'
    record('contour_raw', '对原背景合成可见区域的 IoU ≥90%' if composite_basis else '对预处理掩膜的原始 IoU ≥90%', raw,
           raw >= .90 if numeric(raw) else None,
           note='在原背景合成后对双方使用相同的 RGB 色距 >20；原 alpha IoU 单独保留。' if composite_basis else '诊断指标；视觉验收另列。')
    truth = metrics.get('truth_shape_iou')
    record('contour_truth', '对独立真值的 IoU ≥90%', truth,
           truth >= .90 if numeric(truth) else None,
           note='真值边缘定义仍须核定，此项作为独立诊断记录。')
    topology = metrics.get('truth_topology')
    topology_ok = topology.get('passed') if isinstance(topology, dict) else None
    record('truth_topology', '独立真值区域与孔洞位置诊断', topology,
           topology_ok if isinstance(topology_ok, bool) else None,
           note='对齐掩膜逐区一一匹配；默认检查 ≥4 像素区域、区域 IoU ≥85%。诊断阈值，细小特征及视觉仍须补验。')
    feature = metrics.get('feature_retention')
    record('feature_proxy', '综合特征诊断分 ≥85%', feature,
           feature >= .85 if numeric(feature) else None)
    nodes = metrics.get('node_per_glyph')
    known = metrics.get('glyph_source') == 'provided'
    record('node_average', '按真实字数计算的平均节点 50～500', nodes,
           50 <= nodes <= 500 if known and numeric(nodes) else None,
           note='以 SVG 锚点计数；多字平均合格仍须检查逐字分布。')
    glyphs = metrics.get('glyph_count')
    record('nodes_each_glyph', '逐字节点 50～500', nodes if known and glyphs == 1 else None,
           50 <= nodes <= 500 if known and glyphs == 1 and numeric(nodes) else None)
    color = metrics.get('color_deviation')
    record('color', '颜色偏差诊断值 ≤3%', color,
           color <= .03 if numeric(color) else None,
           note='单色主色值或多色主色匹配，颜色位置及面积另列待验。')
    record('color_spatial', '颜色位置、面积和透明度保持', None)
    levels = metrics.get('color_levels_out')
    record('gradient_levels', '渐变色阶 ≥8', levels,
           levels >= 8 if gradient_required is True and numeric(levels) else None,
           applicable=gradient_required is not False,
           note='适用性由独立素材标签指定；自动引擎检测仅作辅助。')
    record('gradient_visual', '渐变自然、无断层和色斑', None,
           applicable=gradient_required is not False)
    fmt = metrics.get('format', {}).get('passed')
    record('svg_structure', 'SVG 标准结构', fmt, fmt if isinstance(fmt, bool) else None,
           note='结构解析结果；实际渲染和编辑另列。')
    record('single_time', '1000×1000 以内完整转换 ≤5 秒', elapsed_ms,
           elapsed_ms <= 5000 if numeric(elapsed_ms) else None,
           applicable=max(w, h) <= 1000,
           note='完整服务调用，含指标及请求启用的预览；不含 HTTP 传输及下载格式导出。')
    internal = metrics.get('checks', {})
    for key in ['holes_preserved', 'no_specks', 'no_breaks', 'spur_ratio<=0.02']:
        value = internal.get(key)
        record('diagnostic_' + key, '内部几何诊断 ' + key, value,
               value if isinstance(value, bool) else None,
               note='内部几何统计，关键孔洞和连接仍须独立区域检查。')
    record('visual_features', '边缘平滑、无变形断笔粘连、关键特征保留', None)
    record('four_browsers', 'Chrome、Firefox、Edge、Safari 渲染一致', None)
    record('illustrator_edit', 'Illustrator 2020+ 打开编辑', None)
    statuses = [v['status'] for v in items.values()]
    overall = 'fail' if 'fail' in statuses else 'untested' if 'untested' in statuses else 'pass'
    return {'scope': 'single_image_checks', 'status': overall, 'items': items,
            'pending_project_checks': ['100 张合规不同素材批量', '10 次颜色及结构一致性', '非技术用户首次转换', '部署与交付文档']}
