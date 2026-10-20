"""矢量化引擎：自研轮廓引擎 + vtracer 引擎，接口一致，可插拔替换。"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, asdict, field, replace

import cv2
import numpy as np

from .pathparse import parse_path
from .preprocess import PreparedImage, preprocess
from .svgutil import (FittedContour, SvgDocument, SvgLayer, build_linear_gradient,
                      build_radial_gradient, contour_to_path_d, fit_contour,
                      offset_polygon, polygon_area, rgb_to_hex)


@dataclass
class VectorizeOptions:
    engine: str = "auto"            # auto | contour | vtracer
    mode: str = "balanced"          # quality | balanced | compact | 自定义
    max_colors: int = 12            # 单图最大颜色层数
    tolerance: float = 0.5          # 轮廓拟合容差（像素）
    max_nodes: int = 500            # 单字形节点上限
    min_nodes: int = 6
    corner_angle: float = 62.0      # 角点判定阈值（度）
    smooth: bool = True
    gradient_levels: int = 10       # 渐变层数（影响 vtracer layer_difference）
    color_precision: int = 6
    filter_speckle: int = 4
    enforce_node_budget: bool = True
    soft_effects: bool = True       # 保真档：独立拟合光晕／软阴影，局部复核后采用
    transparent_background: bool = True
    posterize: int = 0              # >0 时先做色阶合并，保证渐变分级
    auto_verify: bool = True        # 第三方引擎（vtracer）兜底：自检不达标时对比取优
    contour_grow: float = 0.0       # 轮廓外扩量（像素），亚像素取轮廓后默认不再偏移
    contour_coordinate_offset: float = 0.0
    refine_source_edges: bool = True
    preserve_source_gaps: bool = False
    glyph_count: int | None = None  # 真实字数（manifest/用户手填），不传时连通域估算兜底
    glyph_text: str | None = None   # 原图文字（manifest），用于按字分组时的语义命名
    spot_matching: bool = False     # 显式开启专色匹配；默认保留原色
    spot_colors: list[str] | None = None  # 用户指定 HEX 色板，开启后按 RGB 最近色匹配纯色层
    snap_axis: str = "auto"         # 直线/正交吸附：auto（按素材判定）| on | off
    shadow_layers: str = "auto"     # 阴影分层（柔边阴影独立低节点层）：auto | on | off
    shadow_style: str = "soft"      # 阴影风格：soft（软覆盖率半透明层）| steps（实色阶梯，印刷/刻字）
    snap_deg: float = 0.0           # 解析后的吸附角容差（度，0=关），resolve_profile 写入

    PRESETS = {
        "fidelity": {"filter_speckle": 0, "enforce_node_budget": False},
        "pixel_exact": {"filter_speckle": 0, "enforce_node_budget": False},
        "quality":  {"tolerance": 0.25, "max_colors": 16, "gradient_levels": 14,
                     "color_precision": 7, "filter_speckle": 2, "corner_angle": 55.0},
        "balanced": {"tolerance": 0.5, "max_colors": 12, "gradient_levels": 10,
                     "color_precision": 6, "filter_speckle": 4, "corner_angle": 62.0},
        "compact":  {"tolerance": 0.8, "max_colors": 8, "gradient_levels": 8,
                     "color_precision": 5, "filter_speckle": 8, "corner_angle": 70.0},
    }

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "VectorizeOptions":
        opts = cls()
        preset = cls.PRESETS.get(name)
        if preset:
            for k, v in preset.items():
                setattr(opts, k, v)
            opts.mode = name
        for k, v in overrides.items():
            if v is not None and hasattr(opts, k):
                setattr(opts, k, v)
        return opts

    def to_dict(self) -> dict:
        return asdict(self)

    def resolve_spot_options(self) -> "VectorizeOptions":
        if not self.spot_matching or self.mode == 'pixel_exact':
            return replace(self, spot_matching=False, spot_colors=None)
        from .spot_colors import normalize_palette
        return replace(self, spot_colors=normalize_palette(self.spot_colors))


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------

def estimate_glyph_count(mask: np.ndarray, min_area_ratio: float = 0.002) -> int:
    """以连通域数量估计“字数”，仅作无真实字数时的兜底（预算分母）。

    真实字数应由调用方通过 VectorizeOptions.glyph_count 传入（manifest 或手填）：
    连通域会把飞白、装饰点算成额外“字”，高估分母会让单字节点预算虚高。
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = areas.sum()
    keep = int((areas >= max(16, total * min_area_ratio)).sum())
    return max(1, keep)


def axis_alignment_score(mask: np.ndarray, crisp: np.ndarray | None = None) -> float:
    """轮廓长边里近水平/近竖直的占比，区分几何字与书法手写。

    黑体/宋体/Logo 这类几何字横平竖直，得分高，适合直线吸附；
    书法/手写曲线多，得分低，吸附会伤笔意，应关掉。
    步长取点避免像素锯齿抖动把方向打乱。crisp 传入时只统计主体
    （高对比）轮廓，柔边阴影不会把得分稀释掉。
    """
    use = mask
    if crisp is not None:
        c = (crisp > 0) & (mask > 0)
        if int(c.sum()) >= 0.2 * max(1, int((mask > 0).sum())):
            use = c.astype(np.uint8) * 255
    contours, _ = cv2.findContours((use > 0).astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    t = math.tan(math.radians(8.0))
    hit = total = 0.0
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < 12:
            continue
        if abs(polygon_area(pts)) < 220:
            continue
        stride = max(2, len(pts) // 80)
        v = np.roll(pts, -stride, axis=0) - pts
        length = np.hypot(v[:, 0], v[:, 1])
        sel = length >= 9
        if not sel.any():
            continue
        dx, dy = v[sel, 0], v[sel, 1]
        ls = length[sel]
        axis_hit = (np.abs(dy) <= np.abs(dx) * t) | (np.abs(dx) <= np.abs(dy) * t)
        hit += float(ls[axis_hit].sum())
        total += float(ls.sum())
    return hit / total if total > 0 else 0.0


def _detect_soft_shadow(prepared: PreparedImage, labels: np.ndarray,
                        palette: np.ndarray,
                        opts: VectorizeOptions) -> tuple:
    """按颜色层归并出阴影/光晕区域（含碎层吸附与平滑），返回 (渲染版, 剔除版)。

    柔边投影被色阶量化后会碎成一圈灰阶波段，每层都是碎边，节点数撑爆
    （实测 12_shadow 高质量档单字 1499 节点，需求要求 50~500）。这里把
    灰调、低对比的层合并为一个阴影区域，再用「边界过渡宽（柔边）」与
    硬边色块（3D 立体面、金属面）区分。贴着阴影的灰阶碎层（碎边）
    一并吸收。需求 5.4 要求阴影保留，不删除。
    """
    raw = prepared.rgb_raw if getattr(prepared, "rgb_raw", None) is not None \
        else prepared.rgba[:, :, :3][:, :, ::-1]
    lum = raw @ np.array([0.299, 0.587, 0.114])
    sat = raw.max(2).astype(np.int32) - raw.min(2).astype(np.int32)
    mask = prepared.mask
    fg = mask > 0
    fg_area = int(fg.sum())
    bg = prepared.bg_color
    bg_lum = float(np.dot(bg, [0.299, 0.587, 0.114])) if bg is not None \
        else float(np.median(lum[~fg])) if (~fg).any() else 255.0
    k3 = np.ones((3, 3), np.uint8)

    # 阴影/光晕的明度朝主体方向过渡；白色挖白细节比底色还亮（或同色），
    # 不是阴影（13_cartoon 的白色高光曾被误判）。主体明度取最大色层
    areas = [(int((labels == i).sum()), i) for i in range(len(palette))]
    areas.sort(reverse=True)
    body_lum = (float(np.median(lum[labels == areas[0][1]]))
                if areas and areas[0][0] >= 300 else None)

    shadow = np.zeros(mask.shape, dtype=bool)
    tiny_gray: list[int] = []
    lumf = lum.astype(np.float32)
    # 过渡宽度比：2px 内的变化 / 6px 内的变化（柔边低、硬边高）
    near = cv2.morphologyEx(lumf, cv2.MORPH_GRADIENT, k3)
    far = cv2.dilate(lumf, np.ones((7, 7), np.uint8)) - cv2.erode(lumf, np.ones((7, 7), np.uint8))
    for i in range(len(palette)):
        lm = labels == i
        area = int(lm.sum())
        if area < 400:
            if 120 <= area and float(sat[lm].mean()) <= 45:
                tiny_gray.append(i)
            continue
        lm_lum = float(np.median(lum[lm]))
        contrast = abs(lm_lum - bg_lum)
        toward_body = True if body_lum is None else (lm_lum - bg_lum) * (body_lum - bg_lum) > 0
        between = contrast >= 8 and (body_lum is None or abs(lm_lum - body_lum) >= 8)
        if not (float(sat[lm].mean()) <= 110 and contrast <= 135 and toward_body and between):
            continue
        # 柔边判定：层外缘的过渡宽度比。阴影/光晕是缓变的模糊边（比值低），
        # 硬边色块哪怕只隔一层色阶，或带抗锯齿的平面色块，过渡都在 2px 内
        m8l = lm.astype(np.uint8)
        ring = cv2.dilate(m8l, k3, iterations=3).astype(bool) \
            & ~cv2.erode(m8l, k3, iterations=2).astype(bool)
        if not ring.any():
            continue
        ratio = near[ring] / np.maximum(far[ring], 25.0)
        if float(np.percentile(ratio, 60)) <= 0.45:
            shadow |= lm

    if not shadow.any():
        return None, None

    # 碎层吸附：贴着阴影的灰阶小碎片（碎边）一并归入阴影
    grown = cv2.dilate(shadow.astype(np.uint8), k3, iterations=10).astype(bool)
    for i in tiny_gray:
        lm = labels == i
        if (lm & grown).any():
            shadow |= lm

    area = int(shadow.sum())
    if area < 0.02 * fg_area or area > 0.85 * fg_area:
        return None, None

    # 平滑：轻度闭运算 + 模糊阈值磨掉碎边，同时保住原有轮廓（重平滑会把
    # 阴影吹成云朵状，轮廓还原度跌到 0.9 以下）
    m8 = shadow.astype(np.uint8) * 255
    k = max(3, int(round(math.sqrt(area) / 40.0)) | 1)
    m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    m8 = cv2.GaussianBlur(m8, (0, 0), 1.4)
    m8 = (m8 >= 128).astype(np.uint8) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
    if n > 1:
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= max(1200, 0.012 * fg_area)
        m8 = np.where(keep[lab], 255, 0).astype(np.uint8)
    if int((m8 > 0).sum()) < 800:
        return None, None

    # 主体保护：大块高对比区域（文字本体）不得被阴影吞掉。
    # 渲染用整块阴影（文字下垫满，不留白缝），分色剔除用 cut 版
    hi = (np.abs(lum - bg_lum) > 135) & fg
    nh, hl, hst, _ = cv2.connectedComponentsWithStats(hi.astype(np.uint8), 8)
    cut = m8
    if nh > 1:
        keep_h = np.zeros(nh, dtype=bool)
        keep_h[1:] = hst[1:, cv2.CC_STAT_AREA] >= 3000
        protect = cv2.dilate(np.where(keep_h[hl], 255, 0).astype(np.uint8), k3, iterations=2)
        cut = np.where(protect > 0, 0, m8).astype(np.uint8)
    return m8, cut


def _build_shadow_layer(prepared: PreparedImage, shadow_mask: np.ndarray,
                        opts: VectorizeOptions) -> tuple[list, int, dict]:
    """阴影层：按软 alpha 分级的嵌套低节点层（matting 化，需求 5.4 勿删阴影）。

    柔边阴影按软 alpha 覆盖率理解：亮度向背景反方向的归一距离就是覆盖率，
    分几级取嵌套圈（外浅内深）近似柔和投影，每级都是低节点平滑轮廓，
    避免碎边路径把节点数撑爆；每圈用该圈实测均色实底，颜色保真。
    shadow_style="soft" 给阴影组套 SVG 模糊滤镜，屏幕渲染得到真正的柔边
    质感（EPS/刻字场景用 shadow_style="steps"，纯路径实色阶梯，无滤镜）。
    注意 raw 取 rgb_raw（RGB 序），填色不反通道。
    返回 (图层列表, 节点数, 明细)。
    """
    raw = prepared.rgb_raw if getattr(prepared, "rgb_raw", None) is not None \
        else prepared.rgba[:, :, :3][:, :, ::-1]
    region = shadow_mask > 0
    area = int(region.sum())
    if area < 800:
        return [], 0, {}
    lum = raw @ np.array([0.299, 0.587, 0.114])
    vals = lum[region]
    bg = prepared.bg_color
    bg_lum = float(np.dot(bg, [0.299, 0.587, 0.114])) if bg is not None \
        else (float(np.median(lum[~region])) if (~region).any() else 255.0)
    shadow_opts = replace(opts, snap_deg=0.0, smooth=True, contour_grow=0.4)
    k3 = np.ones((3, 3), np.uint8)
    layers: list[SvgLayer] = []
    total_nodes = 0

    # 软覆盖率：亮度向背景反方向的归一距离（判定阶梯/柔和两种风格与明暗方向）
    direction = -1.0 if float(np.median(vals)) <= bg_lum else 1.0
    extreme = float(np.percentile(vals, 3)) if direction < 0 else float(np.percentile(vals, 97))
    span = abs(extreme - bg_lum)

    steps_style = getattr(opts, "shadow_style", "soft") == "steps" or span < 10
    # 内圈三级（浅→深），最外层用整块阴影区域垫底，保住外缘淡影；
    # 内圈经闭运算+模糊会略胀出区域，恰好补住外圈大容差简化的收缩
    lo, hi = np.percentile(vals, [4, 96])
    inner_thresholds = [float(lo + (hi - lo) * f) for f in (0.22, 0.45, 0.78)] if hi - lo >= 10 else []
    built_masks: list[np.ndarray] = []
    for t in inner_thresholds:
        m = region & (lum <= t) if direction < 0 else region & (lum >= t)
        if int(m.sum()) < 600:
            continue
        mm = cv2.morphologyEx((m.astype(np.uint8) * 255), cv2.MORPH_CLOSE, k3, iterations=2)
        mm = cv2.GaussianBlur(mm, (0, 0), 1.2)
        mm = (mm >= 128).astype(np.uint8) * 255
        built_masks.append(mm > 0)
    draw_order = [region] + list(reversed(built_masks))

    soft_style = not steps_style

    info_steps = []
    for i, m in enumerate(draw_order):          # 外→内依次绘制
        budget = max(10, min(40, int(1.3 * math.sqrt(max(int(m.sum()), 1)))))
        paths, fitted = _fit_layer_paths((m.astype(np.uint8) * 255), shadow_opts, budget,
                                         speckle_min_area=max(300.0, area * 0.012),
                                         tolerance=max(2.2, opts.tolerance * 5.0),
                                         max_hole_nodes=10, deburr_dev=0.0)
        if not paths:
            continue
        nxt = draw_order[i + 1] if i + 1 < len(draw_order) else None
        ring = (m & ~nxt) if nxt is not None else m      # 实际可见的那圈
        px = raw[ring] if int(ring.sum()) >= 30 else raw[m]
        # 每圈用该圈实测均色实底：颜色保真零风险，质感由柔和滤镜完成
        color_hex = rgb_to_hex(px.mean(axis=0)) if len(px) else "#c8c8c8"
        opacity = 1.0
        nodes = int(sum(f.node_count for f in fitted))
        total_nodes += nodes
        layers.append(SvgLayer(color=color_hex, paths=paths, label="shadow",
                               opacity=opacity,
                               filter_id="shadow-soft" if soft_style else None))
        info_steps.append({"color": color_hex, "nodes": nodes, "paths": len(paths),
                           "area": int(m.sum()), "opacity": round(opacity, 3)})
    info = {"gradient": False, "steps": len(layers), "nodes": total_nodes,
            "area": area, "style": "steps" if steps_style else "soft",
            "detail": info_steps}
    return layers, total_nodes, info


def _cluster_palette(colors: np.ndarray, counts: np.ndarray, max_colors: int,
                     merge_de: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """把相近的主色合并成一族，避免同一颜色被量化桶拆成多层。"""
    if len(colors) == 0:
        return colors, counts
    lab = cv2.cvtColor(colors.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)

    order = np.argsort(-counts)
    clusters: list[dict] = []
    for i in order:
        placed = False
        for c in clusters:
            if float(np.linalg.norm(lab[i] - c["lab"])) < merge_de:
                total = c["count"] + counts[i]
                c["color"] = (c["color"] * c["count"] + colors[i] * counts[i]) / max(1, total)
                c["lab"] = (c["lab"] * c["count"] + lab[i] * counts[i]) / max(1, total)
                c["count"] = total
                placed = True
                break
        if not placed:
            clusters.append({"color": colors[i].astype(np.float64), "lab": lab[i],
                             "count": int(counts[i])})

    while len(clusters) > max_colors:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = float(np.linalg.norm(clusters[i]["lab"] - clusters[j]["lab"]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        a, b = clusters[i], clusters[j]
        total = a["count"] + b["count"]
        a["color"] = (a["color"] * a["count"] + b["color"] * b["count"]) / max(1, total)
        a["lab"] = (a["lab"] * a["count"] + b["lab"] * b["count"]) / max(1, total)
        a["count"] = total
        clusters.pop(j)

    clusters.sort(key=lambda c: -c["count"])
    out_colors = np.asarray([c["color"] for c in clusters], dtype=np.float32)
    out_counts = np.asarray([c["count"] for c in clusters], dtype=np.int64)
    return out_colors, out_counts


def _palette_from_foreground(rgb: np.ndarray, core_mask: np.ndarray,
                             max_colors: int, min_ratio: float = 0.002,
                             min_px: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """提取主色。

    先按 4bit 量化统计主色桶，桶内像素占比达标的算作真实颜色层；
    主色桶数量超标（渐变/彩虹）时改用 k-means 聚类压缩到 max_colors。
    """
    pixels = rgb[core_mask > 0]
    if len(pixels) == 0:
        return np.zeros((0, 3), np.float32), np.zeros(0, np.int64)

    q = (pixels // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(-counts)
    threshold = max(min_px, len(pixels) * min_ratio)
    big = [int(i) for i in order if counts[i] >= threshold]

    if big and len(big) <= max(6, max_colors * 2):
        colors = np.zeros((len(big), 3), np.float32)
        for k, i in enumerate(big):
            colors[k] = pixels[inverse == i].mean(axis=0)
        return _cluster_palette(colors, counts[big].astype(np.int64), max_colors)

    if big:
        keep = np.isin(inverse, np.asarray(big, dtype=inverse.dtype))
        pool = pixels[keep]
    else:
        pool = pixels

    sample = pool if len(pool) <= 60000 else pool[:: max(1, len(pool) // 60000)]
    k = int(min(max_colors, max(2, len(np.unique(sample, axis=0)))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.6)
    # 固定随机种子：k-means 初值会改变分层数量与耗时，同一张图必须每次一致
    cv2.setRNGSeed(20240614)
    _, labels, centers = cv2.kmeans(sample.astype(np.float32), k, None,
                                    criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts_k = np.bincount(labels.flatten(), minlength=k).astype(np.int64)
    order2 = np.argsort(-counts_k)
    return _cluster_palette(centers.astype(np.float32)[order2], counts_k[order2], max_colors)


def _label_pixels(rgb: np.ndarray, palette_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把前景像素指派到最近的主色（Lab 空间）。

    先把颜色量化到 4bit/通道（4096 格）建一张查找表，再逐像素查表，
    避免百万像素 × 多主色的逐点距离计算。
    """
    pal_lab = cv2.cvtColor(palette_bgr.reshape(-1, 1, 3).astype(np.uint8),
                           cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    grid = np.arange(0, 16, dtype=np.int32)
    bb, gg, rr = np.meshgrid(grid, grid, grid, indexing="ij")
    centers = np.stack([bb.ravel() * 16 + 8, gg.ravel() * 16 + 8, rr.ravel() * 16 + 8],
                       axis=1).astype(np.uint8)          # BGR 顺序的格子中心
    centers_lab = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB)
    centers_lab = centers_lab.reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(centers_lab[:, None, :] - pal_lab[None, :, :], axis=2)
    lut = np.argmin(dist, axis=1).astype(np.int32).reshape(16, 16, 16)

    q = (rgb // 16).astype(np.int32)
    labels = lut[q[:, :, 0], q[:, :, 1], q[:, :, 2]].astype(np.int32)
    labels[mask == 0] = -1
    return labels


def _propagate_labels(labels: np.ndarray, fg: np.ndarray, iterations: int = 0) -> np.ndarray:
    """把实心区的颜色标签补满整个前景。

    用距离变换取「最近的已标注像素」标签（精确、一趟完成），避免逐步膨胀
    迭代不够时留下未标注缝隙——那会把笔画切碎、内孔也一起消失。
    """
    out = labels.copy()
    fg_bool = fg > 0 if fg.dtype != bool else fg
    known = out >= 0
    pending = (~known) & fg_bool
    if not pending.any():
        return out
    if not known.any():
        return out

    from scipy import ndimage
    _, indices = ndimage.distance_transform_edt(~known, return_indices=True)
    nearest = out[indices[0], indices[1]]
    out[pending] = nearest[pending]
    return out


def _merge_small_layers(labels: np.ndarray, palette_bgr: np.ndarray,
                        fg_area: float, min_ratio: float = 0.008,
                        max_merge: int = 1) -> np.ndarray:
    """把占比过小的颜色层归并到色彩最接近的大层，减少碎色与毛刺。"""
    if len(palette_bgr) <= max_merge:
        return labels
    areas = np.array([(labels == i).sum() for i in range(len(palette_bgr))], dtype=np.float64)
    order = np.argsort(-areas)
    lab = cv2.cvtColor(palette_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)
    lab = lab.reshape(-1, 3).astype(np.float32)

    for idx in order:
        if areas[idx] >= fg_area * min_ratio or areas[idx] <= 0:
            continue
        others = [j for j in order if j != idx and areas[j] >= areas[idx]]
        if not others:
            continue
        dists = [np.linalg.norm(lab[idx] - lab[j]) for j in others]
        target = others[int(np.argmin(dists))]
        labels = np.where(labels == idx, target, labels)
        areas[target] += areas[idx]
        areas[idx] = 0
    return labels


def _remap_palette(labels: np.ndarray, palette_bgr: np.ndarray):
    used = sorted(int(v) for v in np.unique(labels) if v >= 0)
    if not used:
        return labels, palette_bgr
    mapping = {old: new for new, old in enumerate(used)}
    new_labels = np.full_like(labels, -1)
    for old, new in mapping.items():
        new_labels[labels == old] = new
    return new_labels, palette_bgr[used]


def _clean_layer_mask(layer: np.ndarray, min_area: float, grow: int = 0,
                      clip: np.ndarray | None = None) -> np.ndarray:
    m = (layer > 0).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    if grow > 0:
        # 上层图层向外扩 1px 盖住层间缝隙，但不得越过前景轮廓向外扩张
        m = cv2.dilate(m, kernel, iterations=grow)
        if clip is not None:
            m = np.where(clip > 0, m, 0).astype(np.uint8)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    # 闭运算只用来封小于 9px 的层间缝隙；它会把楔形内孔挤窄甚至填实
    # （实测把 44px² 的字形内孔收到 18px²），被填的 ≥9px 区域要还回去
    added = (closed > 0) & (m == 0)
    if added.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(added.astype(np.uint8), 8)
        keep_fill = np.zeros(n, dtype=bool)
        keep_fill[1:] = stats[1:, cv2.CC_STAT_AREA] < 9
        m = np.where(keep_fill[lab], 255, m).astype(np.uint8)
    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.zeros(n, dtype=bool)
            keep[1:] = areas >= min_area
            m = np.where(keep[labels], 255, 0).astype(np.uint8)
    return m


def _find_contours_subpixel(mask: np.ndarray, upsample: int = 2):
    """在 2× 分辨率下取轮廓，得到亚像素精度的字形边界。

    直接在 1× 掩膜上 findContours 得到的是「像素中心点集」，填充后比真实边界
    系统性偏瘦约半像素（细笔画素材尤其明显）。先把掩膜升采样再取轮廓，
    坐标除以倍率，边界误差从 ~0.5px 降到 ~0.25px。
    """
    if upsample <= 1:
        c, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return c, h
    h0, w0 = mask.shape[:2]
    up = cv2.resize(mask, (w0 * upsample, h0 * upsample), interpolation=cv2.INTER_LINEAR)
    binary = (up > 127).astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    scaled = [c.reshape(-1, 2).astype(np.float64) / float(upsample) for c in contours]
    return scaled, hierarchy


def _bg_hole_mask(mask: np.ndarray, min_area: float = 20.0) -> np.ndarray:
    """前景里的背景孔（字形镂空）掩膜。贴着它们的轮廓要保护性拟合。"""
    m = (mask > 0).astype(np.uint8)
    out = np.zeros(mask.shape, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return out
    for i, h in enumerate(hierarchy[0]):
        if h[3] != -1 and abs(cv2.contourArea(contours[i])) >= min_area:
            cv2.drawContours(out, contours, i, 255, -1)
    return out


def _fit_layer_paths(mask: np.ndarray, opts: VectorizeOptions,
                     per_glyph_budget: int, speckle_min_area: float = 0.0,
                     tolerance: float | None = None,
                     max_hole_nodes: int | None = None,
                     protect_mask: np.ndarray | None = None,
                     deburr_dev: float = 1.6) -> tuple[list[str], list[FittedContour]]:
    """把一层蒙版拟合成 SVG 路径。

    每个 path 元素 = 一条外轮廓 + 它的直接内孔（多条子路径），按 per-path even-odd
    渲染时孔才会被挖空；浏览器与 Illustrator 都按这个语义解释 fill-rule。
    嵌套的岛（孔中实体）会作为独立 path 元素、按嵌套深度从小到大排列。
    """
    contours, hierarchy = _find_contours_subpixel(mask)
    if opts.contour_coordinate_offset:
        contours = [np.asarray(c, dtype=float) + opts.contour_coordinate_offset for c in contours]
    if hierarchy is None or len(contours) == 0:
        return [], []
    hier = hierarchy[0]
    min_area = max(2.0, speckle_min_area)
    tol = opts.tolerance if tolerance is None else tolerance

    depths = np.zeros(len(contours), dtype=np.int32)
    for i in range(len(contours)):
        d, parent = 0, int(hier[i][3])
        while parent != -1 and d < 64:
            d += 1
            parent = int(hier[parent][3])
        depths[i] = d

    fitted: list[FittedContour] = []

    def fit_one(idx: int, is_hole: bool = False) -> str | None:
        c = contours[idx]
        if len(c) < 4:
            return None
        pts = c.reshape(-1, 2).astype(np.float64)
        area = abs(polygon_area(pts))
        # 内孔是字形的一部分，不能被前景噪点门槛误杀：用独立的低门槛
        min_a = 4.0 if is_hole else min_area
        if area < min_a:
            return None
        budget = max(opts.min_nodes, min(per_glyph_budget,
                                        int(2.6 * np.sqrt(area)) or opts.min_nodes))
        if is_hole and max_hole_nodes:
            # 叠加层（如描边）的内孔只需与本体层的孔对齐，用少量节点即可
            budget = max(opts.min_nodes, min(budget, int(max_hole_nodes)))
        fit_tol = tol
        if protect_mask is not None and not is_hole:
            # 贴着字形镂空的轮廓用紧容差 + 放宽预算：多层各自拟合的凹口被
            # 磨圆后拼起来会把孔吃小（实测 11_3d 的 50px² 孔被吃到 18px²）
            xs = np.clip(pts[:, 0].astype(int), 0, protect_mask.shape[1] - 1)
            ys = np.clip(pts[:, 1].astype(int), 0, protect_mask.shape[0] - 1)
            if bool(protect_mask[ys, xs].any()):
                fit_tol = max(0.22, tol * 0.5)
                budget = max(budget, 22)
        # 像素中心点集与真实边界的系统性偏差补偿（contour_grow 实测标定）
        fc = fit_contour(offset_polygon(pts, opts.contour_grow), tolerance=fit_tol,
                         corner_angle=opts.corner_angle,
                         max_nodes=budget, min_nodes=opts.min_nodes,
                         closed=True, smooth=opts.smooth, snap_deg=opts.snap_deg,
                         deburr_dev=deburr_dev)
        # 小孔拟合后面积缩水会被填实：收紧容差补回（保留两者中面积更大者）。
        # 触发线设 0.95：小孔栅格化时边界还会再缩一圈，缩水 5% 以上就可能
        # 跌破 24px² 的内孔计数线
        if is_hole and area >= 8.0:
            fitted_area = abs(polygon_area(fc.points)) if len(fc.points) >= 3 else 0.0
            if fitted_area < 0.95 * area:
                refit_max = max(opts.min_nodes, budget)
                if is_hole and max_hole_nodes:
                    refit_max = min(refit_max, int(max_hole_nodes))
                fc2 = fit_contour(offset_polygon(pts, opts.contour_grow),
                                  tolerance=max(0.12, tol * 0.3),
                                  corner_angle=opts.corner_angle,
                                  max_nodes=refit_max,
                                  min_nodes=max(opts.min_nodes, 8),
                                  closed=True, smooth=opts.smooth, snap_deg=opts.snap_deg,
                                  deburr_dev=deburr_dev)
                fitted2 = abs(polygon_area(fc2.points)) if len(fc2.points) >= 3 else 0.0
                if fitted2 > fitted_area:
                    fc = fc2
                    fitted_area = fitted2
                # 吸附把小孔挤小时（实测 11_3d 的 50px² 小孔被挤没），
                # 用不吸附的紧容差重拟合补回面积，保住内孔计数
                if opts.snap_deg > 0 and fitted_area < 0.9 * area:
                    fc3 = fit_contour(offset_polygon(pts, opts.contour_grow),
                                      tolerance=max(0.12, tol * 0.3),
                                      corner_angle=opts.corner_angle,
                                      max_nodes=refit_max,
                                      min_nodes=max(opts.min_nodes, 8),
                                      closed=True, smooth=opts.smooth, snap_deg=0.0,
                                      deburr_dev=deburr_dev)
                    fitted3 = abs(polygon_area(fc3.points)) if len(fc3.points) >= 3 else 0.0
                    if fitted3 > fitted_area:
                        fc = fc3
        d = contour_to_path_d(fc)
        if not d:
            return None
        fitted.append(fc)
        return d

    groups: list[tuple[int, str]] = []
    for i in range(len(contours)):
        if depths[i] % 2 != 0:      # 孔洞由它的父轮廓吸收
            continue
        outer = fit_one(i)
        if outer is None:
            continue
        parts = [outer]
        child = int(hier[i][2])
        while child != -1:
            hole = fit_one(child, is_hole=True)
            if hole:
                parts.append(hole)
            child = int(hier[child][0])
        groups.append((int(depths[i]), " ".join(parts)))

    groups.sort(key=lambda t: t[0])
    return [d for _, d in groups], fitted


def _try_linear_gradient(rgb: np.ndarray, layer_mask: np.ndarray,
                         levels: int,
                         rgb_color: np.ndarray | None = None) -> dict | None:
    """对大面积渐变色层做线性渐变拟合，成功则返回渐变定义。

    拟合与判定用 rgb（预处理后、抗噪），色阶颜色用 rgb_color（原始像素），
    保证输出色值不带 CLAHE 偏移。
    """
    ys, xs = np.nonzero(layer_mask > 0)
    if len(xs) < 2000:
        return None
    step = max(1, len(xs) // 20000)
    xs_s, ys_s = xs[::step].astype(np.float64), ys[::step].astype(np.float64)
    cols = rgb[ys_s.astype(int), xs_s.astype(int)].astype(np.float64)  # BGR
    cols_out = (rgb_color if rgb_color is not None else rgb)[
        ys_s.astype(int), xs_s.astype(int)].astype(np.float64)
    lum = cols @ np.array([0.114, 0.587, 0.299])

    A = np.stack([xs_s, ys_s, np.ones_like(xs_s)], axis=1)
    coef, *_ = np.linalg.lstsq(A, lum, rcond=None)
    pred = A @ coef
    ss_res = float(((lum - pred) ** 2).sum())
    ss_tot = float(((lum - lum.mean()) ** 2).sum()) + 1e-9
    r2 = 1 - ss_res / ss_tot
    if r2 < 0.55:
        return None

    direction = np.array([coef[0], coef[1]])
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    direction /= norm

    proj = xs_s * direction[0] + ys_s * direction[1]
    p_min, p_max = float(proj.min()), float(proj.max())
    if p_max - p_min < 12:
        return None

    levels = max(8, int(levels))
    bins = np.clip(((proj - p_min) / (p_max - p_min) * levels).astype(int), 0, levels - 1)
    stops: list[tuple[float, str]] = []
    for b in range(levels):
        sel = bins == b
        if not sel.any():
            continue
        color = cols_out[sel].mean(axis=0)
        stops.append((float(b + 0.5) / levels, rgb_to_hex(color[::-1])))
    if len(stops) < 6:
        return None

    x1 = direction[0] * p_min if abs(direction[0]) > 1e-6 else 0.0
    y1 = direction[1] * p_min if abs(direction[1]) > 1e-6 else 0.0
    x2 = direction[0] * p_max if abs(direction[0]) > 1e-6 else 0.0
    y2 = direction[1] * p_max if abs(direction[1]) > 1e-6 else 0.0
    return {"stops": stops, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "r2": round(r2, 3)}


# ---------------------------------------------------------------------------
# 引擎一：自研轮廓引擎（平面色 / 描边 / 多色卡通字）
# ---------------------------------------------------------------------------


def _gradient_explained(rgb: np.ndarray, masked_pixels: tuple[np.ndarray, np.ndarray],
                        direction: np.ndarray, bins: int,
                        rgb_color: np.ndarray | None = None) -> tuple[float, list[tuple[float, str]], float, float]:
    """沿方向分箱统计颜色，返回（解释度, 色阶, 投影最小/最大值）。

    除拟合优度外还要求：色阶沿方向单调平滑（无跳变）、像素普适贴合拟合色阶
    （离梯度比例低），避免多色块图像被误判成渐变。判定用 rgb，色阶颜色取
    rgb_color（原始像素）。
    """
    xs, ys = masked_pixels
    proj = xs * direction[0] + ys * direction[1]
    p_min, p_max = float(proj.min()), float(proj.max())
    if p_max - p_min < 12:
        return 0.0, [], p_min, p_max
    idx = np.clip(((proj - p_min) / (p_max - p_min) * bins).astype(int), 0, bins - 1)

    colors = rgb[ys.astype(int), xs.astype(int)].astype(np.float64)
    colors_out = (rgb_color if rgb_color is not None else rgb)[
        ys.astype(int), xs.astype(int)].astype(np.float64)
    total_var = float(colors.var(axis=0).mean())
    if total_var < 20.0:
        return 0.0, [], p_min, p_max

    means = np.zeros((bins, 3), dtype=np.float64)
    valid = np.zeros(bins, dtype=bool)
    stops: list[tuple[float, str]] = []
    within_sum, within_n = 0.0, 0
    for b in range(bins):
        sel = idx == b
        n_sel = int(sel.sum())
        if n_sel < 8:
            continue
        c = colors[sel]
        means[b] = c.mean(axis=0)
        valid[b] = True
        within_sum += float(c.var(axis=0).mean()) * n_sel
        within_n += n_sel
        stops.append((float(b + 0.5) / bins,
                      rgb_to_hex(colors_out[sel].mean(axis=0)[::-1])))
    if len(stops) < 6:
        return 0.0, [], p_min, p_max

    # 真渐变要求色阶沿方向平滑变化：逐通道线性拟合，合并 R²（按通道幅度加权，
    # 避免低幅度通道的噪声拖垮均值）；相邻色阶不跳变
    seq = means[valid]
    x = np.arange(len(seq), dtype=np.float64)
    ss_res_sum, ss_tot_sum = 0.0, 0.0
    for ch in range(3):
        y = seq[:, ch]
        if y.std() < 1e-6:
            continue
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        ss_res_sum += float(((y - pred) ** 2).sum())
        ss_tot_sum += float(((y - y.mean()) ** 2).sum())
    pooled_r2 = 1.0 - ss_res_sum / (ss_tot_sum + 1e-9)
    jumps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    color_range = float(np.linalg.norm(seq.max(axis=0) - seq.min(axis=0)))
    max_jump = float(jumps.max()) if len(jumps) else 0.0
    if pooled_r2 < 0.78 or max_jump > 0.28 * color_range + 12:
        return 0.0, [], p_min, p_max

    within = within_sum / max(1, within_n - len(stops))
    explained = 1.0 - within / max(1e-6, total_var)
    if explained < 0.6:
        return 0.0, [], p_min, p_max

    # 离梯度比例：像素颜色偏离所属色阶的幅度过大则不算渐变（如刮层叠色）
    sel_px = valid[idx]
    if sel_px.sum() < 100:
        return 0.0, [], p_min, p_max
    residual = np.linalg.norm(colors[sel_px] - means[idx[sel_px]], axis=1)
    off_ratio = float((residual > 45.0).mean())
    if off_ratio > 0.15:
        return 0.0, [], p_min, p_max
    return float(explained), stops, p_min, p_max


def _fit_global_gradient(rgb: np.ndarray, mask: np.ndarray, levels: int,
                         rgb_color: np.ndarray | None = None) -> dict | None:
    """整图渐变判定：颜色主要由某个线性方向决定时，输出单层渐变填充。"""
    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    if (core > 0).sum() < 3000:
        core = mask
    ys, xs = np.nonzero(core > 0)
    if len(xs) < 3000:
        return None
    step = max(1, len(xs) // 20000)
    xs_s, ys_s = xs[::step].astype(np.float64), ys[::step].astype(np.float64)
    cols_all = rgb[ys_s.astype(int), xs_s.astype(int)].astype(np.float64)

    # 先剔除占比 ≥18% 的单一纯色层（描边/底色）：它会主导亮度方向拟合，
    # 把渐变本体挤成少数几个分箱；纯单色图会被整体剔除，自然判为非渐变
    q = (cols_all // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    keep = np.ones(len(cols_all), dtype=bool)
    if len(uniq):
        top_share = float(counts.max()) / float(len(cols_all))
        if top_share >= 0.18:
            top_key = uniq[int(np.argmax(counts))]
            keep = keys != top_key
    if int(keep.sum()) < 500:
        return None
    xs_k, ys_k = xs_s[keep], ys_s[keep]
    cols = cols_all[keep]
    lum = cols @ np.array([0.114, 0.587, 0.299])

    A = np.stack([xs_k, ys_k, np.ones(len(xs_k))], axis=1)
    coef, *_ = np.linalg.lstsq(A, lum, rcond=None)
    direction = np.array([coef[0], coef[1]])
    if np.linalg.norm(direction) < 1e-6:
        return None
    direction = direction / np.linalg.norm(direction)

    # 判定用固定分箱数，避免结论随渐变档位抖动
    bins = int(min(max(levels, 8), 12))

    # 两遍拟合：第一遍方向可能被小占比杂色拉偏；用分箱中位数剔除离群
    # 像素（残差 >45）后重拟合方向，只对渐变本体做判定
    proj0 = xs_k * direction[0] + ys_k * direction[1]
    p_min0, p_max0 = float(proj0.min()), float(proj0.max())
    xs_f, ys_f = xs_k, ys_k
    if p_max0 - p_min0 >= 12:
        idx0 = np.clip(((proj0 - p_min0) / (p_max0 - p_min0) * bins).astype(int), 0, bins - 1)
        means0 = np.full((bins, 3), np.nan)
        for b in range(bins):
            sel = idx0 == b
            if int(sel.sum()) >= 8:
                # 分箱中心取中位数：杂色不会把中心拉偏
                means0[b] = np.median(cols[sel], axis=0)
        inlier = np.zeros(len(cols), dtype=bool)
        for k in range(len(cols)):
            m = means0[idx0[k]]
            if not np.isnan(m[0]):
                inlier[k] = float(np.linalg.norm(cols[k] - m)) <= 45.0
        if inlier.mean() >= 0.5 and inlier.any():
            A2 = np.stack([xs_k[inlier], ys_k[inlier], np.ones(int(inlier.sum()))], axis=1)
            coef2, *_ = np.linalg.lstsq(A2, lum[inlier], rcond=None)
            d2 = np.array([coef2[0], coef2[1]])
            if np.linalg.norm(d2) >= 1e-6:
                xs_f, ys_f = xs_k[inlier], ys_k[inlier]
                direction = d2 / np.linalg.norm(d2)

    explained, stops, p_min, p_max = _gradient_explained(rgb, (xs_f, ys_f), direction, bins,
                                                         rgb_color=rgb_color)
    if explained <= 0:
        return None
    x1 = direction[0] * p_min if abs(direction[0]) > 1e-6 else 0.0
    y1 = direction[1] * p_min if abs(direction[1]) > 1e-6 else 0.0
    x2 = direction[0] * p_max if abs(direction[0]) > 1e-6 else 0.0
    y2 = direction[1] * p_max if abs(direction[1]) > 1e-6 else 0.0
    return {"stops": stops, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "explained": round(explained, 3)}


def _match_spot(color_hex: str, spot_colors: list[str],
                tolerance: float = 0.03, force: bool = False) -> tuple[str, str | None]:
    """专色匹配：默认仅替换通道偏差 ≤3% 的图层；显式开启时取最近色板色。

    返回 (最终色, 命中的专色或 None)。只作用于平面色层；渐变层保持连续色。
    """
    from .svgutil import hex_to_rgb
    if force:
        from .spot_colors import nearest_color, normalize_palette
        target = nearest_color(color_hex, normalize_palette(spot_colors))
        return target, target
    c = np.asarray(hex_to_rgb(color_hex), dtype=np.float64)
    best, best_d = None, None
    for spot in spot_colors:
        try:
            s = np.asarray(hex_to_rgb(spot), dtype=np.float64)
        except Exception:
            continue
        d = float(np.abs(c - s).max()) / 255.0
        if best_d is None or d < best_d:
            best, best_d = spot.lower(), d
    if best is not None and best_d is not None and (force or best_d <= tolerance):
        return best, best
    return color_hex, None


def _fit_radial_gradient(rgb: np.ndarray, mask: np.ndarray, levels: int,
                         rgb_color: np.ndarray | None = None) -> dict | None:
    """径向渐变判定：颜色主要由到中心的距离决定时，输出 radialGradient。

    判据与线性渐变相同（逐通道线性拟合 R²、相邻色阶无跳变、方差解释度、
    离梯度比例），只是投影量从线性方向换成半径。
    """
    core = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    if (core > 0).sum() < 3000:
        core = mask
    ys, xs = np.nonzero(core > 0)
    if len(xs) < 3000:
        return None
    step = max(1, len(xs) // 20000)
    xs_s, ys_s = xs[::step].astype(np.float64), ys[::step].astype(np.float64)
    cx, cy = float(xs_s.mean()), float(ys_s.mean())
    radii = np.hypot(xs_s - cx, ys_s - cy)
    r_max = float(radii.max())
    if r_max < 24:
        return None
    colors = rgb[ys_s.astype(int), xs_s.astype(int)].astype(np.float64)
    colors_out = (rgb_color if rgb_color is not None else rgb)[
        ys_s.astype(int), xs_s.astype(int)].astype(np.float64)
    total_var = float(colors.var(axis=0).mean())
    if total_var < 40.0:
        return None
    # 按实际半径跨度归一化：文字往往不覆盖中心区域，0 起点会浪费分箱
    r_min = float(radii.min())
    span = float(r_max - r_min)
    if span < 24:
        return None
    bins = int(min(max(levels, 8), 12))
    idx = np.clip(((radii - r_min) / span * bins).astype(int), 0, bins - 1)
    means = np.zeros((bins, 3))
    valid = np.zeros(bins, dtype=bool)
    stops: list[tuple[float, str]] = []
    within_sum, within_n = 0.0, 0
    for b in range(bins):
        sel = idx == b
        n_sel = int(sel.sum())
        if n_sel < 8:
            continue
        c = colors[sel]
        means[b] = c.mean(axis=0)
        valid[b] = True
        within_sum += float(c.var(axis=0).mean()) * n_sel
        within_n += n_sel
        # SVG 径向渐变的 offset 映射半径 o*r，因此按实际半径换算 offset
        radius = r_min + (b + 0.5) / bins * span
        stops.append((min(1.0, radius / r_max), rgb_to_hex(colors_out[sel].mean(axis=0)[::-1])))
    if len(stops) < 6:
        return None
    seq = means[valid]
    x = np.arange(len(seq), dtype=np.float64)
    ss_res_sum, ss_tot_sum = 0.0, 0.0
    for ch in range(3):
        y = seq[:, ch]
        if y.std() < 1e-6:
            continue
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        ss_res_sum += float(((y - pred) ** 2).sum())
        ss_tot_sum += float(((y - y.mean()) ** 2).sum())
    pooled_r2 = 1.0 - ss_res_sum / (ss_tot_sum + 1e-9)
    jumps = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    color_range = float(np.linalg.norm(seq.max(axis=0) - seq.min(axis=0)))
    max_jump = float(jumps.max()) if len(jumps) else 0.0
    if pooled_r2 < 0.78 or max_jump > 0.28 * color_range + 12:
        return None
    within = within_sum / max(1, within_n - len(stops))
    explained = 1.0 - within / max(1e-6, total_var)
    if explained < 0.6:
        return None
    sel_px = valid[idx]
    if sel_px.sum() < 100:
        return None
    residual = np.linalg.norm(colors[sel_px] - means[idx[sel_px]], axis=1)
    if float((residual > 45.0).mean()) > 0.15:
        return None
    return {"type": "radial", "cx": cx, "cy": cy, "r": r_max,
            "stops": stops, "explained": round(explained, 3)}


def _solid_overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                        min_share: float = 0.18) -> np.ndarray | None:
    """找出渐变字上的大占比纯色层（描边/底色），返回其布尔掩膜。

    占比不足或纯色占比过高（整图单色）时返回 None。
    """
    sel = mask > 0
    px = rgb[sel]
    if len(px) == 0:
        return None
    q = (px // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    if len(uniq) == 0:
        return None
    top = int(np.argmax(counts))
    share = float(counts[top]) / float(len(keys))
    if share < min_share or share > 0.92:
        return None
    out = np.zeros(mask.shape, dtype=bool)
    hit = keys == uniq[top]
    # 同族色（抗锯齿混色与轻微色差）一并归入实色层，避免色带里出楔形缺口
    mean_c = px[hit].mean(axis=0)
    near = np.linalg.norm(px - mean_c, axis=1) <= 45.0
    out[sel] = hit | near
    return out


def _build_stroked_gradient_document(prepared: PreparedImage, opts: VectorizeOptions,
                                     grad: dict, solid_mask: np.ndarray) -> tuple[SvgDocument, dict]:
    """渐变字带实色层（描边/底色）：实色层垫底，渐变字身画在上面（需求 5.3）。

    实色层往往是厚描边/底色（实测占前景可过半），盖在字身上会吃掉字身；
    反过来实色层用整块剪影垫底、字身（前景中的非实色区）盖在上面，两层
    边界永不露白：字身拟合收缩只会多露一点描边，不会露背景。
    """
    mask = prepared.mask
    h, w = mask.shape
    fg_area = float((mask > 0).sum())
    speckle = max(float(opts.filter_speckle), fg_area * 0.00035)
    min_area = float(max(6.0, fg_area * 0.00012))
    raw = prepared.rgb_raw[:, :, ::-1] if getattr(prepared, "rgb_raw", None) is not None else prepared.rgba[:, :, :3]
    is_radial = "radial" in str(grad.get("type", ""))

    doc = SvgDocument(width=w, height=h)
    gid = "grad-1"
    grad_ref = {"ref": f"url(#{gid})", "stops": grad["stops"],
                "type": "radial" if is_radial else "linear"}
    if is_radial:
        doc.defs.append(build_radial_gradient(gid, grad["stops"],
                                              grad["cx"], grad["cy"], grad["r"]))
        grad_ref.update({"cx": grad["cx"], "cy": grad["cy"], "r": grad["r"]})
    else:
        doc.defs.append(build_linear_gradient(gid, grad["stops"], grad["x1"], grad["y1"],
                                              grad["x2"], grad["y2"]))

    # 预算按整字 500 分摊：字身是可见字形，多分一点
    budget_body = max(opts.min_nodes, int(opts.max_nodes * 0.55))
    budget_solid = max(opts.min_nodes, int(opts.max_nodes * 0.3))
    protect = _bg_hole_mask(mask)

    solid_px = (solid_mask > 0) & (mask > 0)
    solid_bin = solid_px.astype(np.uint8) * 255
    # 字身 = 前景中的非实色区；实色边缘的抗锯齿混色归描边，字身略瘦一圈无妨
    interior = (mask > 0) & ~cv2.dilate(solid_bin, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if int(interior.sum()) < 100:
        interior = mask > 0

    # 实色层：整块剪影垫底（内孔与背景孔对齐），填色取实色实测均色
    solid_paths, solid_fitted = _fit_layer_paths(mask, opts, budget_solid,
                                                 speckle_min_area=speckle,
                                                 max_hole_nodes=14, protect_mask=protect)
    px_solid = raw[solid_px] if int(solid_px.sum()) >= 30 else raw[mask > 0]
    solid_color = rgb_to_hex(px_solid.mean(axis=0)[::-1]) if len(px_solid) else "#000000"
    solid_layer = SvgLayer(color=solid_color, paths=solid_paths, label=solid_color)

    # 渐变字身：画在实色层上面，可见字形
    body_paths, body_fitted = _fit_layer_paths((interior.astype(np.uint8) * 255), opts,
                                               budget_body, speckle_min_area=speckle,
                                               protect_mask=protect)
    body_layer = SvgLayer(color="#000000", paths=body_paths, label="gradient", gradient=grad_ref)

    doc.layers.append(solid_layer)
    if body_paths:
        doc.layers.append(body_layer)
    glyphs = opts.glyph_count or estimate_glyph_count(mask)
    doc.metadata = {"glyphs": glyphs, "mode": opts.mode}
    node_body = int(sum(f.node_count for f in body_fitted))
    node_solid = int(sum(f.node_count for f in solid_fitted))
    stats = {
        "engine": "contour",
        "layers": len(doc.layers),
        "glyphs": glyphs,
        "gradient_layers": 1,
        "gradient_stops": len(grad["stops"]),
        "gradient_explained": grad.get("explained"),
        "node_total": node_body + node_solid,
        "mode_detail": "stroked-gradient",
    }
    return doc, stats


def _build_gradient_document(prepared: PreparedImage, opts: VectorizeOptions,
                             grad: dict) -> tuple[SvgDocument, dict]:
    """单层渐变字：整体轮廓 + linearGradient/radialGradient 填充。"""
    mask = prepared.mask
    h, w = mask.shape
    fg_area = float((mask > 0).sum())
    speckle = max(float(opts.filter_speckle), fg_area * 0.00035)
    paths, fitted = _fit_layer_paths(mask, opts, max(opts.min_nodes, opts.max_nodes),
                                     speckle_min_area=speckle,
                                     protect_mask=_bg_hole_mask(mask))
    doc = SvgDocument(width=w, height=h)
    gid = "grad-1"
    is_radial = grad.get("type") == "radial"
    if is_radial:
        doc.defs.append(build_radial_gradient(gid, grad["stops"],
                                              grad["cx"], grad["cy"], grad["r"]))
    else:
        doc.defs.append(build_linear_gradient(gid, grad["stops"], grad["x1"], grad["y1"],
                                              grad["x2"], grad["y2"]))
    layer = SvgLayer(color="#000000", paths=paths, label="gradient",
                     gradient={"ref": f"url(#{gid})", "stops": grad["stops"],
                               "type": "radial" if is_radial else "linear",
                               **({"cx": grad["cx"], "cy": grad["cy"], "r": grad["r"]}
                                  if is_radial else {})})
    doc.layers.append(layer)
    glyphs = opts.glyph_count or estimate_glyph_count(mask)
    doc.metadata = {"glyphs": glyphs, "mode": opts.mode}
    stats = {
        "engine": "contour",
        "layers": 1,
        "glyphs": doc.metadata["glyphs"],
        "gradient_layers": 1,
        "gradient_stops": len(grad["stops"]),
        "gradient_explained": grad["explained"],
        "node_total": int(sum(f.node_count for f in fitted)),
        "mode_detail": "global-gradient",
    }
    return doc, stats

def vectorize_contour(prepared: PreparedImage, opts: VectorizeOptions) -> tuple[SvgDocument, dict]:
    rgba = prepared.rgba
    mask = prepared.mask
    rgb = rgba[:, :, :3]
    h, w = mask.shape

    glyphs = opts.glyph_count or estimate_glyph_count(mask)
    per_glyph_budget = max(opts.min_nodes, opts.max_nodes)

    # 填色参考：未经 CLAHE/去噪的原始像素（分层标签仍用预处理后图像定）
    raw = prepared.rgb_raw[:, :, ::-1] if getattr(prepared, "rgb_raw", None) is not None else rgb

    # 渐变字优先走单层渐变模式：一个轮廓 + 多色阶渐变填充（先线性后径向）；
    # 渐变本体外还有大占比纯色层（描边）时，输出渐变 + 描边双层
    if opts.gradient_levels >= 8:
        ggrad = _fit_global_gradient(rgb, mask, opts.gradient_levels, rgb_color=raw)
        if ggrad is None:
            ggrad = _fit_radial_gradient(rgb, mask, opts.gradient_levels, rgb_color=raw)
        if ggrad:
            solid = _solid_overlay_mask(rgb, mask)
            if solid is not None:
                return _build_stroked_gradient_document(prepared, opts, ggrad, solid)
            return _build_gradient_document(prepared, opts, ggrad)

    # 主色统计核心区：优先用「完全不透明」的笔画内部（避开抗锯齿混色灰阶）；
    # 核心区过小或过碎时退回普通腐蚀区，避免用稀硫采样定主色
    core_all = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    core_opaque = core_all & (prepared.soft_alpha > 250)
    ok_opaque = False
    if (core_opaque > 0).sum() >= max(200.0, (core_all > 0).sum() * 0.4):
        n, _, stats, _ = cv2.connectedComponentsWithStats(core_opaque.astype(np.uint8), 8)
        largest = float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0
        ok_opaque = largest >= (core_opaque > 0).sum() * 0.5
    core = core_opaque if ok_opaque else core_all
    if (core > 0).sum() < 100:
        core = mask

    palette, counts = _palette_from_foreground(rgb, core, opts.max_colors)
    if len(palette) == 0:
        doc = SvgDocument(width=w, height=h)
        return doc, {"engine": "contour", "layers": 0, "glyphs": 0}

    labels = _label_pixels(rgb, palette, core)
    labels = _propagate_labels(labels, mask, iterations=10)
    labels = _merge_small_layers(labels, palette, float((mask > 0).sum()))
    labels, palette = _remap_palette(labels, palette)

    # 阴影分层：柔边低对比区域合并为独立阴影层，不参与主体分色，
    # 否则柔边投影会被量化成一圈碎灰阶层，节点数爆掉（需求 5.4 保留阴影）
    shadow_result = _detect_soft_shadow(prepared, labels, palette, opts) \
        if opts.shadow_layers != "off" else (None, None)
    shadow_mask, shadow_cut = shadow_result
    shadow_layers: list = []
    shadow_nodes = 0
    shadow_info: dict = {}
    if shadow_mask is not None:
        fg_main = (mask > 0) & (shadow_cut == 0)
        core_main = (core > 0) & fg_main
        if int(core_main.sum()) < 100:
            core_main = fg_main
        palette2, _ = _palette_from_foreground(rgb, core_main, opts.max_colors)
        if len(palette2):
            labels2 = _label_pixels(rgb, palette2, core_main)
            labels2 = _propagate_labels(labels2, fg_main.astype(np.uint8) * 255, iterations=10)
            labels2 = _merge_small_layers(labels2, palette2, float((mask > 0).sum()))
            labels, palette = _remap_palette(labels2, palette2)
        # 主体保护带里残留的阴影碎片并回阴影层：否则它们会变成碎边小层，
        # 把节点数又撑上去（实测 12_shadow 高质量档单字 884 节点）
        lum_rgb = (prepared.rgb_raw if getattr(prepared, "rgb_raw", None) is not None
                   else prepared.rgba[:, :, :3][:, :, ::-1]) @ np.array([0.299, 0.587, 0.114])
        sat_rgb = (prepared.rgb_raw if getattr(prepared, "rgb_raw", None) is not None
                   else prepared.rgba[:, :, :3][:, :, ::-1])
        sat_rgb = sat_rgb.max(2).astype(np.int32) - sat_rgb.min(2).astype(np.int32)
        _bgc = prepared.bg_color
        _bg_lum_s = float(np.dot(_bgc, [0.299, 0.587, 0.114])) if _bgc is not None \
            else float(np.median(lum_rgb[mask == 0]))
        _body_lum_s = float(np.median(lum_rgb[(labels == max(
            range(len(palette)), key=lambda j: int((labels == j).sum())))])) if len(palette) else _bg_lum_s
        grown = cv2.dilate((shadow_mask > 0).astype(np.uint8), np.ones((3, 3), np.uint8),
                           iterations=8).astype(bool)
        drop = []
        for i in range(len(palette)):
            lm = labels == i
            area_i = int(lm.sum())
            if not (0 < area_i <= 6000):
                continue
            lm_lum = float(np.median(lum_rgb[lm]))
            toward = (lm_lum - _bg_lum_s) * (_body_lum_s - _bg_lum_s) > 0
            if (float(sat_rgb[lm].mean()) <= 45 and abs(lm_lum - _bg_lum_s) <= 135
                    and toward and bool((lm & grown).any())):
                drop.append(i)
        if drop:
            extra = np.isin(labels, np.asarray(drop, dtype=labels.dtype))
            shadow_mask = np.where(extra, 255, shadow_mask).astype(np.uint8)
            labels = np.where(extra, -1, labels).astype(labels.dtype)
            labels, palette = _remap_palette(labels, palette)
        shadow_layers, shadow_nodes, shadow_info = _build_shadow_layer(
            prepared, shadow_mask, opts)

    min_area = float(max(6.0, mask.sum() / 255.0 * 0.00012))
    fg_area = float((mask > 0).sum())
    speckle = max(float(opts.filter_speckle), fg_area * 0.00035)
    node_cap = max(opts.min_nodes, opts.max_nodes)
    protect = _bg_hole_mask(mask)

    def build(tolerance: float, merge_ratio: float):
        labels_local, palette_local = labels, palette
        if merge_ratio > 0:
            labels_local = _merge_small_layers(labels, palette, fg_area, min_ratio=merge_ratio)
            labels_local, palette_local = _remap_palette(labels_local, palette)
        built: list[tuple[float, SvgLayer, dict]] = []
        total_nodes = 0
        # 先按面积排序：小层稍后绘制且向外扩 1px，抵消层间缝隙
        sizes = []
        for i in range(len(palette_local)):
            lm = _clean_layer_mask((labels_local == i).astype(np.uint8), min_area)
            sizes.append((float((lm > 0).sum()), i, lm))
        sizes.sort(key=lambda t: (-t[0], t[1]))
        # 节点预算按非空图层数分摊：总节点控制在「单字上限 × 字形数」以内
        nonempty = sum(1 for area, _, _ in sizes if area >= 1)
        layer_budget = max(opts.min_nodes,
                           min(node_cap, int(node_cap * glyphs / max(1, nonempty))))
        for rank, (area, i, layer_mask) in enumerate(sizes):
            if area < 1:
                continue
            if rank > 0:
                layer_mask = _clean_layer_mask(layer_mask, min_area, grow=1, clip=mask)
            if opts.preserve_source_gaps:
                layer_mask = np.where(mask > 0, layer_mask, 0).astype(np.uint8)
            paths, fitted = _fit_layer_paths(layer_mask, opts, layer_budget,
                                             speckle_min_area=speckle, tolerance=tolerance,
                                             protect_mask=protect)
            if not paths:
                continue
            nodes = int(sum(f.node_count for f in fitted))
            total_nodes += nodes
            # 图层颜色从原始像素取样：只取贴近本层主色的实心像素，
            # 避免层间/背景过渡带的混色把均值拉偏（否则黄字被描边拉暗、
            # 灰层被面色拉红）
            color_hex = rgb_to_hex(palette_local[i][::-1])
            solid = (layer_mask > 0) & (prepared.soft_alpha > 250)
            px_all = raw[solid] if int(solid.sum()) >= 30 else raw[layer_mask > 0]
            if len(px_all):
                ref = palette_local[i].astype(np.float64)   # BGR
                dist = np.linalg.norm(px_all.astype(np.float64) - ref, axis=1)
                near = px_all[dist <= 45]
                px = near if len(near) >= 30 else px_all
                color_hex = rgb_to_hex(px.mean(axis=0)[::-1])
            spot_hit = None

            gradient = None
            if area / max(1.0, fg_area) > 0.35 and opts.gradient_levels >= 8:
                grad = _try_linear_gradient(rgb, layer_mask, opts.gradient_levels, rgb_color=raw)
                if grad:
                    gid = f"grad-{len(built) + 1}"
                    gradient = grad
                    gradient["ref"] = f"url(#{gid})"

            layer = SvgLayer(color=color_hex, paths=paths,
                             label=("gradient" if gradient else color_hex),
                             gradient=gradient)
            built.append((area, layer, {
                "color": color_hex, "area": int(area), "paths": len(paths),
                "nodes": nodes, "gradient": bool(gradient),
                "spot": spot_hit,
            }))
        built.sort(key=lambda t: -t[0])
        return built, total_nodes

    # 节点超限时逐步放宽容差并归并小色层，把单字节点数压回指标区间。
    # 先只放容差（保颜色与细节），仍超限再逐级归并小色层；归并会减少
    # 路径数，因此即使当前路径下限（路径数× min_nodes）高于预算也先试；
    # 平均节点已接近路径下限时不再重拟合，避免无意义开销
    schedule = [(1.7, 0.008), (2.9, 0.024), (4.9, 0.05)]
    tolerance = opts.tolerance
    merge_ratio = 0.008
    built, total_nodes = build(tolerance, merge_ratio)
    best_built, best_nodes = built, total_nodes
    attempts = 0
    path_count = sum(int(info["paths"]) for _, _, info in built)
    while ((total_nodes + shadow_nodes) / max(1, glyphs) > node_cap and attempts < len(schedule)
           and path_count > 0):
        attempts += 1
        tolerance = opts.tolerance * schedule[attempts - 1][0]
        merge_ratio = schedule[attempts - 1][1]
        prev_total = total_nodes
        built, total_nodes = build(tolerance, merge_ratio)
        if total_nodes < best_nodes:
            best_built, best_nodes = built, total_nodes
        path_count = sum(int(info["paths"]) for _, _, info in built)
        # 本轮重试收益过小（节点保留 85% 以上）且已进预算，提前收工；
        # 没进预算就把重试档跑完（归并档往往才是真正把节点压下来的一步）
        if ((prev_total - total_nodes) < prev_total * 0.15
                and (total_nodes + shadow_nodes) / max(1, glyphs) <= node_cap):
            break
    # 取节点最少的一版作为最终结果
    if best_nodes < total_nodes:
        built, total_nodes = best_built, best_nodes

    layers_info = [info for _, _, info in built]
    doc = SvgDocument(width=w, height=h)
    for sl in shadow_layers:
        doc.layers.append(sl)      # 阴影垫底（外浅内深），主体层绘在上面
    for _, layer, info in built:
        if layer.gradient:
            gid = layer.gradient["ref"].split("#")[1].rstrip(")")
            doc.defs.append(build_linear_gradient(gid, layer.gradient["stops"],
                                                  layer.gradient["x1"], layer.gradient["y1"],
                                                  layer.gradient["x2"], layer.gradient["y2"]))
            info["gradient_stops"] = len(layer.gradient["stops"])
        doc.layers.append(layer)

    doc.metadata = {"glyphs": glyphs, "mode": opts.mode}
    stats = {
        "engine": "contour",
        "layers": len(doc.layers),
        "glyphs": glyphs,
        "layer_detail": layers_info,
        "gradient_layers": sum(1 for i in layers_info if i["gradient"]),
        "palette_size": len(palette),
        "node_total": int(total_nodes + shadow_nodes),
        "shadow_layers": len(shadow_layers),
        "shadow": shadow_info,
        "tolerance_used": round(tolerance, 3),
        "refit_attempts": attempts,
    }
    return doc, stats


# ---------------------------------------------------------------------------
# 引擎二：vtracer（渐变 / 复杂多色，支持节点预算复核）
# ---------------------------------------------------------------------------

def _pick_sentinel(rgb_fg: np.ndarray) -> tuple[int, int, int]:
    """选一个与前景颜色距离最远的哨兵色，用作透明背景占位。"""
    candidates = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
    if len(rgb_fg) == 0:
        return candidates[0]
    sample = rgb_fg[:: max(1, len(rgb_fg) // 4000)].astype(np.float32)
    lab = cv2.cvtColor(sample.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    mean = lab.mean(axis=0)
    best, best_d = candidates[0], -1.0
    for c in candidates:
        bgr = np.uint8([[list(c[::-1])]])
        cl = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
        d = float(np.linalg.norm(cl - mean))
        if d > best_d:
            best, best_d = c, d
    return best


def _vtracer_svg(rgba: np.ndarray, opts: VectorizeOptions) -> tuple[str, tuple[int, int, int]]:
    import vtracer

    img = rgba.copy()
    if opts.posterize and opts.posterize >= 3:
        bgr = img[:, :, :3].astype(np.float32)
        levels = int(opts.posterize) - 1
        bgr = np.round(bgr / 255.0 * levels) / levels * 255.0
        img[:, :, :3] = bgr.astype(np.uint8)

    # vtracer 不输出 alpha，透明背景改成哨兵色占位，事后再剔除
    sentinel = _pick_sentinel(img[:, :, :3][img[:, :, 3] > 200])
    transparent = img[:, :, 3] < 250
    img[transparent] = [sentinel[2], sentinel[1], sentinel[0], 255]

    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("图像编码失败")
    layer_diff = int(max(6, min(80, round(255.0 / max(4, opts.gradient_levels) * 0.9))))
    svg = vtracer.convert_raw_image_to_svg(
        buf.tobytes(), "png",
        colormode="color",
        hierarchical="cutout",
        mode="spline",
        filter_speckle=int(opts.filter_speckle),
        color_precision=int(opts.color_precision),
        layer_difference=layer_diff,
        corner_threshold=60,
        length_threshold=4.0,
        max_iterations=10,
        splice_threshold=45,
        path_precision=2,
    )
    return svg, sentinel


def vectorize_vtracer(prepared: PreparedImage, opts: VectorizeOptions) -> tuple[SvgDocument, dict]:
    svg_text, sentinel = _vtracer_svg(prepared.rgba, opts)
    glyphs = opts.glyph_count or estimate_glyph_count(prepared.mask)
    cap = max(opts.min_nodes, opts.max_nodes)
    scale = 1.0
    doc = None
    for _ in range(3):
        doc = _normalize_svg(svg_text, prepared, opts, sentinel, budget_scale=scale)
        total = sum(_count_nodes(l.paths) for l in doc.layers)
        if total / max(1, glyphs) <= cap:
            break
        scale *= 0.45
    stats = {
        "engine": "vtracer",
        "layers": len(doc.layers),
        "glyphs": glyphs,
        "sentinel": "#%02x%02x%02x" % sentinel,
        "node_total": int(sum(_count_nodes(l.paths) for l in doc.layers)),
        "budget_scale": round(scale, 3),
        "layer_detail": [{"paths": len(l.paths),
                          "gradient": bool(l.gradient)} for l in doc.layers],
    }
    return doc, stats


def _count_nodes(paths: list[str]) -> int:
    from .svgutil import count_nodes_from_d
    return int(sum(count_nodes_from_d(d) for d in paths))


def _normalize_svg(svg_text: str, prepared: PreparedImage, opts: VectorizeOptions,
                   sentinel: tuple[int, int, int] | None = None,
                   budget_scale: float = 1.0) -> SvgDocument:
    """把外部引擎输出重排进标准模板，剔除背景层并按需压缩节点。"""
    from .pathparse import SvgRaster, _grad_id, translate_path
    raster = SvgRaster(svg_text)

    sentinel_lab = None
    if sentinel is not None:
        bgr = np.uint8([[list(sentinel[::-1])]])
        sentinel_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)

    h, w = prepared.mask.shape
    doc = SvgDocument(width=w, height=h)
    glyphs = opts.glyph_count or estimate_glyph_count(prepared.mask)
    budget = max(opts.min_nodes, int(max(opts.min_nodes, opts.max_nodes) * budget_scale))

    for layer in raster.layers:
        fill = layer["fill"]
        new_paths: list[str] = []
        for item in layer["d"]:
            d, offset = item if isinstance(item, tuple) else (item, (0.0, 0.0))
            if sentinel_lab is not None:
                rgb = np.uint8([[list(_hex_triplet(fill)[::-1])]]) if fill.startswith("#") else None
                if rgb is not None:
                    lab = cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)
                    if float(np.linalg.norm(lab - sentinel_lab)) < 16.0:
                        continue  # 背景占位色，剔除
            polys = parse_path(d, offset)
            if not polys:
                continue
            total_nodes = sum(len(p) for p in polys)
            if not opts.enforce_node_budget or total_nodes <= budget:
                new_paths.append(translate_path(d, offset))
                continue
            refit: list[str] = []
            total_pts = max(1, sum(len(p) for p in polys))
            for poly in polys:
                if len(poly) < 4:
                    continue
                sub_budget = max(opts.min_nodes, int(budget * len(poly) / total_pts))
                fc = fit_contour(poly, tolerance=max(0.8, opts.tolerance),
                                 corner_angle=opts.corner_angle,
                                 max_nodes=sub_budget, min_nodes=opts.min_nodes,
                                 closed=True, smooth=opts.smooth)
                refit.append(contour_to_path_d(fc))
            new_paths.append(" ".join(refit) if refit else d)
        if not new_paths:
            continue
        grad = None
        if fill.startswith("url(") and layer["stops"]:
            g = layer["stops"]
            gid = _grad_id(fill)
            doc.defs.append(build_linear_gradient(gid, g["stops"], g["x1"], g["y1"], g["x2"], g["y2"]))
            grad = {"ref": fill, "stops": g["stops"]}
        doc.layers.append(SvgLayer(color=fill if not grad else "#000000",
                                   paths=new_paths, gradient=grad,
                                   opacity=layer["opacity"],
                                   label="gradient" if grad else fill))
    doc.metadata = {"glyphs": glyphs, "mode": opts.mode}
    return doc


def _hex_triplet(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    if len(v) < 6:
        return (255, 255, 255)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


# ---------------------------------------------------------------------------
# 引擎选择与统一入口
# ---------------------------------------------------------------------------

def choose_engine(prepared: PreparedImage, opts: VectorizeOptions) -> str:
    """auto 模式默认走自研轮廓引擎，内建复核不达标时由 vectorize() 切 vtracer。"""
    if opts.engine in ("contour", "vtracer"):
        return opts.engine
    return "contour"


def resolve_profile(prepared: PreparedImage, opts: VectorizeOptions) -> tuple[VectorizeOptions, dict]:
    """按素材属性路由参数（需求六：引擎与参数可插拔可替换）。

    可测属性：主体轮廓的正交程度（几何字 vs 书法手写），决定直线/正交
    吸附开关与角容差；阴影分层、渐变、描边在矢量化途中按属性自动分流。
    """
    bg = prepared.bg_color
    raw = prepared.rgb_raw if getattr(prepared, "rgb_raw", None) is not None \
        else prepared.rgba[:, :, :3][:, :, ::-1]
    lum = raw @ np.array([0.299, 0.587, 0.114])
    bg_lum = float(np.dot(bg, [0.299, 0.587, 0.114])) if bg is not None \
        else float(np.median(lum[prepared.mask == 0]))
    crisp = (np.abs(lum - bg_lum) > 135).astype(np.uint8) * 255
    score = axis_alignment_score(prepared.mask, crisp=crisp)
    snap_deg = 0.0
    if opts.snap_axis == "on" or (opts.snap_axis == "auto" and score >= 0.42):
        snap_deg = 6.5
    profile = {
        "axis_alignment": round(score, 3),
        "snap_deg": snap_deg,
        "snap_axis": opts.snap_axis,
        "shadow_layers": opts.shadow_layers,
    }
    if snap_deg != opts.snap_deg:
        opts = replace(opts, snap_deg=snap_deg)
    return opts, profile


def vectorize(prepared: PreparedImage, opts: VectorizeOptions) -> tuple[SvgDocument, dict]:
    """统一入口。auto 模式先跑自研引擎，达不到还原度线时再与 vtracer 对比取优。"""
    from .metrics import quick_shape_iou
    from .svgutil import assign_glyph_groups

    opts, profile = resolve_profile(prepared, opts.resolve_spot_options())
    engine = choose_engine(prepared, opts)
    if engine == 'contour':
        from .direct import monochrome_document
        direct = monochrome_document(prepared, opts)
        if direct is None:
            from .shadow_model import shadow_document
            direct = shadow_document(prepared, opts)
        if direct:
            doc, stats = direct
        else:
            doc, stats = vectorize_contour(prepared, opts)
        if doc.layers and opts.glyph_count == 1 and stats.get('node_total',0) > opts.max_nodes:
            from .brush import brush_candidate
            from .candidate_guard import compare_candidate
            compact = brush_candidate(prepared, opts)
            if compact:
                candidate,candidate_stats = compact
                diagnostic = compare_candidate(prepared,opts,doc,stats,candidate,candidate_stats)
                # Record texture even when a shape/colour gate already rejected
                # the candidate. resvg supports long gradients that MuPDF cannot
                # render; the independent PDF geometry gate remains mandatory.
                from .native_render import render_svg
                from .texture import texture_agreement
                texture = texture_agreement(prepared, render_svg(candidate.to_svg()))
                texture['renderer'] = 'resvg'
                diagnostic['texture'] = texture
                if not texture['passed']:
                    diagnostic['rejected'].append('local_brush_texture')
                if not diagnostic['rejected']:
                    candidate_stats['brush_comparison'] = diagnostic
                    doc,stats = candidate,candidate_stats
                else:
                    stats['brush_comparison'] = diagnostic
        if doc.layers and opts.refine_source_edges and not stats.get('brush_candidate') and not stats.get('source_silhouette') and not stats.get('source_shadow_model'):
            from .source_edges import select_source_edges
            doc, stats = select_source_edges(prepared, opts, doc, stats)
            from .source_tones import select_source_tones
            doc, stats = select_source_tones(prepared, opts, doc, stats)
        if not doc.layers:  # 颜色过于复杂时自动切 vtracer
            doc, stats = vectorize_vtracer(prepared, opts)
            stats["fallback_from"] = "contour"
        elif opts.engine == "auto" and opts.auto_verify and not (stats.get('source_tones_selected') or stats.get('source_silhouette') or stats.get('source_shadow_model')):
            score = quick_shape_iou(prepared, doc.to_svg())
            stats["self_check_iou"] = round(score, 4)
            if score < 0.93:
                alt_doc, alt_stats = vectorize_vtracer(prepared, opts)
                alt_score = quick_shape_iou(prepared, alt_doc.to_svg())
                alt_stats["self_check_iou"] = round(alt_score, 4)
                stats["alt_iou"] = round(alt_score, 4)
                from .pathparse import SvgRaster
                from .topology import compare_topology
                base_pixels = SvgRaster(doc.to_svg()).rasterize(scale=1)
                alt_pixels = SvgRaster(alt_doc.to_svg()).rasterize(scale=1)
                base_topology = compare_topology(prepared.mask > 127, base_pixels[:, :, 3] > 127)
                alt_topology = compare_topology(prepared.mask > 127, alt_pixels[:, :, 3] > 127)
                def defects(report):
                    return sum(sum(region['status'] != 'pass' for region in report[k]['regions'])
                               + len(report[k]['extra_regions']) for k in ('components','holes'))
                def composite_error(pixels):
                    alpha = pixels[:, :, 3:4].astype(float) / 255
                    bg = np.asarray(prepared.bg_color if prepared.bg_color is not None else (255,255,255))
                    painted = pixels[:, :, :3] * alpha + bg * (1-alpha)
                    raw = prepared.rgb_raw
                    return float(np.abs(painted[prepared.mask > 127]-raw[prepared.mask > 127]).mean()/255)
                reasons = []
                if stats.get('shadow_layers', 0) and not alt_stats.get('shadow_layers', 0):
                    reasons.append('editable_shadow_layers')
                if defects(alt_topology) > defects(base_topology):
                    reasons.append('local_topology')
                if composite_error(alt_pixels) > composite_error(base_pixels):
                    reasons.append('composite_color')
                cap = opts.max_nodes * max(1, opts.glyph_count or stats.get('glyphs', 1))
                if alt_stats.get('node_total', 0) > max(cap, stats.get('node_total', 0)):
                    reasons.append('node_budget')
                if alt_score > score + 0.015 and not reasons:
                    doc, stats = alt_doc, alt_stats
                    stats['switched_to'] = 'vtracer'
                else:
                    stats['kept'] = 'contour'
                    stats['fallback_rejections'] = reasons
    else:
        doc, stats = vectorize_vtracer(prepared, opts)
    stats["engine_selected"] = engine
    stats["engine_used"] = stats.get("engine")
    stats["profile"] = profile

    # Match after engine selection so contour and VTracer share the same branch.
    # Matching is intentional recoloring and must not affect source-fit selection.
    if opts.spot_matching:
        from .spot_colors import nearest_color
        changed = 0
        mappings = {}
        for layer in doc.layers:
            if layer.gradient:
                continue
            source = layer.color
            target = nearest_color(source, opts.spot_colors)
            layer.color = target
            if target != source:
                changed += 1
                mappings[source, target] = mappings.get((source, target), 0) + 1
        details = stats.get('layer_detail', [])
        if len(details) == len(doc.layers):
            for layer, detail in zip(doc.layers, details):
                detail.update(color=layer.color, spot=None if layer.gradient else layer.color)
        stats['spot_matching'] = {
            'enabled': True, 'matched_layers': changed, 'palette': opts.spot_colors,
            'gradients_unchanged': True,
            'mapping_counts': [{'from': a, 'to': b, 'layers': n} for (a, b), n in sorted(mappings.items())],
        }

    # 按字分组：每字一个 <g>，内含该字的全部颜色子层（需求 5.2 / 六节输出模板）
    glyphs = int(opts.glyph_count or doc.metadata.get("glyphs")
                 or estimate_glyph_count(prepared.mask))
    groups = assign_glyph_groups(doc, glyphs)
    if groups:
        doc.glyph_groups = groups
        if opts.glyph_text:
            chars = [c for c in opts.glyph_text if not c.isspace() and c != "/"]
            if len(chars) == len(groups):
                doc.glyph_labels = chars
        grouped = doc._glyph_paint_order_is_safe()
        stats["glyph_groups"] = len(groups) if grouped else 0
        stats["grouping_mode"] = "glyph" if grouped else "layer"
    return doc, stats


def vectorize_bytes(data: bytes, opts: VectorizeOptions | None = None,
                    preprocess_options: dict | None = None) -> dict:
    """字节流入口：解码 -> 预处理 -> 矢量化。返回 {svg, stats, quality, ...}。"""
    opts = opts or VectorizeOptions()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法解码图片，请确认是 PNG/JPG/WEBP 格式")

    if opts.mode in {'fidelity', 'pixel_exact'}:
        from .service import convert_bytes
        result = convert_bytes('image.png', data, opts, with_preview=False, strict_input=False)
        if result.error:
            raise ValueError(result.error)
        return {'svg': result.svg, 'stats': result.stats, 'quality': result.quality, 'size': result.size}

    p_opts = {"remove_background": True, "denoise": True, "enhance": True, "deskew": True}
    if preprocess_options:
        p_opts.update(preprocess_options)
    prepared = preprocess(img, **p_opts)
    doc, stats = vectorize(prepared, opts)
    return {
        "svg": doc.to_svg(),
        "stats": stats,
        "quality": prepared.quality.to_dict(),
        "size": {"width": doc.width, "height": doc.height},
    }
