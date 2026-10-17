"""SVG 路径构造工具：轮廓简化、角点保护、曲线拟合、节点统计、文档装配。

坐标体系与原图像素一一对应，输出 SVG 使用 viewBox="0 0 W H"。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence
from xml.sax.saxutils import escape as _xml_escape, quoteattr

import cv2
import numpy as np


# --------------------------------------------------------------------------
# 几何基础
# --------------------------------------------------------------------------

def polygon_area(pts: np.ndarray) -> float:
    """鞋带公式，返回有符号面积。"""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker 简化（闭合轮廓首尾点由调用方保证）。

    用切片代替逐段建索引，避免大量临时数组；点到「线段」的距离，与标典实现一致。
    （不用 cv2.approxPolyDP：它在细小轮廓上会多切特征，实测会让内孔变形甚至丢失。）
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 3 or epsilon <= 0:
        return pts

    xs, ys = pts[:, 0], pts[:, 1]
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        s, e = stack.pop()
        if e <= s + 1:
            continue
        dx = xs[e] - xs[s]
        dy = ys[e] - ys[s]
        seg_len = math.hypot(dx, dy)
        rel_x = xs[s + 1:e] - xs[s]
        rel_y = ys[s + 1:e] - ys[s]
        if seg_len < 1e-9:
            dist = np.hypot(rel_x, rel_y)
        else:
            dist = np.abs(dx * rel_y - dy * rel_x) / seg_len
        k = int(np.argmax(dist))
        if dist[k] > epsilon:
            pivot = s + 1 + k
            keep[pivot] = True
            stack.append((s, pivot))
            stack.append((pivot, e))
    return pts[keep]


def turning_angles(pts: np.ndarray) -> np.ndarray:
    """闭合多边形每个顶点的转角（度），逆时针轮廓上凸点为正。"""
    n = len(pts)
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    v1 = pts - prev
    v2 = nxt - pts
    a1 = np.arctan2(v1[:, 1], v1[:, 0])
    a2 = np.arctan2(v2[:, 1], v2[:, 0])
    d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    return np.degrees(d)


def mark_corners(pts: np.ndarray, angle_threshold: float = 62.0) -> np.ndarray:
    """返回布尔数组，True 表示该顶点是角点（曲线在此断开，不做平滑跨越）。"""
    ang = turning_angles(pts)
    return np.abs(ang) >= angle_threshold


def smooth_bumps(pts: np.ndarray, closed: bool = True, max_dev: float = 1.6,
                 max_edge: float = 2.6, max_turn: float = 62.0,
                 min_area: float = 60.0) -> np.ndarray:
    """抹掉像素台阶留下的 1~2px 小凸起，真角点保住。

    抗锯齿与分色边界是像素级阶梯，RDP 会把台阶的拐点当形状细节留下
    （弦高 0.4~1.4px、邻边 ≤2.6px、转角 <max_turn 的顶点簇），Catmull-Rom
    又逐点插值过去，放大看就是笔画边上的小疙瘩。判定签名：弦高在
    (0.18, max_dev)、邻边短、转角未到角点线（max_turn 传各档角点阈值，
    尖角、楷体顿笔永不削）。小面积轮廓（装饰点、细环）整条跳过。
    """
    out = np.asarray(pts, dtype=np.float64)
    for _ in range(3):
        n = len(out)
        if n < 8:
            break
        if abs(polygon_area(out)) < min_area:
            break
        prev = np.roll(out, 1, axis=0)
        nxt = np.roll(out, -1, axis=0)
        v1 = out - prev
        v2 = nxt - out
        chord = nxt - prev
        cl = np.maximum(np.linalg.norm(chord, axis=1), 1e-6)
        dev = np.abs(v1[:, 0] * chord[:, 1] - v1[:, 1] * chord[:, 0]) / cl
        edge = np.maximum(np.hypot(v1[:, 0], v1[:, 1]), np.hypot(v2[:, 0], v2[:, 1]))
        cand = (dev > 0.18) & (dev < max_dev) & (edge <= max_edge)
        if not closed:
            cand[0] = cand[-1] = False
        if not cand.any():
            break
        # 转角只对候选点算（arctan2 是主要开销）
        idx = np.where(cand)[0]
        a1 = np.arctan2(v1[idx, 1], v1[idx, 0])
        a2 = np.arctan2(v2[idx, 1], v2[idx, 0])
        turn = np.abs(np.degrees((a2 - a1 + np.pi) % (2 * np.pi) - np.pi))
        kill = np.zeros(n, dtype=bool)
        kill[idx] = turn < max_turn
        if not kill.any():
            break
        keep = ~kill
        if int(keep.sum()) < 6:
            break
        out = out[keep]
    return out


def snap_orthogonal(pts: np.ndarray, closed: bool = True, tol_deg: float = 6.5,
                    min_len: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把近水平/近竖直的连续边拉成真正的水平/垂直边（直线/正交吸附）。

    像素轮廓自带抗锯齿抖动，直接拟合会把「横不平竖不直」带进矢量结果；
    这里把容差角度内的连续边合成链，链上顶点的法向坐标统一（整链拉平），
    拉平后的链在输出时合并为一条直线段，节点数也随之下降。
    书法/手写类素材不适合吸附，由调用方关掉（tol_deg=0）。

    返回 (新点集, 每条边是否直线段, 每条边的轴向 0=自由 1=水平 2=垂直)。
    """
    n = len(pts)
    seg_n = n if closed else n - 1
    if n < 4 or seg_n < 2 or tol_deg <= 0:
        return pts, np.zeros(max(seg_n, 0), dtype=bool), np.zeros(max(seg_n, 0), dtype=np.int8)

    idx = np.arange(seg_n)
    nxt = (idx + 1) % n if closed else idx + 1
    v = pts[nxt] - pts[idx]
    dx, dy = v[:, 0], v[:, 1]
    length = np.hypot(dx, dy)
    t = math.tan(math.radians(tol_deg))
    horiz = (np.abs(dy) <= np.abs(dx) * t) & (length >= min_len)
    vert = (np.abs(dx) <= np.abs(dy) * t) & (length >= min_len)
    axis = np.where(horiz, 1, np.where(vert, 2, 0)).astype(np.int8)

    # 找出连续同轴的链（首尾可环绕相接）
    runs: list[tuple[int, int, int]] = []
    i = 0
    while i < seg_n:
        if axis[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < seg_n and axis[j + 1] == axis[i]:
            j += 1
        runs.append((i, j, int(axis[i])))
        i = j + 1
    if closed and len(runs) > 1 and runs[0][0] == 0 and runs[-1][1] == seg_n - 1 \
            and runs[0][2] == runs[-1][2]:
        head, tail = runs[0], runs[-1]
        runs = runs[1:-1] + [(tail[0], head[1] + seg_n, head[2])]

    out = pts.copy()
    for a0, a1, a in runs:
        segs_idx = [(k % seg_n) for k in range(a0, a1 + 1)]
        verts = [k % n for k in range(a0, a1 + 2)]
        P = pts[verts]
        # 最小二乘拟合链的方向：拉直但保留设计斜体，只有真横平竖直的链
        # （与轴向偏差 ≤2°）才强制成水平/垂直
        w = np.array([max(length[k], 1e-6) for k in segs_idx] + [1e-6])
        w = w / w.sum()
        c = (P * w[:, None]).sum(axis=0)
        X = (P - c) * np.sqrt(w)[:, None]
        # 2×2 协方差解析主方向（比逐链 SVD 快一个量级，链多时不卡）
        cov = X.T @ X
        ang_dir = 0.5 * math.atan2(2.0 * cov[0, 1], cov[0, 0] - cov[1, 1])
        direction = np.array([math.cos(ang_dir), math.sin(ang_dir)])
        ang = abs(math.degrees(math.atan2(direction[1], direction[0]))) % 90.0
        ang = min(ang, 90.0 - ang)
        if ang <= 2.0:
            if a == 1:      # 水平链：统一 y
                ys = np.array([0.5 * (pts[k % n][1] + pts[(k + 1) % n][1]) for k in segs_idx])
                y = float((ys * w[:len(segs_idx)]).sum() / w[:len(segs_idx)].sum())
                for vv in verts:
                    out[vv][1] = y
            else:            # 垂直链：统一 x
                xs = np.array([0.5 * (pts[k % n][0] + pts[(k + 1) % n][0]) for k in segs_idx])
                x = float((xs * w[:len(segs_idx)]).sum() / w[:len(segs_idx)].sum())
                for vv in verts:
                    out[vv][0] = x
        else:
            # 斜而直的边：沿拟合直线投影，拉直不拉平
            proj = (P - c) @ direction
            moved = c + np.outer(proj, direction)
            for j, vv in enumerate(verts):
                out[vv] = moved[j]

    straight = axis > 0
    return out, straight, axis


def catmull_rom_to_bezier(points: np.ndarray, closed: bool = True,
                          smooth_at: np.ndarray | None = None,
                          alpha: float = 0.5) -> list[tuple]:
    """把稀疏多边形转成三次贝塞尔段列表（向量化实现）。

    用向心 Catmull-Rom（alpha=0.5）计算切线，并对切线长度做弦长限制，
    避免均匀参数化在尖角附近过冲形成自交小环（自交在浏览器里会渲染成
    细长孔洞与尖刺）。smooth_at 标记允许平滑的点，角点处切线沿弦方向。
    """
    n = len(points)
    if n < 2:
        return []
    if not closed and n == 2:
        return [(points[0], None, None, points[1])]

    if closed:
        idx_prev = (np.arange(n) - 1) % n
        idx0 = np.arange(n)
        idx1 = (np.arange(n) + 1) % n
        idx_next = (np.arange(n) + 2) % n
    else:
        idx0 = np.arange(n - 1)
        idx1 = idx0 + 1
        idx_prev = np.maximum(idx0 - 1, 0)
        idx_next = np.minimum(idx1 + 1, n - 1)

    p_prev = points[idx_prev].astype(np.float64)
    p0 = points[idx0].astype(np.float64)
    p1 = points[idx1].astype(np.float64)
    p_next = points[idx_next].astype(np.float64)

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(b - a, axis=1) ** alpha
        return np.maximum(d, 1e-6)

    t1 = dist(p_prev, p0)
    t2 = t1 + dist(p0, p1)
    t3 = t2 + dist(p1, p_next)

    span01 = (t2 - t1)[:, None]
    m1 = (p1 - p0) / (t1[:, None]) - (p1 - p_prev) / (t2[:, None]) + (p1 - p0) / span01
    m2 = (p1 - p0) / span01 - (p_next - p0) / ((t3 - t1)[:, None]) + (p_next - p1) / ((t3 - t2)[:, None])
    c1 = p0 + m1 * span01 / 3.0
    c2 = p1 - m2 * span01 / 3.0

    chord = p1 - p0
    chord_len = np.linalg.norm(chord, axis=1)

    if smooth_at is not None:
        s0 = smooth_at[idx0].astype(bool)
        s1 = smooth_at[idx1].astype(bool)
        c1 = np.where(s0[:, None], c1, p0 + chord / 3.0)
        c2 = np.where(s1[:, None], c2, p1 - chord / 3.0)

    # 控制点限制在弦中点为圆心的半弦圆内，避免过冲与自交
    limit = (chord_len * 0.5)[:, None]
    for base, ctrl, sgn in ((p0, c1, 1.0), (p1, c2, -1.0)):
        v = ctrl - base
        d = np.linalg.norm(v, axis=1)[:, None]
        scale = np.where(d > limit, limit / np.maximum(d, 1e-9), 1.0)
        ctrl = base + v * scale
        if sgn > 0:
            c1 = ctrl
        else:
            c2 = ctrl

    if not closed:
        c1 = np.asarray(c1, dtype=np.float64)
        c2 = np.asarray(c2, dtype=np.float64)
    segs = []
    for i in range(len(p0)):
        if not closed and i == len(p0) - 1:
            segs.append((p0[i], None, None, p1[i]))
        else:
            segs.append((p0[i], c1[i], c2[i], p1[i]))
    return segs


# --------------------------------------------------------------------------
# 轮廓 -> 路径
# --------------------------------------------------------------------------

def offset_polygon(pts: np.ndarray, dist: float) -> np.ndarray:
    """沿法线把闭合多边形向外扩大 dist 像素（dist 为负则收缩）。

    cv2.findContours 返回的是像素中心点集，而真实边界（50% 覆盖率）在像素中心外侧约
    0.5px，按几何面积填充时会系统地「瘦一圈」。这里统一补偿：外轮廓向外扩，
    内孔同样向外扩（即孔变大、笔画变粗），与区域膨胀语义一致。
    法线方向用面积实测确定，不依赖坐标系的绕行方向约定。
    """
    n = len(pts)
    if n < 3 or abs(dist) < 1e-9:
        return pts
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    tangent = nxt - prev
    length = np.linalg.norm(tangent, axis=1)
    length[length < 1e-9] = 1.0
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1) / length[:, None]

    base_area = abs(polygon_area(pts))
    if abs(polygon_area(pts + normal * abs(dist))) < base_area:
        normal = -normal
    return pts + normal * dist


@dataclass
class FittedContour:
    segs: list[tuple]
    points: np.ndarray
    area: float
    closed: bool = True

    @property
    def node_count(self) -> int:
        return len(self.segs) + (1 if self.segs else 0)


def _emit_segments(points: np.ndarray, segs: list[tuple], straight: np.ndarray,
                   axis: np.ndarray, closed: bool) -> list[tuple]:
    """直线段输出为 L，同轴相邻直线段合并成一条，节点数随之下沉。"""
    out: list[tuple] = []
    for k, (p0, c1, c2, p1) in enumerate(segs):
        is_line = bool(straight[k]) if k < len(straight) else False
        a = int(axis[k]) if k < len(axis) else 0
        if is_line and a:
            if out and out[-1][1] is None and out[-1][4] == a:
                out[-1] = (out[-1][0], None, None, p1, a)
            else:
                out.append((p0, None, None, p1, a))
        else:
            out.append((p0, c1, c2, p1, 0))
    if closed and len(out) >= 2 and out[0][1] is None and out[-1][1] is None \
            and out[0][4] and out[0][4] == out[-1][4]:
        first = out.pop(0)
        last = out.pop()
        out.append((last[0], None, None, first[3], first[4]))
    return [(a, b, c, d) for a, b, c, d, _ in out]


def fit_contour(contour: np.ndarray, tolerance: float = 1.0,
                corner_angle: float = 62.0, max_nodes: int | None = None,
                min_nodes: int = 4, closed: bool = True,
                smooth: bool = True, snap_deg: float = 0.0,
                deburr_dev: float = 1.6) -> FittedContour:
    """把一条像素轮廓拟合成贝塞尔路径，节点数受上限约束。

    tolerance 为允许的形状偏差（像素）。节点超限时自动放宽容差重拟合，
    直到满足上限或达到放开阈值。snap_deg > 0 时启用直线/正交吸附：
    容差角度内的近水平/近竖直边拉成真直线（适合黑体/宋体/Logo 这类
    几何字，书法手写类应传 0）。拟合前先保角去毛刺（smooth_bumps），
    吸附链长门槛 4px，短横也能进链。
    """
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if closed and len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]

    area = abs(polygon_area(pts))
    eps = max(tolerance, 0.35)

    def simplify(base_pts: np.ndarray, eps_v: float) -> np.ndarray:
        if closed:
            ring = np.vstack([base_pts, base_pts[:1]])
            s = rdp(ring, eps_v)
            if len(s) > 1 and np.allclose(s[0], s[-1]):
                s = s[:-1]
        else:
            s = rdp(base_pts, eps_v)
        if snap_deg > 0 and len(s) >= 4:
            # 预算估计用长链门槛；4px 短链门槛只在最终吸附用（链多时逐链开销大）
            s, _, _ = snap_orthogonal(s, closed=closed, tol_deg=snap_deg, min_len=10.0)
        return s

    simplified = simplify(pts, eps)

    if max_nodes:
        guard = 0
        while len(simplified) > max_nodes and guard < 40:
            eps *= 1.28
            guard += 1
            s = simplify(pts, eps)
            if closed and len(s) < 3:
                break
            if not closed and len(s) < 2:
                break
            simplified = s

    if not closed and len(simplified) < 2:
        simplified = pts[[0, -1]]

    if closed and len(simplified) < min_nodes and len(pts) >= min_nodes:
        # 细节太少的轮廓：降低容差补回细节，避免小图形被压成三角形
        s = simplify(pts, eps / 2.5)
        if len(s) > len(simplified):
            simplified = s

    straight, axis = np.zeros(0, dtype=bool), np.zeros(0, dtype=np.int8)
    if snap_deg > 0 and len(simplified) >= 8:
        # 保角去毛刺只做一次（放在重试循环外，避免每次放宽容差都重复开销）；
        # 与吸附同门限：几何字去毛刺，书法手写的笔触毛边保留（需求二）；
        # 转角达到角点阈值的顶点永不削（尖角、顿笔保形）
        simplified = smooth_bumps(simplified, closed=closed, max_dev=deburr_dev,
                                  max_turn=corner_angle)
    if snap_deg > 0 and len(simplified) >= 4:
        simplified, straight, axis = snap_orthogonal(simplified, closed=closed,
                                                     tol_deg=snap_deg, min_len=6.0)

    corners = mark_corners(simplified, corner_angle) if closed and smooth else np.ones(len(simplified), dtype=bool)
    if closed and smooth:
        smooth_at = ~corners
    else:
        smooth_at = np.ones(len(simplified), dtype=bool)

    if smooth:
        segs = catmull_rom_to_bezier(simplified, closed=closed, smooth_at=smooth_at)
    else:
        segs = [(simplified[i], None, None, simplified[i + 1])
                for i in range(len(simplified) - 1)]
    if len(straight):
        segs = _emit_segments(simplified, segs, straight, axis, closed)

    return FittedContour(segs=list(segs), points=simplified, area=area, closed=closed)


_NUM = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")

SHADOW_SOFT_FILTER = ('<filter id="shadow-soft" x="-25%" y="-25%" width="150%" height="150%">'
                      '<feGaussianBlur stdDeviation="2.5"/></filter>')


def _fmt(v: float) -> str:
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s else "0"


def contour_to_path_d(fc: FittedContour) -> str:
    """转成 SVG d 属性字符串。"""
    if not fc.segs:
        return ""
    parts: list[str] = []
    start = fc.segs[0][0]
    parts.append(f"M{_fmt(start[0])} {_fmt(start[1])}")
    for (p0, c1, c2, p1) in fc.segs:
        if c1 is None:
            parts.append(f"L{_fmt(p1[0])} {_fmt(p1[1])}")
        else:
            parts.append(
                f"C{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} "
                f"{_fmt(p1[0])} {_fmt(p1[1])}"
            )
    if fc.closed:
        parts.append("Z")
    return "".join(parts)


def count_nodes_from_d(d: str) -> int:
    """统计路径锚点数（on-curve 点），即 Illustrator 里选中的可编辑节点数。

    M/L/H/V/C/S/Q/T/A 每个绘制指令落一个锚点，Z 不计。
    """
    cmds = re.findall(r"[MmLlHhVvCcSsQqTtAa]", d)
    return len(cmds)


# --------------------------------------------------------------------------
# 文档装配
# --------------------------------------------------------------------------

@dataclass
class SvgLayer:
    """一个颜色层。"""

    color: str
    paths: list[str] = field(default_factory=list)
    label: str = ""
    gradient: dict | None = None
    opacity: float = 1.0
    stroke: str | None = None
    stroke_width: float = 1.0
    filter_id: str | None = None


@dataclass
class SvgDocument:
    width: int
    height: int
    layers: list[SvgLayer] = field(default_factory=list)
    title: str = "vectorized-art-text"
    defs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # 按字分组：每个字一组 (图层号, 路径号)；None 时按平铺图层输出
    glyph_groups: list[list[tuple[int, int]]] | None = None
    glyph_labels: list[str] | None = None

    def to_svg(self) -> str:
        # 标题/标签来自文件名，必须转义，否则含 & < > " 的文件名会产出非法 XML
        title = _xml_escape(self.title)
        head = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'version="1.1" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">',
            f"  <title>{title}</title>",
            f"  <desc>art-text vectorizer | layers={len(self.layers)}</desc>",
        ]
        if self.defs:
            head.append("  <defs>")
            head.extend(f"    {d}" for d in self.defs)
            head.append("  </defs>")
        elif any(l.filter_id for l in self.layers):
            head.append("  <defs>")
            head.append(f"    {SHADOW_SOFT_FILTER}")
            head.append("  </defs>")

        body: list[str] = []
        if self._glyph_paint_order_is_safe():
            body.extend(self._grouped_body())
        else:
            for i, layer in enumerate(self.layers):
                fill = layer.gradient["ref"] if layer.gradient else layer.color
                label = layer.label or layer.color
                attrs = [f'id="layer-{i + 1}"', f'fill="{fill}"',
                         f'data-name={quoteattr(label)}']
                if layer.stroke:
                    attrs.extend([f'stroke="{layer.stroke}"', f'stroke-width="{layer.stroke_width:.3f}"', 'stroke-linecap="round"'])
                if layer.gradient:
                    attrs.append(f'data-fill-type="gradient"')
                else:
                    attrs.append(f'data-fill-type="solid"')
                if layer.opacity < 1.0:
                    attrs.append(f'opacity="{layer.opacity:.3f}"')
                if layer.filter_id:
                    attrs.append(f'filter="url(#{layer.filter_id})"')
                attrs.append('fill-rule="evenodd"')
                body.append(f"  <g {' '.join(attrs)}>")
                for j, d in enumerate(layer.paths):
                    body.append(f'    <path id="p{i + 1}-{j + 1}" d="{d}"/>')
                body.append("  </g>")

        tail = ["</svg>", ""]
        return "\n".join(head + body + tail)

    def _glyph_paint_order_is_safe(self) -> bool:
        """字级分组须保留全部路径，且不能颠倒相交图形的覆盖关系。"""
        groups = self.glyph_groups
        if not groups:
            return False
        if any(layer.stroke for layer in self.layers) and len(groups)>1:
            return False
        original = [(li, pi) for li, layer in enumerate(self.layers)
                    for pi in range(len(layer.paths))]
        assigned = [pair for group in groups for pair in group]
        if len(assigned) != len(original) or set(assigned) != set(original):
            return False
        # 滤镜范围及组透明度依赖图层整体，保留原始图层分组。
        if any(layer.filter_id or layer.opacity != 1.0 for layer in self.layers):
            return False
        shadow = sorted(pair for pair in assigned
                        if self.layers[pair[0]].label == "shadow")
        reordered = shadow + [pair for group in groups for pair in sorted(group)
                              if self.layers[pair[0]].label != "shadow"]
        if reordered == original:
            return True
        ranks = {pair: rank for rank, pair in enumerate(original)}
        bounds = []
        for li, pi in reordered:
            path = self.layers[li].paths[pi]
            # 绝对三次贝塞尔位于控制点凸包内，可用其外接矩形保守判断。
            commands = re.findall(r"[A-DF-Za-df-z]", path)
            if any(command not in "MLCZ" for command in commands):
                return False
            numbers = _NUM.findall(path)
            if len(numbers) < 2 or len(numbers) % 2:
                return False
            points = np.asarray(numbers, dtype=np.float64).reshape(-1, 2)
            if not np.isfinite(points).all():
                return False
            bounds.append(np.r_[points.min(axis=0), points.max(axis=0)])
        boxes = np.asarray(bounds)
        ordered_ranks = np.array([ranks[pair] for pair in reordered])
        for i, pair in enumerate(reordered):
            inverted = np.flatnonzero(ordered_ranks[:i] > ordered_ranks[i])
            if not len(inverted):
                continue
            box = boxes[i]
            candidates = boxes[inverted]
            overlaps = ((candidates[:, 0] <= box[2]) & (candidates[:, 2] >= box[0]) &
                        (candidates[:, 1] <= box[3]) & (candidates[:, 3] >= box[1]))
            for other_index in inverted[overlaps]:
                layer = self.layers[pair[0]]
                other = self.layers[reordered[other_index][0]]
                same_opaque_color = (layer.gradient is None and other.gradient is None
                                     and layer.opacity == other.opacity == 1.0
                                     and layer.color == other.color
                                     and re.fullmatch(r"#[0-9a-fA-F]{6}", layer.color) is not None)
                if not same_opaque_color:
                    return False
        return True

    def _grouped_body(self) -> list[str]:
        """按字分组输出：外层 <g id="glyph-N"> 每字一个，内含各颜色子层。

        满足 Illustrator 的可编辑分组要求：单个字的全部颜色层包在同一个
        字级分组里，选中字即可整体移动/缩放。阴影层是全局层（柔边投影
        跨字连成整块），统一放最前面的「阴影」组，避免某个字的阴影盖住
        别的字的笔画。
        """
        out: list[str] = []
        shadow_pairs: list[tuple[int, int]] = []
        plain_groups: list[list[tuple[int, int]]] = []
        for pairs in self.glyph_groups or []:
            rest: list[tuple[int, int]] = []
            for li, pi in pairs:
                (shadow_pairs if self.layers[li].label == "shadow" else rest).append((li, pi))
            plain_groups.append(rest)
        if shadow_pairs:
            out.append('  <g id="shadow" data-name="阴影">')
            by_layer_s: dict[int, list[int]] = {}
            for li, pi in shadow_pairs:
                by_layer_s.setdefault(li, []).append(pi)
            for li in sorted(by_layer_s):
                layer = self.layers[li]
                fill = layer.gradient["ref"] if layer.gradient else layer.color
                kind = "gradient" if layer.gradient else "solid"
                attrs = [f'id="shadow-layer-{li + 1}"', f'fill="{fill}"',
                         f'data-name={quoteattr(layer.label or "shadow")}',
                         f'data-fill-type="{kind}"', 'fill-rule="evenodd"']
                if layer.opacity < 1.0:
                    attrs.append(f'opacity="{layer.opacity:.3f}"')
                if layer.filter_id:
                    attrs.append(f'filter="url(#{layer.filter_id})"')
                out.append(f"    <g {' '.join(attrs)}>")
                for pi in sorted(by_layer_s[li]):
                    out.append(f'      <path id="sp{li + 1}-{pi + 1}" d="{layer.paths[pi]}"/>')
                out.append("    </g>")
            out.append("  </g>")
        for gi, pairs in enumerate(plain_groups):
            if not pairs:
                continue
            name = (self.glyph_labels[gi]
                    if self.glyph_labels and gi < len(self.glyph_labels)
                    else f"字{gi + 1}")
            out.append(f'  <g id="glyph-{gi + 1}" data-name={quoteattr(name)}>')
            by_layer: dict[int, list[int]] = {}
            for li, pi in pairs:
                by_layer.setdefault(li, []).append(pi)
            for li in sorted(by_layer):
                layer = self.layers[li]
                fill = layer.gradient["ref"] if layer.gradient else layer.color
                label = layer.label or layer.color
                kind = "gradient" if layer.gradient else "solid"
                attrs = [f'id="g{gi + 1}-layer-{li + 1}"', f'fill="{fill}"',
                         f'data-name={quoteattr(label)}', f'data-fill-type="{kind}"']
                if layer.stroke:
                    attrs.extend([f'stroke="{layer.stroke}"', f'stroke-width="{layer.stroke_width:.3f}"', 'stroke-linecap="round"'])
                if layer.opacity < 1.0:
                    attrs.append(f'opacity="{layer.opacity:.3f}"')
                attrs.append('fill-rule="evenodd"')
                out.append(f"    <g {' '.join(attrs)}>")
                for pi in sorted(by_layer[li]):
                    out.append(f'      <path id="p{gi + 1}-{li + 1}-{pi + 1}" d="{layer.paths[pi]}"/>')
                out.append("    </g>")
            out.append("  </g>")
        return out


def build_linear_gradient(gid: str, stops: Sequence[tuple[float, str]],
                          x1: float, y1: float, x2: float, y2: float) -> str:
    stop_tags = "".join(
        f'<stop offset="{o:.4f}" stop-color="{c}"/>' for o, c in stops
    )
    return (f'<linearGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'x1="{_fmt(x1)}" y1="{_fmt(y1)}" x2="{_fmt(x2)}" y2="{_fmt(y2)}">'
            f"{stop_tags}</linearGradient>")


def build_radial_gradient(gid: str, stops: Sequence[tuple[float, str]],
                          cx: float, cy: float, r: float) -> str:
    stop_tags = "".join(
        f'<stop offset="{o:.4f}" stop-color="{c}"/>' for o, c in stops
    )
    return (f'<radialGradient id="{gid}" gradientUnits="userSpaceOnUse" '
            f'cx="{_fmt(cx)}" cy="{_fmt(cy)}" r="{_fmt(r)}">{stop_tags}'
            f"</radialGradient>")


def rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = (int(max(0, min(255, round(v)))) for v in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def assign_glyph_groups(doc: "SvgDocument", glyphs: int) -> list[list[tuple[int, int]]] | None:
    """把路径按空间位置聚成每字一组，返回 [[(图层号, 路径号)], ...]。

    聚类中心按（行，列）排序，保证分组顺序符合阅读顺序；字数为 1 时
    全部路径归一组。k-means 固定随机种子，同一张图分组结果稳定。
    """
    if not doc.layers or glyphs < 1:
        return None
    from .pathparse import parse_path   # 局部导入避免循环依赖

    entries: list[tuple[int, int, float, float]] = []
    for li, layer in enumerate(doc.layers):
        for pi, d in enumerate(layer.paths):
            xs: list[float] = []
            ys: list[float] = []
            for poly in parse_path(d, resolution=6):
                if len(poly):
                    xs.append(float(poly[:, 0].mean()))
                    ys.append(float(poly[:, 1].mean()))
            if not xs:
                continue
            entries.append((li, pi, float(np.mean(xs)), float(np.mean(ys))))
    if not entries:
        return None
    k = min(int(glyphs), len(entries))
    if k == 1:
        return [[(li, pi) for li, pi, _, _ in entries]]

    data = np.array([[e[2], e[3]] for e in entries], dtype=np.float32)
    scale = max(1.0, float(data.max(axis=0).max() - data.min(axis=0).min()))
    norm = (data - data.min(axis=0)) / scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(20240614)
    _, labels, centers = cv2.kmeans(norm, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    # 阅读顺序：先按行（y，40px 分档）再按列（x）
    order = sorted(range(k), key=lambda i: (round(float(centers[i][1]) * scale / 40.0),
                                            float(centers[i][0])))
    remap = {old: new for new, old in enumerate(order)}
    groups: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    for idx, (li, pi, _, _) in enumerate(entries):
        groups[remap[int(labels[idx][0])]].append((li, pi))
    return [g for g in groups if g] or None


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)
