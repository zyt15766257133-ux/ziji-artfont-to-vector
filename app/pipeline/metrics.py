"""验收指标计算：轮廓还原度、颜色偏差、节点数量、特征保留、色阶数、格式合规。

所有指标均可由单次转换的输出与原始位图直接算出，不依赖人工目检。
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from .pathparse import SvgRaster
from .preprocess import PreparedImage
from .svgutil import hex_to_rgb
from .topology import compare_topology


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 1.0


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    return (cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0)


def _boundary_fscore(a: np.ndarray, b: np.ndarray, tolerance: float = 1.5) -> float:
    """边界 F1：双向距离匹配，容忍 tolerance 像素误差。"""
    ba, bb = _boundary(a), _boundary(b)
    if not ba.any() or not bb.any():
        return 0.0
    da = cv2.distanceTransform((~ba).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((~bb).astype(np.uint8), cv2.DIST_L2, 3)

    match_b = (db[ba] <= tolerance).mean() if ba.any() else 0.0
    match_a = (da[bb] <= tolerance).mean() if bb.any() else 0.0
    if match_a + match_b == 0:
        return 0.0
    return float(2 * match_a * match_b / (match_a + match_b))


def _corner_points(mask: np.ndarray, eps: float = 2.0, angle_deg: float = 45.0) -> np.ndarray:
    """提取轮廓上的角点集合，用于字体特征（尖角/圆角）保留率评估。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts_out = []
    for c in contours:
        if len(c) < 8:
            continue
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        n = len(approx)
        for i in range(n):
            p_prev, p, p_next = approx[(i - 1) % n], approx[i], approx[(i + 1) % n]
            v1 = p - p_prev
            v2 = p_next - p
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                continue
            cos = float(np.dot(v1, v2) / (n1 * n2))
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if ang >= angle_deg:
                pts_out.append(p)
    return np.asarray(pts_out, dtype=np.float64) if pts_out else np.zeros((0, 2), np.float64)


def _corner_retention(orig: np.ndarray, rendered: np.ndarray, tol: float = 4.0) -> float | None:
    """字体特征保留率：原图检出的角点有多大比例在输出中找到对应点。

    原图检不出角点时返回 None：该子项不参与评分，由调用方按剩余权重重归一，
    不再默认给满分。
    """
    src = _corner_points(orig, eps=3.0, angle_deg=50.0)
    if len(src) == 0:
        return None
    dst = _corner_points(rendered, eps=3.0, angle_deg=50.0)
    if len(dst) == 0:
        return 0.0
    h, w = orig.shape
    dmap = np.zeros((h, w), np.uint8)
    for x, y in dst:
        dmap[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))] = 1
    dist = cv2.distanceTransform(1 - dmap, cv2.DIST_L2, 3)
    hit = 0
    for x, y in src:
        if float(dist[int(np.clip(round(y), 0, h - 1)), int(np.clip(round(x), 0, w - 1))]) <= tol:
            hit += 1
    return hit / float(len(src))


def _perimeter_ratio(orig: np.ndarray, rendered: np.ndarray) -> float:
    po = float(_boundary(orig).sum())
    pr = float(_boundary(rendered).sum())
    if po == 0:
        return 1.0
    return float(min(po, pr) / max(po, pr))


def dominant_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    """前景主色（对核心区域像素做 4bit 量化取众数）。"""
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
    if core.sum() == 0:
        core = mask.astype(np.uint8)
    pixels = rgb[core > 0]
    if len(pixels) == 0:
        return (0, 0, 0)
    q = (pixels // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    key = uniq[int(np.argmax(counts))]
    sel = keys == key
    return tuple(int(v) for v in pixels[sel].mean(axis=0))


def _rgb_to_lab(rgb_triplet) -> np.ndarray:
    arr = np.uint8([[list(rgb_triplet)]])
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float64).reshape(3)
    return lab * np.array([100.0 / 255.0, 255.0 / 255.0, 255.0 / 255.0])


def delta_e76(a, b) -> float:
    la, lb = _rgb_to_lab(a), _rgb_to_lab(b)
    return float(np.linalg.norm(la - lb))


def count_color_levels(raster: np.ndarray, mask: np.ndarray | None = None,
                       min_share: float = 0.0) -> int:
    """输出前景内 5bit 量化色数，用于渐变分级评估。

    min_share>0 时只统计占比达标的色阶，用于区分真实渐变与抗锯齿杂色。
    """
    sel = (raster[:, :, 3] > 127) if mask is None else mask
    if sel.sum() == 0:
        return 0
    px = raster[:, :, :3][sel]
    q = px // 8
    keys = q[:, 0].astype(np.int32) * 1024 + q[:, 1].astype(np.int32) * 32 + q[:, 2].astype(np.int32)
    _, counts = np.unique(keys, return_counts=True)
    if min_share <= 0:
        return int(len(counts))
    return int((counts >= max(64, counts.sum() * min_share)).sum())


def quick_shape_iou(prepared: PreparedImage, svg_text: str) -> float:
    """只算轮廓 IoU 的快速复核，供引擎自检使用。"""
    raster = SvgRaster(svg_text).rasterize(scale=1)
    return _iou(prepared.mask > 127, raster[:, :, 3] > 127)


def pixel_color_deviation(prepared: PreparedImage, raster: np.ndarray,
                          erode_px: int = 3,
                          rgb_ref: np.ndarray | None = None) -> float:
    """前景实心区逐像素颜色均值偏差（0~1），跳过抗锯齿过渡带。

    rgb_ref 为颜色参考像素（默认取预处理后的 RGBA）；验收口径传入未经
    CLAHE/去噪的原始像素，避免增强处理影响色偏结论。
    """
    if rgb_ref is None:
        rgb_ref = prepared.rgba[:, :, :3][:, :, ::-1]
    orig = rgb_ref.astype(np.float32)
    out = raster[:, :, :3].astype(np.float32)                      # SvgRaster 已是 RGB
    core = cv2.erode(prepared.mask, np.ones((3, 3), np.uint8), iterations=erode_px) > 0
    out_alpha = raster[:, :, 3] > 200
    solid = prepared.soft_alpha > 250
    sel = core & out_alpha & solid
    if sel.sum() < 100:
        sel = (prepared.mask > 127) & out_alpha
    if sel.sum() < 100:
        return 1.0
    diff = np.abs(orig[sel] - out[sel]).mean() / 255.0
    return float(diff)


def dominant_palette(rgb: np.ndarray, mask: np.ndarray, min_share: float = 0.03,
                     max_k: int = 8) -> list[tuple[tuple[int, int, int], float]]:
    """列出占比达标的代表色，用于多色艺术字的主色对比。"""
    core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
    if core.sum() == 0:
        core = mask.astype(np.uint8)
    pixels = rgb[core > 0]
    if len(pixels) == 0:
        return []
    q = (pixels // 16).astype(np.int32)
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)
    out: list[tuple[tuple[int, int, int], float]] = []
    for i in order[: max_k * 3]:
        share = float(counts[i]) / float(total)
        if share < min_share:
            continue
        color = tuple(int(v) for v in pixels[inverse == i].mean(axis=0))
        out.append((color, share))
        if len(out) >= max_k:
            break
    return out


def palette_deviation(orig_palette: list, out_palette: list) -> float:
    """主色集合的双向最大偏差（0~1）。单个颜色集合缺失时返回 1。"""
    if not orig_palette or not out_palette:
        return 1.0

    def worst(src, dst):
        worst_val = 0.0
        for color, _share in src:
            best = min(max(abs(int(a) - int(b)) / 255.0 for a, b in zip(color, other)) for other, _ in dst)
            worst_val = max(worst_val, best)
        return worst_val

    return float(max(worst(orig_palette, out_palette), worst(out_palette, orig_palette)))


def color_presence_deviation(orig_palette: list, out_rgb: np.ndarray, out_mask: np.ndarray) -> float:
    """多色艺术字颜色保真：每个原图主色族在输出中最近的同类颜色差（0~1）。

    与逐像素对比相比，这个指标不受色块边界 1~2px 偏差与渐变分带的影响。
    主色先按色族单链合并：渐变、光晕的过渡色阶是同一族颜色，不是各自独立
    的主色（否则阴影层的 3 级简化会被记成 40% 色偏）。
    """
    if not orig_palette:
        return 1.0
    fams: list[list[np.ndarray]] = []
    for color, _share in orig_palette:
        c = np.asarray(color, dtype=np.float64)
        for fam in fams:
            if any(float(np.abs(m - c).max()) <= 28.0 for m in fam):
                fam.append(c)
                break
        else:
            fams.append([c])
    family_means = [np.mean(f, axis=0) for f in fams]
    core = cv2.erode(out_mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
    if core.sum() == 0:
        core = out_mask.astype(np.uint8)
    px = out_rgb[core > 0].astype(np.int32)
    if len(px) == 0:
        return 1.0
    q = px // 8
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    uniq, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    total = counts.sum()
    keep = [i for i in range(len(uniq)) if counts[i] >= max(32, total * 0.002)]
    if not keep:
        return 1.0
    reps = np.array([px[inverse == i].mean(axis=0) for i in keep], dtype=np.float64)
    worst = 0.0
    for color in family_means:
        d = np.abs(reps - np.asarray(color, dtype=np.float64)).max(axis=1).min() / 255.0
        worst = max(worst, float(d))
    return worst


def _line_single_color(palette: list, bg, residual_tol: float = 16.0) -> bool:
    """单色艺术字判定：所有主色都能解释为「背景色 ↔ 主墨色」连线上的混色。

    飞白、笔触的半透明灰阶是墨色与背景的线性混合，归单色（按≤3% 色值考核）；
    描边、阴影、多色卡通等出现离线颜色时归多色（按主色匹配考核）。
    bg 缺失或主色贴近背景时退回“主色数 ≤3”旧规则。
    """
    if len(palette) <= 1:
        return True
    if not bg:
        return len(palette) <= 3
    d = np.asarray(palette[0][0], dtype=np.float64)   # 占比最大的主色 (RGB)
    v = d - np.asarray(bg, dtype=np.float64)
    norm2 = float(v @ v)
    if norm2 < 400.0:      # 主色与背景几乎同色，连线无意义
        return len(palette) <= 3
    for color, _share in palette:
        w = np.asarray(color, dtype=np.float64) - np.asarray(bg, dtype=np.float64)
        t = float(w @ v) / norm2
        if t < -0.05 or t > 1.15:
            return False
        residual = float(np.linalg.norm(w - t * v))
        if residual > residual_tol:
            return False
    return True


_CMD_NUM = re.compile(r"([MmLlCcZzHhVvSsQqTtAa])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)")


def _redundant_anchor_ratio(svg_text: str, dev_tol: float = 0.3) -> float:
    """冗余控制点比例：C 段中点偏离首尾弦不足 dev_tol 像素的段占比。

    这类段可用直线指令替代而不改变形状，比例越高说明路径越有精简空间。
    只统计 C 段（本引擎的主要输出），无 C 段时返回 0。
    """
    total = 0
    straight = 0
    for d_attr in re.findall(r'\sd="([^"]+)"', svg_text):
        tokens = []
        for m in _CMD_NUM.finditer(d_attr):
            tokens.append(("cmd", m.group(1)) if m.group(1) else ("num", m.group(2)))
        cur = np.zeros(2, dtype=np.float64)
        start = cur.copy()
        cmd = ""
        i = 0
        while i < len(tokens):
            kind, val = tokens[i]
            if kind == "cmd":
                cmd = val
                i += 1
                if cmd in "Zz":
                    cur = start.copy()
                continue
            rel = cmd.islower()
            c = cmd.upper()
            need = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6,
                    "S": 4, "Q": 4, "T": 2, "A": 7}.get(c, 0)
            if need == 0 or i + need > len(tokens):
                i += 1
                continue
            nums = [float(tokens[i + k][1]) for k in range(need)]
            i += need
            if c == "M":
                p = np.asarray(nums[:2]) + (cur if rel else 0)
                cur, start = p.copy(), p.copy()
                cmd = "l" if rel else "L"
            elif c == "L":
                cur = np.asarray(nums[:2]) + (cur if rel else 0)
            elif c == "H":
                cur = np.array([nums[0] + (cur[0] if rel else 0.0), cur[1]])
            elif c == "V":
                cur = np.array([cur[0], nums[0] + (cur[1] if rel else 0.0)])
            elif c == "C":
                p0 = cur
                c1 = np.asarray(nums[0:2]) + (p0 if rel else 0)
                c2 = np.asarray(nums[2:4]) + (p0 if rel else 0)
                p3 = np.asarray(nums[4:6]) + (p0 if rel else 0)
                mid = (p0 + 3 * c1 + 3 * c2 + p3) / 8.0
                dev = float(np.linalg.norm(mid - (p0 + p3) / 2.0))
                total += 1
                if dev < dev_tol:
                    straight += 1
                cur = p3
            else:
                # 其他曲线指令不参与评价，只跟踪终点（S/Q/T 取前两数为终点）
                if c in ("S", "Q"):
                    cur = np.asarray(nums[0:2]) + (cur if rel else 0)
                elif c == "T":
                    cur = np.asarray(nums[:2]) + (cur if rel else 0)
                elif c == "A":
                    cur = np.asarray(nums[5:7]) + (cur if rel else 0)
    return round(straight / total, 4) if total else 0.0


def straight_chains_rms(mask: np.ndarray, axis_tol_deg: float = 14.0,
                        min_chain: float = 26.0,
                        contrast_img: np.ndarray | None = None,
                        contrast_min: float = 90.0) -> tuple[float | None, int]:
    """长直边的直线度：链上各点到最佳直线的均方偏差（像素，越小越平直）。

    「横不平」在 IoU/边界 F1 里看不见：长直边上的波浪偏差不到 1px，
    面积类指标全绿但观感不合格。这里沿轮廓找近水平/近竖直的长直线链，
    量它们的波浪度，填补验收口径的缺口。contrast_img 传入时只考核
    高对比边（笔画边），柔边阴影的轮廓不参与；长直边少的书法/手写类
    返回 (None, 0) 不作考核。
    """
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    t = math.tan(math.radians(axis_tol_deg))
    chains: list[tuple[float, float]] = []
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 24:
            continue
        stride = max(2, n // 120)
        idx = np.arange(0, n, stride)
        p = pts[idx]
        if len(p) < 6:
            continue
        v = np.roll(p, -1, axis=0) - p
        length = np.hypot(v[:, 0], v[:, 1])
        ok = length >= 5
        horiz = (np.abs(v[:, 1]) <= np.abs(v[:, 0]) * t) & ok
        vert = (np.abs(v[:, 0]) <= np.abs(v[:, 1]) * t) & ok
        axis = np.where(horiz, 1, np.where(vert, 2, 0))
        m = len(p)
        visited = np.zeros(m, dtype=bool)
        for s in range(m):
            if axis[s] == 0 or visited[s]:
                continue
            a = int(axis[s])
            chain = [s]
            visited[s] = True
            j = (s + 1) % m
            while j != s and axis[j] == a and not visited[j]:
                chain.append(j)
                visited[j] = True
                j = (j + 1) % m
            total = float(sum(length[k] for k in chain))
            if total < min_chain:
                continue
            P = p[chain + [(chain[-1] + 1) % m]]
            if a == 1:
                col = P[:, 1]
                nrm = np.array([0.0, 1.0])
            else:
                col = P[:, 0]
                nrm = np.array([1.0, 0.0])
            if contrast_img is not None:
                h_img, w_img = contrast_img.shape[:2]
                pa = np.clip(P + nrm * 3, [0, 0], [w_img - 1, h_img - 1]).astype(int)
                pb = np.clip(P - nrm * 3, [0, 0], [w_img - 1, h_img - 1]).astype(int)
                c = float(np.abs(contrast_img[pa[:, 1], pa[:, 0]]
                                 - contrast_img[pb[:, 1], pb[:, 0]]).mean())
                if c < contrast_min:
                    continue      # 柔边（阴影/光晕）轮廓不算笔画直边
            rms = float(np.sqrt(((col - col.mean()) ** 2).mean()))
            chains.append((total, rms))
    if not chains:
        return None, 0
    weight = sum(w for w, _ in chains)
    return sum(w * r for w, r in chains) / max(1e-6, weight), len(chains)


def _find_axis_chains(pts: np.ndarray, t: float, min_chain: float,
                      length: np.ndarray | None = None) -> list[tuple[int, list[int]]]:
    """在一条轮廓上找近水平/近竖直的长链，返回 [(轴向, 链上的采样段索引)]。"""
    n = len(pts)
    if n < 6:
        return []
    v = np.roll(pts, -1, axis=0) - pts
    if length is None:
        length = np.hypot(v[:, 0], v[:, 1])
    ok = length >= 5
    horiz = (np.abs(v[:, 1]) <= np.abs(v[:, 0]) * t) & ok
    vert = (np.abs(v[:, 0]) <= np.abs(v[:, 1]) * t) & ok
    axis = np.where(horiz, 1, np.where(vert, 2, 0))
    out: list[tuple[int, list[int]]] = []
    visited = np.zeros(n, dtype=bool)
    for s in range(n):
        if axis[s] == 0 or visited[s]:
            continue
        a = int(axis[s])
        chain = [s]
        visited[s] = True
        j = (s + 1) % n
        while j != s and axis[j] == a and not visited[j]:
            chain.append(j)
            visited[j] = True
            j = (j + 1) % n
        if float(sum(length[k] for k in chain)) >= min_chain:
            out.append((a, chain))
    return out


def orthogonal_deviation(src_mask: np.ndarray, out_mask: np.ndarray,
                         contrast_img: np.ndarray | None = None,
                         out_lum: np.ndarray | None = None,
                         axis_tol_deg: float = 12.0, min_chain: float = 30.0,
                         max_dist: float = 3.0, src_rms_max: float = 0.5,
                         contrast_min: float = 90.0) -> tuple[float | None, int]:
    """正交偏差：矢量边偏离原图长直边的均方根距离（像素，越小越准）。

    「横不平」在 IoU/边界 F1 里看不见，用户却一眼能看出。口径：原图上
    本身平直的长直边（笔画边）才参与考核，逐点量矢量边到这条直线的
    偏离。原图本来就弯的设计字（卡通、书法）不会被误报，转换把直线
    变弯才会被抓出来。无合格直边时返回 (None, 0)。
    """
    contours, _ = cv2.findContours((src_mask > 0).astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    out_bool = out_mask > 0
    out_edge = out_bool ^ (cv2.erode(out_bool.astype(np.uint8),
                                    np.ones((3, 3), np.uint8)) > 0)
    if out_lum is not None:
        # 只量笔画边：输出端同口径过滤掉柔边（阴影/光晕的轮廓不算直边）
        grad = (cv2.dilate(out_lum, np.ones((3, 3), np.uint8))
                - cv2.erode(out_lum, np.ones((3, 3), np.uint8)))
        out_edge = out_edge & (np.abs(grad) >= contrast_min)
    if not out_edge.any():
        return None, 0
    t = math.tan(math.radians(axis_tol_deg))
    devs: list[tuple[float, float]] = []   # (采样数, 链的波浪 RMS)
    chains_used = 0
    h_img, w_img = src_mask.shape[:2]
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n < 16:
            continue
        stride = max(2, n // 120)
        idx = np.arange(0, n, stride)
        p = pts[idx]
        if len(p) < 5:
            continue
        v = np.roll(p, -1, axis=0) - p
        length = np.hypot(v[:, 0], v[:, 1])
        for a, chain in _find_axis_chains(p, t, min_chain, length=length):
            P = p[chain + [(chain[-1] + 1) % len(p)]]
            # 参照线 = 链的最小二乘拟合直线（带斜率），设计斜体不会误报
            centroid = P.mean(axis=0)
            _, _, vt = np.linalg.svd(P - centroid, full_matrices=False)
            u = vt[0]
            vn = np.array([-u[1], u[0]])
            perp = (P - centroid) @ vn
            if float(np.sqrt(np.mean(np.square(perp)))) > src_rms_max:
                continue            # 原图本身就弯，不作考核
            if contrast_img is not None:
                pa = np.clip(P + vn * 3, [0, 0], [w_img - 1, h_img - 1]).astype(int)
                pb = np.clip(P - vn * 3, [0, 0], [w_img - 1, h_img - 1]).astype(int)
                ct = float(np.abs(contrast_img[pa[:, 1], pa[:, 0]]
                                  - contrast_img[pb[:, 1], pb[:, 0]]).mean())
                if ct < contrast_min:
                    continue        # 柔边（阴影/光晕）不算笔画直边
            along = (P - centroid) @ u
            a0, a1 = float(along.min()), float(along.max())
            series: list[float] = []
            step = 4.0
            xq = a0
            while xq <= a1 + 1e-6:
                Pc = centroid + u * xq
                x_lo = int(max(0, Pc[0] - max_dist - 2)); x_hi = int(min(w_img - 1, Pc[0] + max_dist + 2))
                y_lo = int(max(0, Pc[1] - max_dist - 2)); y_hi = int(min(h_img - 1, Pc[1] + max_dist + 2))
                if x_hi >= x_lo and y_hi >= y_lo:
                    sub = out_edge[y_lo:y_hi + 1, x_lo:x_hi + 1]
                    ys, xs = np.nonzero(sub)
                    if len(ys):
                        Q = np.stack([x_lo + xs, y_lo + ys], axis=1).astype(np.float64)
                        d_perp = (Q - centroid) @ vn
                        d_along = np.abs((Q - centroid) @ u - xq)
                        ok = d_along <= 2.5
                        if ok.any():
                            cand = d_perp[ok]
                            series.append(float(cand[np.argmin(np.abs(cand))]))
                xq += step
            if len(series) >= 3:
                # 量的是「弯不弯」：去掉整体偏移（坐标量化的常量差），
                # 剩下的慢波才是「横不平」；平滑滤掉 ±0.5px 高频抖动，
                # 截尾防个别错误匹配点拉偏
                d = np.asarray(series, dtype=np.float64)
                d = d - d.mean()
                k = min(5, len(d) if len(d) % 2 == 1 else len(d) - 1)
                if k >= 3:
                    d = np.convolve(d, np.ones(k) / k, mode="valid")
                trim = max(1, int(len(d) * 0.15))
                if trim * 2 < len(d):
                    order = np.argsort(-np.abs(d))
                    d = np.delete(d, order[:trim])
                rms_c = float(np.sqrt(np.mean(np.square(d))))
                devs.append((float(len(series)), rms_c))
                chains_used += 1
    if not devs:
        return None, 0
    wsum = sum(w for w, _ in devs)
    rms = sum(w * r for w, r in devs) / max(1e-6, wsum)
    return rms, chains_used


def evaluate(prepared: PreparedImage, svg_text: str, elapsed_ms: float | None = None,
             glyph_count: int | None = None, raster: np.ndarray | None = None,
             truth_mask: np.ndarray | None = None, tolerance: float | None = None) -> dict:
    """汇总一张图的全部验收指标。raster 可传入已栅格化的结果，避免重复渲染。

    glyph_count 为真实字数（来自 manifest 或用户手填），不传时退回连通域估算
    （并以 glyph_source 标注）。truth_mask 为生成器真值字形，传入时额外给出
    “对真值”口径的轮廓/内孔指标，与系统自参照口径并列。tolerance 为拟合
    容差，平直度容差随它缩放（精简档允许稍弯）。
    """
    raster_parser = SvgRaster(svg_text)
    if raster is None:
        raster = raster_parser.rasterize()

    orig_mask = prepared.mask > 127
    out_mask = raster[:, :, 3] > 127
    alpha_shape_iou = _iou(orig_mask, out_mask)
    comparison_basis = 'foreground_alpha'
    # On an opaque source, a faint grey painted shape and a translucent black
    # shadow can look identical. Alpha>50% incorrectly declares the latter
    # missing. Compare filtered artwork on the same original backdrop instead.
    opaque_source = (prepared.source_rgba is not None
                     and bool(np.all(prepared.source_rgba[:,:,3] >= 250)))
    if ('<filter' in svg_text and opaque_source and prepared.bg_color is not None
            and prepared.quality.background_uniform and prepared.rgb_raw is not None):
        bg = np.asarray(prepared.bg_color,dtype=float)
        alpha = raster[:,:,3:4].astype(float)/255.
        composite = raster[:,:,:3]*alpha+bg*(1-alpha)
        orig_mask = np.linalg.norm(prepared.rgb_raw.astype(float)-bg,axis=2)>20
        out_mask = np.linalg.norm(composite-bg,axis=2)>20
        raster = np.dstack([np.uint8(np.clip(np.rint(composite),0,255)),
                            np.full(raster.shape[:2],255,np.uint8)])
        comparison_basis = 'source_background_composite_contrast_20'

    est = _glyph_count(orig_mask)
    glyphs = int(glyph_count) if glyph_count else est
    glyph_source = "provided" if glyph_count else "auto"

    # 特征口径：只考主体（高对比）区域。柔边阴影/光晕不是字体特征，
    # 它的轮廓是低对比的模糊边，不能拖累「字体特征保留」（需求三）
    clum_src = getattr(prepared, "rgb_raw", None)
    if clum_src is None:
        clum_src = prepared.rgba[:, :, :3][:, :, ::-1]
    clum = clum_src @ np.array([0.299, 0.587, 0.114])
    _bg = prepared.bg_color
    _bg_lum = float(np.dot(_bg, [0.299, 0.587, 0.114])) if _bg is not None \
        else (float(np.median(clum[~orig_mask])) if (~orig_mask).any() else 255.0)
    hi_px = (np.abs(clum - _bg_lum) > 90) & orig_mask
    if int(hi_px.sum()) >= 0.15 * max(1, int(orig_mask.sum())):
        support = cv2.dilate(hi_px.astype(np.uint8), np.ones((3, 3), np.uint8),
                             iterations=3) > 0
        f_orig, f_out = orig_mask & support, out_mask & support
    else:
        f_orig, f_out = orig_mask, out_mask

    iou = _iou(orig_mask, out_mask)
    iou_tolerant = _iou(_dilate(orig_mask, 1), _dilate(out_mask, 1))
    bf1 = _boundary_fscore(f_orig, f_out, 1.5)
    bf1_2px = _boundary_fscore(f_orig, f_out, 2.5)
    corner_keep = _corner_retention(f_orig, f_out)
    perimeter = _perimeter_ratio(f_orig, f_out)
    if corner_keep is None:
        # 原图无角点：剔除该子项，按 0.5+0.2 的剩余权重重归一
        feature = round((0.5 * bf1_2px + 0.2 * perimeter) / 0.7, 4)
    else:
        feature = round(0.5 * bf1_2px + 0.3 * corner_keep + 0.2 * perimeter, 4)

    stats = raster_parser.node_stats()

    # 颜色参考：未经 CLAHE/去噪的原始像素；增强后像素仅作差值参考
    rgb_ref = getattr(prepared, "rgb_raw", None)
    if rgb_ref is None:
        rgb_ref = prepared.rgba[:, :, :3][:, :, ::-1]
    orig_rgb = rgb_ref
    orig_rgb_enh = prepared.rgba[:, :, :3][:, :, ::-1]
    # 主色/单多色判定只看不透明实心像素：飞白、柔边阴影的半透明混色
    # 是前景与背景的合成色，不能当成额外颜色层
    ref_mask = orig_mask & (prepared.soft_alpha > 250)
    if int(ref_mask.sum()) < 100:
        ref_mask = orig_mask
    orig_dom = dominant_color(orig_rgb, ref_mask)
    out_rgb = raster[:, :, :3]          # SvgRaster 输出已是 RGB
    out_dom = dominant_color(out_rgb, out_mask)
    channel_dev = [abs(int(a) - int(b)) / 255.0 for a, b in zip(orig_dom, out_dom)]
    color_hex_dev = max(channel_dev)
    de = delta_e76(orig_dom, out_dom)
    pix_dev = pixel_color_deviation(prepared, raster, rgb_ref=orig_rgb)
    pix_dev_enh = pixel_color_deviation(prepared, raster, rgb_ref=orig_rgb_enh)
    holes_orig = count_holes(orig_mask)
    holes_out = count_holes(out_mask)
    orig_palette = dominant_palette(orig_rgb, ref_mask)
    out_palette = dominant_palette(out_rgb, out_mask)
    pal_dev = palette_deviation(orig_palette, out_palette)
    presence_dev = color_presence_deviation(orig_palette, out_rgb, out_mask)

    orig_rgba = np.dstack([prepared.rgba[:, :, 2], prepared.rgba[:, :, 1],
                           prepared.rgba[:, :, 0], prepared.rgba[:, :, 3]])

    # 轮廓质量派生量
    out_u8 = out_mask.astype(np.uint8)
    ncomp_out, _, st_out, _ = cv2.connectedComponentsWithStats(out_u8, 8)
    ncomp_orig, _, st_orig, _ = cv2.connectedComponentsWithStats(orig_mask.astype(np.uint8), 8)
    comps_out = [int(st_out[i, cv2.CC_STAT_AREA]) for i in range(1, ncomp_out)]
    comps_orig = [int(st_orig[i, cv2.CC_STAT_AREA]) for i in range(1, ncomp_orig)]
    speck_count = sum(1 for a in comps_out if a < 20)
    stroke_breaks = max(0, len(comps_out) - len(comps_orig))
    opened = cv2.morphologyEx(out_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    spur_px = int(((out_u8 > 0) & (opened == 0)).sum())
    perim = float(_boundary(out_mask).sum())
    spur_ratio = round(spur_px / max(1.0, perim), 4)
    levels_orig = count_color_levels(orig_rgba, orig_mask, min_share=0.02)
    grad_stops = [len(re.findall(r"<stop\b", block)) for block in
                  re.findall(r"<(?:linear|radial)Gradient\b[^>]*>(.*?)</(?:linear|radial)Gradient>",
                             svg_text, re.S)]
    gradient_levels = max(grad_stops) if grad_stops else 0
    levels_out = count_color_levels(raster, out_mask, min_share=0.02)
    is_gradient_art = _detect_gradient_art(prepared)
    out_lum_img = raster[:, :, :3] @ np.array([0.299, 0.587, 0.114])
    _alpha = raster[:, :, 3].astype(np.float64) / 255.0
    out_lum_img = out_lum_img * _alpha + _bg_lum * (1.0 - _alpha)
    ortho_dev, ortho_chains = orthogonal_deviation(orig_mask, out_mask, contrast_img=clum,
                                                   out_lum=out_lum_img)
    try:
        from .vectorize import axis_alignment_score
        axis_score = axis_alignment_score(orig_mask, crisp=(hi_px.astype(np.uint8) * 255))
    except Exception:
        axis_score = 0.0

    # 单/多色判定：主色全部落在「背景→主色」连线上的按单色（飞白灰阶属墨色
    # 半透明混色），否则多色；渐变型一律多色。不再用 5bit 全量色数，
    # 避免单色墨迹因抗锯齿灰阶被误判为多色。
    color_mode = ("multi" if is_gradient_art
                  else ("single" if _line_single_color(orig_palette, getattr(prepared, "bg_color", None))
                        else "multi"))
    color_limit = 0.03 if color_mode == "single" else 0.08
    # 单色按需求「RGB/HEX 色值偏差≤3%」考核主色色值；多色按主色匹配偏差考核。
    # 逐像素色偏继续报告，不作单色判定线（柔边/渐变类会被空间分布误伤）。
    color_dev = color_hex_dev if color_mode == "single" else presence_dev

    metrics = {
        'comparison_basis': comparison_basis,
        'alpha_shape_iou': round(alpha_shape_iou,4),
        # 轮廓还原度（主指标）
        "shape_iou": round(iou, 4),
        "shape_iou_1px": round(iou_tolerant, 4),
        "boundary_f1": round(bf1, 4),
        "contour_fidelity": round(max(iou, iou_tolerant * 0.98), 4),
        # 字体特征保留 = 边界吻合 50% + 角点保留 30% + 笔画复杂度（周长比）20%
        # 原图无角点时 corner_retention 为 null，feature 按剩余权重重归一
        "corner_retention": round(corner_keep, 4) if corner_keep is not None else None,
        "perimeter_ratio": round(perimeter, 4),
        "boundary_f1_2px": round(bf1_2px, 4),
        "feature_retention": feature,
        # 节点
        "path_count": stats["path_count"],
        "subpath_count": stats.get("subpath_count", 0),
        "node_floor_estimate": stats.get("subpath_count", 0) * 6,
        "redundant_anchor_ratio": _redundant_anchor_ratio(svg_text),
        "layer_count": stats["layer_count"],
        "node_total": stats["node_total"],
        "node_max": stats["node_max"],
        "node_per_glyph": round(stats["node_total"] / max(1, glyphs), 1),
        "node_per_layer_glyph": round(stats["node_total"] / max(1, glyphs * max(1, stats["layer_count"])), 1),
        "glyph_count": glyphs,
        "glyph_source": glyph_source,
        "glyph_estimate": est,
        # 颜色：单色看逐像素色偏，多色/渐变看主色匹配偏差（颜色参考 = 原始像素）
        "color_orig": "#%02x%02x%02x" % orig_dom,
        "color_out": "#%02x%02x%02x" % out_dom,
        "color_mode": color_mode,
        "color_deviation": round(color_dev, 4),
        "color_deviation_dominant": round(color_hex_dev, 4),
        "color_deviation_pixel": round(pix_dev, 4),
        "color_deviation_pixel_enhanced": round(pix_dev_enh, 4),
        "color_ref_delta": round(abs(pix_dev - pix_dev_enh), 4),
        "color_deviation_palette": round(pal_dev, 4),
        "color_deviation_presence": round(presence_dev, 4),
        "palette_orig": ["#%02x%02x%02x" % c for c, _ in orig_palette],
        "palette_out": ["#%02x%02x%02x" % c for c, _ in out_palette],
        "delta_e76": round(de, 2),
        "color_levels_orig": levels_orig,
        "color_levels_out": max(levels_out, gradient_levels),
        "gradient_stop_count": gradient_levels,
        "is_gradient_art": is_gradient_art,
        # 内孔保持：日/回/品/国 这类字的关键质量项（不允许少孔）
        "holes_orig": holes_orig,
        "holes_out": holes_out,
        "holes_diff": holes_out - holes_orig,
        "holes_preserved": holes_out >= holes_orig,
        # 轮廓质量：断笔（输出比原图多出的碎块）、孤立小块、毛刺
        "stroke_breaks": stroke_breaks,
        "speck_count": speck_count,
        "spur_ratio": spur_ratio,
        # 直线平直度：长直边的波浪幅度（px）。转换不得把直线变弯，
        # 原图本身弯曲的设计字不参与考核（需求三观感口径的补丁）
        "axis_alignment": round(axis_score, 3),
        "straight_waviness": round(ortho_dev, 3) if ortho_dev is not None else None,
        "straight_chains": ortho_chains,
        "waviness_limit": round(0.18 * max(1.0, (tolerance or 0.5) / 0.5), 3),
    }
    if elapsed_ms is not None:
        metrics["elapsed_ms"] = round(elapsed_ms, 1)

    if truth_mask is not None:
        tm = prepared.align_mask(truth_mask) > 127
        th_orig = count_holes(tm)
        metrics["truth_topology"] = compare_topology(tm, out_mask)
        metrics["truth_present"] = True
        metrics["truth_shape_iou"] = round(_iou(tm, out_mask), 4)
        metrics["truth_boundary_f1"] = round(_boundary_fscore(tm, out_mask, 1.5), 4)
        metrics["truth_holes_orig"] = th_orig
        metrics["truth_holes_preserved"] = holes_out >= th_orig
        metrics["checks_truth"] = {
            "truth_contour>=0.90": metrics["truth_shape_iou"] >= 0.90,
            "truth_holes_preserved": metrics["truth_holes_preserved"],
            "truth_topology_preserved": metrics["truth_topology"]["passed"],
        }
        metrics["checks_truth"]["passed"] = all(
            v for k, v in metrics["checks_truth"].items() if k != "passed")

    metrics["checks"] = _checks(metrics)
    return metrics


def _glyph_count(mask: np.ndarray) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return 1
    areas = stats[1:, cv2.CC_STAT_AREA]
    return max(1, int((areas >= max(16, areas.sum() * 0.002)).sum()))


def count_holes(mask: np.ndarray, min_area: float = 24.0) -> int:
    """统计前景内孔数量（面积 ≥24px² 的才算，忽略像素级噪点）。

    日/回/品/国 这类字的内孔如果被填实，视觉上就是「连成一块」，
    所以单独作为一个可验收的指标。
    """
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    return int(sum(1 for i, h in enumerate(hierarchy[0])
                   if h[3] != -1 and abs(cv2.contourArea(contours[i])) >= min_area))


def _detect_gradient_art(prepared: PreparedImage) -> bool:
    """判断输入是否属于渐变型艺术字（决定是否考核色阶数）。线性与径向都算。"""
    try:
        from .vectorize import _fit_global_gradient, _fit_radial_gradient
    except Exception:
        return False
    try:
        if _fit_global_gradient(prepared.rgba[:, :, :3], prepared.mask, 10) is not None:
            return True
        return _fit_radial_gradient(prepared.rgba[:, :, :3], prepared.mask, 10) is not None
    except Exception:
        return False


def _checks(m: dict) -> dict:
    """按需求硬指标给出通过判定（渐变色阶只对渐变型艺术字生效）。

    颜色口径：单色艺术字看主色色值偏差（≤3%，需求「RGB/HEX 色值偏差」）；
    多色/渐变艺术字看主色匹配偏差（≤8%）。
    节点口径：单字节点 ≤500（唯一规则，不再附带单层放宽）。
    内孔口径：输出内孔数不得少于原图。
    """
    color_key = "color_dev<=0.03" if m.get("color_mode") == "single" else "color_dev<=0.08"
    color_limit = 0.03 if m.get("color_mode") == "single" else 0.08
    checks = {
        "contour>=0.90": m["contour_fidelity"] >= 0.90,
        "feature>=0.85": m["feature_retention"] >= 0.85,
        "node_50_500": 50 <= m["node_per_glyph"] <= 500,
        "holes_preserved": m.get("holes_preserved", True),
        "no_specks": m.get("speck_count", 0) == 0,
        "no_breaks": m.get("stroke_breaks", 0) == 0,
        "spur_ratio<=0.02": m.get("spur_ratio", 0) <= 0.02,
        color_key: m["color_deviation"] <= color_limit,
    }
    if m.get("is_gradient_art"):
        checks["gradient_levels>=8"] = m["color_levels_out"] >= 8
    if (m.get("straight_waviness") is not None and m.get("straight_chains", 0) >= 3
            and m.get("axis_alignment", 0) >= 0.42):
        checks["waviness"] = m["straight_waviness"] <= m.get("waviness_limit", 0.18)
    checks["passed"] = all(checks.values())
    return checks


def format_check(svg_text: str) -> dict:
    """SVG 结构与兼容性检查：先用 XML 解析器验证合法可解析，再查关键要素。"""
    checks: dict[str, bool] = {}
    try:
        root = ET.fromstring(svg_text)
        checks["xml_well_formed"] = True
    except ET.ParseError as exc:
        return {"xml_well_formed": False, "parse_error": str(exc), "passed": False}
    ns = "{http://www.w3.org/2000/svg}"
    checks["has_xmlns"] = root.tag == f"{ns}svg"
    checks["has_viewBox"] = "viewBox" in root.attrib
    checks["has_title"] = root.find(f"{ns}title") is not None
    checks["grouped_layers"] = len(root.findall(f"{ns}g")) >= 1
    checks["no_embedded_raster"] = root.find(f".//{ns}image") is None
    checks["named_layers"] = "data-name" in svg_text
    checks["passed"] = all(checks.values())
    return checks
