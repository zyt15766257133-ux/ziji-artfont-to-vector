"""测试样本集生成：23 组不同风格艺术字，覆盖需求 5.4 全部风格。

产出 samples/images/*.png、samples/truth/*.png（真值字形掩膜）与 samples/manifest.json。
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "images"
TRUTH = ROOT / "truth"
FONTS = Path("C:/Windows/Fonts")

SIZE = 1000

FONT_MAP = {
    "hei": "simhei.ttf",
    "sun": "simsun.ttc",
    "kai": "simkai.ttf",
    "xingkai": "STXINGKA.TTF",
    "li": "SIMLI.TTF",
    "you": "SIMYOU.TTF",
    "shuti": "FZSTK.TTF",
    "xihei": "STXIHEI.TTF",
    "song": "STSONG.TTF",
    "kai_hw": "STKAITI.TTF",
}


def font(key: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONTS / FONT_MAP[key]
    return ImageFont.truetype(str(path), size)


def text_size(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=f)
    return box[2] - box[0], box[3] - box[1]


def render_mask(text: str, f: ImageFont.FreeTypeFont, size: int = SIZE,
                stroke: int = 0, dx: float = 0, dy: float = 0) -> np.ndarray:
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    w, h = text_size(draw, text, f)
    pos = ((size - w) / 2 - dx, (size - h) / 2 - dy)
    draw.text(pos, text, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
    return np.asarray(img)


def linear_gradient(size: int, c1: tuple, c2: tuple, angle: float = 45.0) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rad = math.radians(angle)
    proj = xx * math.cos(rad) + yy * math.sin(rad)
    t = (proj - proj.min()) / (proj.max() - proj.min() + 1e-6)
    out = np.zeros((size, size, 3), np.uint8)
    for i in range(3):
        out[:, :, i] = (c1[i] + (c2[i] - c1[i]) * t).astype(np.uint8)
    return out


def radial_gradient(size: int, c1: tuple, c2: tuple,
                    cx: float = 0.5, cy: float = 0.5) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.clip(np.hypot(xx - size * cx, yy - size * cy) / (size * 0.5), 0, 1)[:, :, None]
    out = np.asarray(c1, np.float32) * (1 - r) + np.asarray(c2, np.float32) * r
    return out.astype(np.uint8)


def rainbow_gradient(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    hue = (xx / size * 360 * 0.9 + yy / size * 60) % 360
    h = hue / 60.0
    x = 1 - np.abs(h % 2 - 1)
    z = np.zeros_like(x)
    one = np.ones_like(x)
    cond = h.astype(int) % 6
    r = np.select([cond == 0, cond == 1, cond == 2, cond == 3, cond == 4, cond == 5], [one, x, z, z, x, one])
    g = np.select([cond == 0, cond == 1, cond == 2, cond == 3, cond == 4, cond == 5], [x, one, one, x, z, z])
    b = np.select([cond == 0, cond == 1, cond == 2, cond == 3, cond == 4, cond == 5], [z, z, x, one, one, x])
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)


def composite(mask: np.ndarray, color_rgb: np.ndarray | tuple) -> np.ndarray:
    m = mask.astype(np.float32) / 255.0
    if isinstance(color_rgb, tuple):
        color = np.zeros_like(m)[:, :, None] + np.asarray(color_rgb, np.float32)
    else:
        color = color_rgb.astype(np.float32)
    alpha = m[:, :, None] * 255
    return np.dstack([color, alpha]).astype(np.uint8)


def on_bg(fg: np.ndarray, bg_color: tuple | None = (255, 255, 255)) -> np.ndarray:
    if bg_color is None:
        return fg
    h, w = fg.shape[:2]
    bg = np.zeros((h, w, 4), np.uint8)
    bg[:, :, 0], bg[:, :, 1], bg[:, :, 2] = bg_color
    bg[:, :, 3] = 255
    a = fg[:, :, 3:4].astype(np.float32) / 255.0
    out = bg[:, :, :3].astype(np.float32) * (1 - a) + fg[:, :, :3].astype(np.float32) * a
    res = np.dstack([out, np.full((h, w), 255, np.uint8)]).astype(np.uint8)
    return res


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img)


def feibai(mask: np.ndarray, seed: int, strength: float = 0.55) -> np.ndarray:
    """模拟毛笔飞白：沿笔画方向叠一层毛边噪声。"""
    rng = np.random.default_rng(seed)
    noise = rng.random(mask.shape).astype(np.float32)
    noise = np.asarray(Image.fromarray((noise * 255).astype(np.uint8))
                       .filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255.0
    streaks = (np.sin(np.linspace(0, 40, mask.shape[1]))[None, :] * 0.5 + 0.5)
    local = noise * 0.75 + streaks * 0.25
    keep = (local > (0.42 + (1 - strength) * 0.2)).astype(np.float32)
    keep = np.asarray(Image.fromarray((keep * 255).astype(np.uint8))
                      .filter(ImageFilter.GaussianBlur(0.6))).astype(np.float32) / 255.0
    out = mask.astype(np.float32) * (0.55 + 0.45 * keep)
    return np.clip(out, 0, 255).astype(np.uint8)


def save(arr_rgba: np.ndarray, name: str) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    Image.fromarray(arr_rgba).save(path)
    return str(path)


def save_truth(mask: np.ndarray, name: str) -> str:
    """保存真值字形掩膜（L 模式，255 = 字形前景），供验收对真值评测。"""
    TRUTH.mkdir(parents=True, exist_ok=True)
    path = TRUTH / name
    Image.fromarray(mask.astype(np.uint8)).save(path)
    return str(path)


def text_glyph_count(text: str) -> int:
    """真实字数：忽略空白与多行分隔符。"""
    return max(1, sum(1 for ch in text if not ch.isspace() and ch != "/"))


def auto_truth(arr: np.ndarray, dist_thr: float = 20.0) -> np.ndarray:
    """从成品图自动取真值字形：与可见背景色差 >20 的像素。

    与预处理的前景提取用同一判据，柔边阴影/发光的可见尾巴、
    与背景接近的浅色装饰点（如 13 号白点）都在真值内；
    带透明背景的图改用 alpha 判定。
    """
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    if arr.shape[2] == 4 and float(np.median(border[:, 3])) < 250:
        return (arr[:, :, 3] > 127).astype(np.uint8) * 255
    bg = np.median(border[:, :3], axis=0).astype(np.float64)
    dist = np.linalg.norm(arr[:, :, :3].astype(np.float64) - bg, axis=2)
    return (dist > dist_thr).astype(np.uint8) * 255


def build_samples() -> list[dict]:
    samples: list[dict] = []

    def add(name, arr, style, desc, text, mask=None):
        path = save(arr, name)
        if mask is not None:
            tm = (np.asarray(mask) > 127).astype(np.uint8) * 255
        else:
            tm = auto_truth(arr)
        save_truth(tm, name)
        samples.append({"file": name, "path": path, "style": style,
                        "desc": desc, "text": text,
                        "glyphs": text_glyph_count(text)})

    # 1 单色黑体（无衬线）
    m = render_mask("矢量艺术", font("hei", 190))
    add("01_hei_solid.png", on_bg(composite(m, (24, 24, 26)), (255, 255, 255)),
        "无衬线单色", "黑体四字，纯白底黑字", "矢量艺术")

    # 2 宋体（衬线）
    m = render_mask("宋韵品牌", font("sun", 180))
    add("02_song_serif.png", on_bg(composite(m, (18, 42, 96)), (250, 248, 244)),
        "衬线单色", "宋体衬线，衬线细节完整", "宋韵品牌")

    # 3 楷体
    m = render_mask("品茗听雨", font("kai", 185))
    add("03_kai.png", on_bg(composite(m, (34, 34, 34)), (255, 255, 255)),
        "楷体", "楷体笔画起收笔特征", "品茗听雨")

    # 4 行楷书法
    m = render_mask("墨韵", font("xingkai", 320))
    add("04_xingkai.png", on_bg(composite(m, (12, 12, 12)), (245, 243, 238)),
        "书法行楷", "华文行楷，连笔走势", "墨韵")

    # 5 隶书
    m = render_mask("长安古意", font("li", 175))
    add("05_lishu.png", on_bg(composite(m, (60, 30, 20)), (240, 234, 222)),
        "隶书", "隶书蚕头燕尾", "长安古意")

    # 6 幼圆
    m = render_mask("开心小熊", font("you", 190))
    add("06_youyuan.png", on_bg(composite(m, (255, 122, 0)), (255, 250, 240)),
        "圆体", "幼圆粗笔画圆角", "开心小熊")

    # 7 方正舒体（手写）
    m = render_mask("手写便签", font("shuti", 175))
    add("07_shuti_hand.png", on_bg(composite(m, (30, 60, 120)), (252, 252, 250)),
        "手写体", "方正舒体手写笔画", "手写便签")

    # 8 华文细黑（细笔画）
    m = render_mask("细线轻语", font("xihei", 180))
    add("08_thin_line.png", on_bg(composite(m, (40, 40, 40)), (255, 255, 255)),
        "细笔画", "细笔画对小容差拟合的要求", "细线轻语")

    # 9 描边字
    m = render_mask("描边标题", font("hei", 180), stroke=8)
    inner = render_mask("描边标题", font("hei", 180))
    fg = composite(m, (20, 20, 20))
    fg = np.where(inner[:, :, None] > 100, composite(inner, (255, 214, 0)), fg)
    add("09_stroke.png", on_bg(fg.astype(np.uint8), (255, 255, 255)),
        "描边字", "深色外描边 + 亮黄字身", "描边标题")

    # 10 渐变字
    m = render_mask("渐变光影", font("hei", 185))
    grad = linear_gradient(SIZE, (255, 87, 34), (33, 150, 243), 35)
    add("10_gradient.png", on_bg(composite(m, grad), (255, 255, 255)),
        "渐变字", "橙蓝线性渐变填充", "渐变光影")

    # 11 立体字
    m = render_mask("立体时代", font("hei", 175))
    layers = []
    base = np.zeros((SIZE, SIZE, 4), np.uint8)
    for off in range(14, 0, -2):
        shadow_mask = np.roll(np.roll(m, off, axis=1), off, axis=0)
        c = 40 + off * 6
        side = composite(shadow_mask, (c, c, c))
        a = side[:, :, 3] > 0
        base[a] = side[a]
    top = composite(m, (255, 92, 92))
    a = top[:, :, 3] > 0
    base[a] = top[a]
    add("11_3d.png", on_bg(base, (255, 255, 255)),
        "立体字", "右下立体挤出 + 面色覆盖", "立体时代")

    # 12 阴影字
    m = render_mask("投影层次", font("hei", 180))
    shadow = np.roll(np.roll(blur_mask(m, 6), 12, axis=1), 12, axis=0)
    base = composite(shadow, (150, 150, 150))
    top = composite(m, (16, 16, 16))
    a = top[:, :, 3] > 100
    base[a] = top[a]
    add("12_shadow.png", on_bg(base, (255, 255, 255)),
        "阴影字", "柔边阴影 + 主体", "投影层次")

    # 13 卡通多彩
    base = np.zeros((SIZE, SIZE, 4), np.uint8)
    chars = ["卡", "通", "乐", "园"]
    colors = [(255, 82, 82), (255, 193, 7), (76, 175, 80), (33, 150, 243)]
    f = font("you", 220)
    d = ImageDraw.Draw(Image.new("L", (10, 10)))
    for i, (ch, col) in enumerate(zip(chars, colors)):
        img = Image.new("L", (SIZE, SIZE), 0)
        dd = ImageDraw.Draw(img)
        dd.text((90 + i * 210, 330), ch, font=f, fill=255, stroke_width=7, stroke_fill=255)
        mm = np.asarray(img)
        fg = composite(mm, col)
        a = fg[:, :, 3] > 100
        base[a] = fg[a]
        # 装饰圆点
        dot = Image.new("L", (SIZE, SIZE), 0)
        dr = ImageDraw.Draw(dot)
        dr.ellipse([90 + i * 210 + 150, 260, 90 + i * 210 + 190, 300], fill=255)
        dm = np.asarray(dot)
        dfg = composite(dm, (255, 255, 255))
        a = dfg[:, :, 3] > 100
        base[a] = dfg[a]
    add("13_cartoon.png", on_bg(base, (255, 246, 230)),
        "卡通多彩", "四色卡通字 + 装饰点", "卡通乐园")

    # 14 毛笔飞白
    m = render_mask("書", font("kai", 480))
    m = feibai(m, seed=7)
    add("14_brush_feibai.png", on_bg(composite(m, (10, 10, 10)), (247, 243, 235)),
        "书法毛笔", "带飞白笔触的单字", "書")

    # 15 手写连笔
    m = render_mask("写意人生", font("xingkai", 180))
    add("15_lianbi.png", on_bg(composite(m, (26, 26, 26)), (255, 255, 251)),
        "手写连笔", "连笔走势与笔锋", "写意人生")

    # 16 渐变 + 描边
    m = render_mask("赛博霓虹", font("hei", 175), stroke=6)
    grad = linear_gradient(SIZE, (0, 229, 255), (156, 39, 176), 60)
    add("16_gradient_stroke.png", on_bg(composite(m, grad), (12, 12, 16)),
        "渐变描边", "青紫渐变 + 描边", "赛博霓虹")

    # 17 反白字（深底白字）
    m = render_mask("暗夜模式", font("hei", 180))
    add("17_inverse.png", on_bg(composite(m, (245, 245, 245)), (18, 18, 20)),
        "反白字", "深底亮字，背景扣除测试", "暗夜模式")

    # 18 透明背景 PNG
    m = render_mask("透明底", font("hei", 200))
    add("18_transparent.png", composite(m, (0, 122, 255)),
        "透明背景", "带 alpha 通道的透明底素材", "透明底")

    # 19 多行标题
    base = np.zeros((SIZE, SIZE, 4), np.uint8)
    for i, line in enumerate(["新品发布", "限时登场"]):
        m = render_mask(line, font("hei", 150), dy=(190 if i == 0 else -190))
        fg = composite(m, (200, 30, 60) if i == 0 else (30, 30, 30))
        a = fg[:, :, 3] > 100
        base[a] = fg[a]
    add("19_multiline.png", on_bg(base, (255, 255, 255)),
        "多行标题", "两行标题字，行距紧", "新品发布/限时登场")

    # 20 金属渐变
    m = render_mask("黄金时代", font("li", 180))
    grad = linear_gradient(SIZE, (255, 236, 179), (140, 90, 20), 90)
    band = linear_gradient(SIZE, (255, 255, 255), (190, 140, 40), 90)
    mix = (grad.astype(np.float32) * 0.6 + band.astype(np.float32) * 0.4).astype(np.uint8)
    add("20_metal.png", on_bg(composite(m, mix), (30, 30, 30)),
        "金属渐变", "多段金属渐变", "黄金时代")

    # 21 霓虹发光
    m = render_mask("夜行", font("xingkai", 300))
    glow = blur_mask(m, 18)
    base = composite(glow, (0, 190, 255))
    core = composite(m, (255, 255, 255))
    a = core[:, :, 3] > 100
    base[a] = core[a]
    add("21_neon.png", on_bg(base, (10, 12, 20)),
        "霓虹发光", "发光晕 + 亮芯", "夜行")

    # 23 内孔密集字（日回品国目圆）：验收内孔是否被填实
    m = render_mask("日回品国目", font("hei", 190))
    add("23_holes_dense.png", on_bg(composite(m, (26, 26, 30)), (255, 255, 255)),
        "内孔密集", "验字与回形内孔必须保持镂空", "日回品国目")

    # 24 需求 5.3 标准案例：1000×1000 白底渐变描边字
    m = render_mask("标准案例", font("hei", 210), stroke=10)
    inner = render_mask("标准案例", font("hei", 210))
    grad = linear_gradient(SIZE, (123, 97, 255), (255, 87, 109), 40)
    fg = composite(m, (28, 30, 44))
    gfill = composite(inner, grad)
    a = gfill[:, :, 3] > 100
    fg[a] = gfill[a]
    add("24_standard_53.png", on_bg(fg.astype(np.uint8), (255, 255, 255)),
        "标准渐变描边（5.3）", "1000×1000 白底，深描边 + 紫粉渐变字身", "标准案例")

    # 25 径向渐变：中心亮、外缘深的发光感渐变
    m = render_mask("中心发光", font("hei", 195))
    grad = radial_gradient(SIZE, (255, 241, 118), (255, 87, 34))
    add("25_radial.png", on_bg(composite(m, grad), (255, 255, 255)),
        "径向渐变", "中心亮黄到外缘橙红的径向渐变填充", "中心发光")

    return samples


def main() -> None:
    samples = build_samples()
    manifest = ROOT / "manifest.json"
    manifest.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated {len(samples)} samples -> {OUT}")


if __name__ == "__main__":
    main()
