"""User-supplied HEX palettes; shared by the curve and layered pipelines."""
from __future__ import annotations

import re


def normalize_hex(value: str) -> str | None:
    value = str(value).strip().lower()
    if re.fullmatch(r'#[0-9a-f]{6}', value):
        return value
    if re.fullmatch(r'#[0-9a-f]{3}', value):
        return '#' + ''.join(ch * 2 for ch in value[1:])
    return None


def normalize_palette(colors: list[str] | None) -> list[str]:
    if not colors:
        raise ValueError('启用专色匹配后请填写至少一个 HEX 色值')
    palette = []
    for value in colors:
        color = normalize_hex(value)
        if color is None:
            raise ValueError('专色色值格式错误，请使用 #RGB 或 #RRGGBB，多个颜色用英文逗号分隔')
        if color not in palette:
            palette.append(color)
    return palette


def nearest_color(color: str, palette: list[str]) -> str:
    def rgb(value):
        return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
    source = rgb(color)
    return min(palette, key=lambda c: sum((a - b) ** 2 for a, b in zip(source, rgb(c))))
