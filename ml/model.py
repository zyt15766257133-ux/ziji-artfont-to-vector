"""小型 U-Net，学习字形覆盖率；输出仍须由几何引擎拟合为 SVG。"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def features(rgb: np.ndarray) -> np.ndarray:
    """固定特征，无训练集统计，也不使用标签。RGB 输入 uint8。"""
    pixels = rgb.astype(np.float32) / 255.0
    border = np.concatenate([pixels[:3].reshape(-1, 3), pixels[-3:].reshape(-1, 3),
                             pixels[:, :3].reshape(-1, 3), pixels[:, -3:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    delta = pixels - bg
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    local = gray - cv2.GaussianBlur(gray, (0, 0), 5)
    distance = np.max(np.abs(delta), axis=2)
    return np.ascontiguousarray(np.dstack([delta, local, distance]).transpose(2, 0, 1))


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class GlyphUNet(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.enc1 = block(5, width)
        self.enc2 = block(width, width * 2)
        self.center = block(width * 2, width * 4)
        self.dec2 = block(width * 6, width * 2)
        self.dec1 = block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        c = self.center(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([F.interpolate(c, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], 1))
        # 以固定色差阈值为起点，训练网络学习形状和上下文修正。
        return self.head(d1) + (x[:, 4:5] - 0.10) * 12.0


def load_model(checkpoint, threads: int = 4) -> GlyphUNet:
    torch.set_num_threads(threads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = GlyphUNet(state["width"])
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


@torch.inference_mode()
def predict(model: GlyphUNet, rgb: np.ndarray, tile: int = 128) -> np.ndarray:
    """原分辨率重叠分块推理，整图一次计算背景特征，避免拼缝。"""
    h, w = rgb.shape[:2]
    x = features(rgb)
    if max(h, w) <= tile:
        padded = np.pad(x, ((0, 0), (0, (-h) % 4), (0, (-w) % 4)), mode="reflect")
        return model(torch.from_numpy(padded)[None]).sigmoid()[0, 0, :h, :w].numpy()
    context, step = 24, tile - 48
    padded = np.pad(x, ((0, 0), (context, tile), (context, tile)), mode="reflect")
    out = np.zeros((h, w), np.float32)
    for y in range(0, h, step):
        for xx in range(0, w, step):
            part = torch.from_numpy(padded[:, y:y + tile, xx:xx + tile])[None]
            prob = model(part).sigmoid()[0, 0].numpy()
            hh, ww = min(step, h - y), min(step, w - xx)
            out[y:y + hh, xx:xx + ww] = prob[context:context + hh, context:context + ww]
    return out
