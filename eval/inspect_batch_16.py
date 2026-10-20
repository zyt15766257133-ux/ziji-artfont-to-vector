"""Read the downloaded pair and show it against explicit preview backgrounds."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.native_render import render_svg


def main():
    archive = Path('C:/Users/ZYTCOM/Downloads/vectorized-1792481521002.zip')
    output = ROOT.parent / '第16组显示核查_2026-10-20'
    output.mkdir(exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    canvas = Image.new('RGB', (1050, 560), '#e8e8e8')
    draw = ImageDraw.Draw(canvas)
    records = []
    with zipfile.ZipFile(archive) as source:
        for row, name in enumerate(('16_gradient_stroke.svg', '16_gradient_stroke__2.svg')):
            payload = source.read(name)
            (output / name).write_bytes(payload)
            rgba = render_svg(payload.decode('utf-8'))
            # Crop a common region only for the comparison image. SVG copies
            # remain byte-for-byte identical to the archive entries.
            alpha = rgba[:, :, 3:4].astype(float) / 255
            records.append({'file': name, 'visible_pixels': int((rgba[:, :, 3] > 0).sum())})
            draw.text((12, row * 280 + 8), name, fill='black', font=font)
            for column, (label, color) in enumerate((('白底预览', 255), ('黑底预览', 0), ('灰底预览', 128))):
                composite = np.rint(rgba[:, :, :3] * alpha + color * (1 - alpha)).astype('uint8')
                panel = Image.fromarray(composite).crop((120, 385, 880, 625)).resize((330, 170))
                x, y = column * 350 + 10, row * 280 + 85
                draw.text((x, y - 33), label, fill='black', font=font)
                canvas.paste(panel, (x, y))
    canvas.save(output / '16_黑白灰背景对照.png')
    print(json.dumps({'output': str(output), 'files': records}, ensure_ascii=False))


if __name__ == '__main__':
    main()
