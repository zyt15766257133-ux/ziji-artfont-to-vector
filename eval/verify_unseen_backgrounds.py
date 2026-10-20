"""Independent, previously unseen lettering uploaded through the live HTTP API.

Truth comes from Pillow glyph rendering, never the product foreground mask.
These are evaluation fixtures and are not added to the sample manifest.
"""
import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import sys

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.native_render import render_svg


def lettering():
    scale = 3
    image = Image.new('RGBA', (640*scale, 512*scale))
    font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 225*scale)
    draw = ImageDraw.Draw(image)
    # Top two characters deliberately cross the canvas boundary.
    for char, x, y, color in [('艺', 65, -9, '#b92e60'), ('术', 341, -9, '#246fba'),
                              ('新', 65, 262, '#713fac'), ('图', 341, 262, '#cf7041')]:
        draw.text((x*scale, y*scale), char, font=font, fill=color, anchor='lt')
    return np.array(image.resize((640, 512), Image.Resampling.LANCZOS))


def hashes():
    return {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'samples/images').glob('*')) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--url', default='http://127.0.0.1:8768')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    before = hashes()
    truth = lettering()
    Image.fromarray(truth).save(args.output/'independent_foreground_truth.png')
    blank = cv2.erode((truth[:, :, 3] == 0).astype('uint8'), np.ones((21, 21), 'uint8')) > 0
    near_blank = cv2.erode((truth[:, :, 3] == 0).astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    body = cv2.erode((truth[:, :, 3] == 255).astype('uint8'), np.ones((7, 7), 'uint8')) > 0
    backgrounds = [('white', (255, 255, 255)), ('dark', (13, 18, 27)),
                   ('green', (32, 165, 105)), ('gold', (233, 187, 52))]
    alpha = truth[:, :, 3:4]/255.
    rows, panels = [], []
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 19)
    with httpx.Client(base_url=args.url, timeout=240) as client:
        health = client.get('/api/health').raise_for_status().json()
        for title, bg in backgrounds:
            source = np.rint(truth[:, :, :3]*alpha+np.array(bg)*(1-alpha)).astype('uint8')
            for fmt, ext in [('PNG', 'png'), ('JPEG', 'jpg'), ('WEBP', 'webp')]:
                name = f'unseen-{title}.{ext}'
                payload = io.BytesIO()
                Image.fromarray(source).save(payload, format=fmt, quality=85)
                data = payload.getvalue()
                (args.output/name).write_bytes(data)
                files = {'file': (name, data, 'image/'+('jpeg' if ext == 'jpg' else ext))}
                preflight = client.post('/api/inspect', files=files).raise_for_status().json()
                result = client.post('/api/convert', files=files, data={
                    'preset': 'fidelity', 'with_preview': 'false', 'strict_input': 'false',
                }).raise_for_status().json()
                assert not result.get('error'), result
                svg = result['svg']
                (args.output/(name+'.svg')).write_text(svg, encoding='utf-8')
                output = render_svg(svg)
                Image.fromarray(output).save(args.output/(name+'.render.png'))
                transparent_fraction = float(np.mean(output[:, :, 3][blank] == 0))
                near_blank_leak = float(np.mean(output[:, :, 3][near_blank] > 32))
                body_missing = float(np.mean(output[:, :, 3][body] < 240))
                body_color_mae = float(np.abs(output[:, :, :3].astype(float)-truth[:, :, :3])[body].mean())
                removal = result['metrics']['background_removal']
                record = dict(name=name, bytes=len(data), shape=[512, 640],
                    preflight_reason=preflight['background_removal']['reason'],
                    preflight_subject_ratio=preflight['quality'].get('orig_bbox_ratio'),
                    blank_transparent_fraction=transparent_fraction, missing_body_fraction=body_missing,
                    near_blank_leak_fraction=near_blank_leak,
                    body_color_mae=body_color_mae, removal=removal,
                    representation=result['stats']['representation'],
                    retried='foreground_retry' in result['stats'], elapsed_ms=result['elapsed_ms'])
                record['passed'] = (preflight['background_removal']['applied'] and removal['applied']
                    and removal['export_verification']['passed'] and transparent_fraction > .995
                    and near_blank_leak < .0005 and body_missing < .005
                    and body_color_mae < 12 and '<image' not in svg)
                rows.append(record)
                print(json.dumps(record, ensure_ascii=True), flush=True)
                if fmt == 'JPEG':
                    y, x = np.mgrid[:512, :640]
                    checker = np.where(((x//16+y//16) % 2)[:, :, None], 225, 245)
                    out_alpha = output[:, :, 3:4]/255.
                    checker_image = np.rint(output[:, :, :3]*out_alpha+checker*(1-out_alpha)).astype('uint8')
                    blue = np.rint(output[:, :, :3]*out_alpha+np.array([33, 64, 110])*(1-out_alpha)).astype('uint8')
                    row = Image.new('RGB', (1200, 355), 'white')
                    for i, (image, label) in enumerate([(source, title+' / JPG input'),
                                                     (checker_image, 'SVG / transparent checker'),
                                                     (blue, 'SVG / blue background')]):
                        ImageDraw.Draw(row).text((i*400+8, 8), label, font=font, fill='black')
                        row.paste(Image.fromarray(image).resize((400, 320)), (i*400, 35))
                    panels.append(row)
    comparison = Image.new('RGB', (1200, 355*len(panels)), 'white')
    for i, row in enumerate(panels):
        comparison.paste(row, (0, i*355))
    comparison.save(args.output/'comparison.png')
    after = hashes()
    report = dict(server=health, cases=rows, all_passed=all(r['passed'] for r in rows),
                  sample_originals_unchanged=before == after, sample_hashes_before=before, sample_hashes_after=after,
                  truth='Pillow supersampled glyph RGBA; independent of product segmentation',
                  scope='12 unseen Chinese lettering uploads, 640x512, 4 backgrounds x PNG/JPEG/WebP; not arbitrary images')
    (args.output/'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert report['all_passed'] and report['sample_originals_unchanged']


if __name__ == '__main__':
    main()
