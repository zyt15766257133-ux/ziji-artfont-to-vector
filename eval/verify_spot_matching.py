"""Verify the optional palette branch against a running local HTTP service."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.native_render import render_svg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8768')
    parser.add_argument('--out', type=Path, default=ROOT.parent / '专色匹配验证_2026-10-19')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = np.full((500, 500, 4), 255, dtype=np.uint8)
    source[40:460, 40:240, :3] = [220, 30, 50]
    source[40:460, 260:460, :3] = [25, 65, 205]
    buffer = io.BytesIO()
    Image.fromarray(source).save(buffer, format='PNG')
    data = buffer.getvalue()
    (args.out / 'input.png').write_bytes(data)

    with httpx.Client(base_url=args.url, timeout=90) as client:
        def convert(**options):
            return client.post('/api/convert', files={'file': ('spot-test.png', data, 'image/png')},
                               data={'preset': 'fidelity', 'with_preview': 'false',
                                     'soft_effects': 'false', **options})

        def save(name, response):
            assert response.status_code == 200, response.text
            result = response.json()
            assert not result.get('error'), result
            (args.out / f'{name}.svg').write_text(result['svg'], encoding='utf-8')
            pixels = render_svg(result['svg'])
            Image.fromarray(pixels).save(args.out / f'{name}.png')
            return result, pixels

        baseline, original_pixels = save('default-off', convert())
        disabled, _ = save('palette-but-off', convert(spot_colors='#00ff00'))
        assert disabled['svg'] == baseline['svg']
        enabled, pixels = save('matching-on', convert(spot_matching='true', spot_colors='#00ff00'))
        assert enabled['stats']['spot_matching']['matched_layers'] > 0
        for x in (140, 360):
            assert pixels[250, x, :3].tolist() == [0, 255, 0]
        np.testing.assert_array_equal(pixels[:, :, 3], original_pixels[:, :, 3])
        assert pixels[0, 0, 3] == 0
        for palette in ('', '#12345', '#123456,bad'):
            response = convert(spot_matching='true', spot_colors=palette)
            assert response.status_code == 422, response.text
        archive, _ = save('pixel-exact', convert(preset='pixel_exact', spot_matching='true', spot_colors='invalid'))
        assert archive['metrics']['source_fidelity']['native_grid_exact']
        assert not archive['options']['spot_matching']
        vtracer, _ = save('vtracer-matching', convert(preset='balanced', engine='vtracer',
                                                    spot_matching='true', spot_colors='#00ff00'))
        assert vtracer['stats']['spot_matching']['matched_layers'] > 0
        export = client.post('/api/export', data={'svg': enabled['svg'], 'fmt': 'svg', 'stem': 'spot-matched'})
        assert export.status_code == 200, export.text
        np.testing.assert_array_equal(render_svg(export.text), pixels)
        (args.out / 'downloaded.svg').write_bytes(export.content)

    checks = {'default_off': True, 'palette_ignored_when_off': True,
              'fidelity_matching': True, 'transparent_background_and_alpha_preserved': True,
              'invalid_palette_returns_422': True, 'pixel_exact_unchanged': True,
              'vtracer_matching': True, 'download_matches_result': True}
    report = {'url': args.url, 'checks': checks, 'spot_matching': enabled['stats']['spot_matching'],
              'elapsed_ms': enabled['elapsed_ms'],
              'scope': 'generated 500x500 test image; engineering checks, not printing acceptance'}
    (args.out / 'http-verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
