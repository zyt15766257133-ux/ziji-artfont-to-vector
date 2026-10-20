"""Save source-preserving before/after service evidence without replacing baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.native_render import render_svg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    parser.add_argument('--samples', nargs='*')
    args = parser.parse_args()
    output = ROOT / 'eval/results/soft_effects' / args.label
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted((ROOT / 'samples/images').glob('*.png')):
        if args.samples and path.stem not in args.samples:
            continue
        original = path.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        started = time.perf_counter()
        result = convert_bytes(path.name, original, VectorizeOptions.from_preset('fidelity'),
                               with_preview=True, strict_input=False)
        wall = (time.perf_counter() - started) * 1000
        folder = output / path.stem
        folder.mkdir()
        (folder / 'result.svg').write_text(result.svg, encoding='utf-8')
        Image.fromarray(render_svg(result.svg)).save(folder / 'render.png')
        if result.preview_png:
            (folder / 'comparison.png').write_bytes(result.preview_png)
        source = np.asarray(Image.open(path).convert('RGBA'))
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
        row = {'file': path.name, 'source_sha256': digest, 'wall_ms': wall,
               'reported_ms': result.elapsed_ms, 'svg_bytes': len(result.svg.encode('utf-8')),
               'stats': result.stats, 'metrics': result.metrics, 'timings': result.timings}
        if path.stem == '21_neon':
            pixels = render_svg(result.svg)
            bg = np.median(source[0, :, :3], axis=0)
            distance = np.max(np.abs(source[:, :, :3].astype(float)-bg), axis=2)
            halo = (distance > 2) & (distance < 120)
            delta = (pixels[:, :, :3].astype(float)-source[:, :, :3]) @ [.2126, .7152, .0722]
            row['halo_roi'] = {'pixels': int(halo.sum()), 'signed_srgb_luma_mean': float(delta[halo].mean()),
                               'srgb_luma_mae': float(np.abs(delta[halo]).mean()),
                               'srgb_luma_p95': float(np.percentile(np.abs(delta[halo]), 95))}
        rows.append(row)
        (output / 'metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{path.name}: {wall:.0f} ms, {result.stats.get("node_total")} nodes, '
              f'{result.stats.get("algorithm", result.stats["engine"])}', flush=True)


if __name__ == '__main__':
    main()
