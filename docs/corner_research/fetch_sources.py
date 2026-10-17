"""Archive public primary sources for the corner-preservation literature review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
from datetime import datetime, timezone
from lxml import html

ROOT = Path(__file__).parent / 'sources'
SOURCES = {
    'illustrator': 'https://helpx.adobe.com/illustrator/using/image-trace.html',
    'inkscape': 'https://inkscape.org/doc/tutorials/tracing/tutorial-tracing.html',
    'potrace': 'https://potrace.sourceforge.net/',
    'potrace-man': 'https://potrace.sourceforge.net/potrace.1.html',
    'vectormagic': 'https://vectormagic.com/',
    'vectorizer-ai': 'https://vectorizer.ai/',
    'vtracer': 'https://raw.githubusercontent.com/visioncortex/vtracer/master/README.md',
    'svgcode': 'https://raw.githubusercontent.com/tomayac/SVGcode/main/README.md',
    'polyfit': 'https://www.cs.ubc.ca/labs/imager/tr/2020/ClipArtVectorization/',
    'keypoint-flow': 'https://puhachov.xyz/publications/keypoint-driven-polyvector-flow/',
    'polyvector': 'https://arxiv.org/html/1801.01922v2',
    'live': 'https://raw.githubusercontent.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization/main/README.md',
    'hanmo': 'https://www.hanmoai.com/detail/19.html',
    'pixso': 'https://pixso.cn/designskills/image-to-svg-online-generator/',
}

def fetch(item):
    key, url = item
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=35) as response:
            raw = response.read()
            encoding = response.headers.get_content_charset() or 'utf-8'
            final_url = response.url
        text = raw.decode(encoding, errors='replace')
        if '<html' in text[:1000].lower() or '<!doctype' in text[:1000].lower():
            doc = html.fromstring(text)
            for el in doc.xpath('//script|//style|//nav|//footer'):
                el.drop_tree()
            content = '\n'.join(line.strip() for line in doc.text_content().splitlines() if line.strip())
        else:
            content = text
        (ROOT / (key + '.txt')).write_text(content, encoding='utf-8')
        return {'id': key, 'url': url, 'final_url': final_url, 'characters': len(content), 'status': 'ok'}
    except Exception as exc:
        return {'id': key, 'url': url, 'status': 'error', 'error': str(exc)}

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(fetch, SOURCES.items()))
    (ROOT / 'manifest.json').write_text(json.dumps({'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'sources': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
