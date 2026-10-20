"""Read public primary sources; never execute downloaded code."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import json
from lxml import html

URLS = {
    'vtracer': 'https://raw.githubusercontent.com/visioncortex/vtracer/master/README.md',
    'vtracer-source': 'https://raw.githubusercontent.com/visioncortex/vtracer/master/crates/vtracer/src/pipeline.rs',
    'gimp': 'https://docs.gimp.org/3.0/en/gimp-filter-color-to-alpha.html',
    'imagemagick': 'https://usage.imagemagick.org/masking/',
    'vectorizer-ai': 'https://vectorizer.ai/',
    'vectormagic': 'https://vectormagic.com/support/tutorials/basic',
    'inkscape': 'https://inkscape.org/doc/tutorials/tracing/tutorial-tracing.html',
}
ROOT = Path(__file__).parent


def fetch(item):
    key, url = item
    try:
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=25) as response:
            raw, final_url = response.read(), response.url
        content = raw.decode('utf-8', 'replace')
        if '<html' in content[:2000].lower() or '<!doctype' in content[:2000].lower():
            doc = html.fromstring(raw)
            for element in doc.xpath('//script|//style|//nav|//footer'):
                element.drop_tree()
            content = '\n'.join(line.strip() for line in doc.text_content().splitlines() if line.strip())
        (ROOT/(key+'.txt')).write_text(content, encoding='utf-8')
        return dict(id=key, url=url, final_url=final_url, characters=len(content), ok=True)
    except Exception as exc:
        return dict(id=key, url=url, ok=False, error=str(exc))


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(fetch, URLS.items()))
    manifest = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(), sources=results)
    (ROOT/'sources.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
