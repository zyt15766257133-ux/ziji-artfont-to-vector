"""Repeatable full-service comparison and fourfold vector-rendered details."""
from pathlib import Path
import sys, io, json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.native_render import render_svg

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'eval/results/shadow_quality'


def white(pixels):
    a=pixels[:,:,3:4]/255.
    return np.uint8(np.rint(pixels[:,:,:3]*a+255*(1-a)))


def main(names=None):
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    manifest=json.loads((ROOT/'samples/manifest.json').read_text(encoding='utf-8'))
    if names:
        manifest=[item for item in manifest if item['file'] in names]
    for item in manifest:
        name=item['file'];data=(ROOT/'samples/images'/name).read_bytes()
        opts=VectorizeOptions.from_preset('balanced',glyph_count=item['glyphs'])
        with patch('app.pipeline.shadow_model.shadow_document',return_value=None):
            before=convert_bytes(name,data,opts,strict_input=False,with_preview=True)
        after=convert_bytes(name,data,opts,strict_input=False,with_preview=True)
        record={'file':name,'new_model':bool(after.stats.get('source_shadow_model')),
                'unchanged_svg':before.svg==after.svg,'before_ms':before.elapsed_ms,
                'after_ms':after.elapsed_ms,'error':after.error}
        records.append(record)
        print(json.dumps(record),flush=True)
        if name!='12_shadow.png':
            continue
        source=np.asarray(Image.open(io.BytesIO(after.source_png)).convert('RGBA'))
        reference=white(source)
        dark=reference.mean(2)<80
        band=(cv2.dilate(dark.astype(np.uint8),np.ones((5,5),np.uint8))
              != cv2.erode(dark.astype(np.uint8),np.ones((5,5),np.uint8)))
        visible=reference.mean(2)<250
        panels=[('原图',reference)]
        scores={}
        for label,result in [('before',before),('after',after)]:
            (OUT/f'{label}.svg').write_text(result.svg,encoding='utf-8')
            (OUT/f'{label}.json').write_text(json.dumps(result.to_dict(False),ensure_ascii=False,indent=2),encoding='utf-8')
            pixels=render_svg(result.svg); painted=white(pixels)
            Image.fromarray(pixels).save(OUT/f'{label}.png')
            output_dark=painted.mean(2)<80
            scores[label]={'dark_body_area_ratio':float(output_dark.sum()/dark.sum()),
                           'dark_body_iou':float((output_dark&dark).sum()/(output_dark|dark).sum()),
                           'edge_rgb_mae_255':float(np.abs(painted.astype(float)-reference)[band].mean()),
                           'visible_rgb_mae_255':float(np.abs(painted.astype(float)-reference)[visible].mean()),
                           'nodes':result.stats['node_total'],'elapsed_ms':result.elapsed_ms}
            panels.append(('修复前' if label=='before' else '修复后',painted))
        (OUT/'source.png').write_bytes(after.source_png)
        (OUT/'metrics.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
        f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
        board=Image.new('RGB',(reference.shape[1],(reference.shape[0]+36)*3),'#f5f6f8')
        for i,(label,painted) in enumerate(panels):
            y=i*(reference.shape[0]+36)
            ImageDraw.Draw(board).text((12,y+7),label,font=f,fill='#374151')
            board.paste(Image.fromarray(painted),(0,y+36))
        board.save(OUT/'comparison.png')
        # Fourfold vector rendering exposes curves; source enlargement is only
        # interpolated raster evidence, never claimed to restore source details.
        crops=[]
        for label in ['before','after']:
            import xml.etree.ElementTree as ET
            root=ET.fromstring((OUT/f'{label}.svg').read_text(encoding='utf-8'))
            root.set('width',str(reference.shape[1]*4));root.set('height',str(reference.shape[0]*4))
            enlarged=white(render_svg(ET.tostring(root,encoding='unicode')))
            crops.append(Image.fromarray(enlarged).crop((230*4,20*4,410*4,185*4)))
        zoom=Image.new('RGB',(1440,696),'#f5f6f8')
        for i,im in enumerate(crops):
            ImageDraw.Draw(zoom).text((12+i*720,7),'修复前 · 4倍' if i==0 else '修复后 · 4倍',font=f,fill='#374151')
            zoom.paste(im,(i*720,36))
        zoom.save(OUT/'detail_4x.png')
    (OUT/('focused_regression.json' if names else 'regression.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    main(sys.argv[1:])
