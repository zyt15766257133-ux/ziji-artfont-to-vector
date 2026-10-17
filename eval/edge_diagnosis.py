"""Reproduce transparent-lettering edges through the complete conversion service."""
from pathlib import Path
import sys
import json
import io
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.native_render import render_svg

def composite(rgba):
    rgba=np.asarray(rgba)
    yy,xx=np.indices(rgba.shape[:2])
    board=np.where((xx//12+yy//12)%2,238,224)[:,:,None]
    a=rgba[:,:,3:4]/255
    return Image.fromarray(np.uint8(np.clip(rgba[:,:,:3]*a+board*(1-a),0,255)))

def main(label):
    out=ROOT/'eval/results/edge_diagnosis'/label
    if label == 'baseline' and (out/'output.svg').exists():
        raise FileExistsError('Baseline evidence already exists; use a different label to preserve it.')
    out.mkdir(parents=True,exist_ok=True)
    name='18_transparent.png'
    result=convert_bytes(name,(ROOT/'samples/images'/name).read_bytes(),
                         VectorizeOptions.from_preset('balanced',glyph_count=3),
                         strict_input=False,with_preview=True)
    if result.error: raise RuntimeError(result.error)
    (out/'output.svg').write_text(result.svg,encoding='utf-8')
    (out/'aligned.png').write_bytes(result.source_png)
    raster=render_svg(result.svg)
    Image.fromarray(raster).save(out/'output.png')
    raw=np.asarray(Image.open(io.BytesIO(result.source_png)).convert('RGBA'))
    panels=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for title,image in [('原图',raw),('矢量结果 · '+label,raster)]:
        preview=composite(image)
        preview.thumbnail((850,340),Image.Resampling.LANCZOS)
        panel=Image.new('RGB',(900,preview.height+42),'#f5f6f8')
        ImageDraw.Draw(panel).text((16,8),title,font=font,fill='#4b5563')
        panel.paste(preview,((900-preview.width)//2,40))
        panels.append(panel)
    board=Image.new('RGB',(900,sum(p.height for p in panels)),'white')
    y=0
    for panel in panels:
        board.paste(panel,(0,y));y+=panel.height
    board.save(out/'comparison.png')
    if label == 'fixed':
        baseline = out.parent/'baseline'
        if (baseline/'output.png').exists():
            before=np.asarray(Image.open(baseline/'output.png').convert('RGBA'))
            rows=[]
            for title,image in [('原图',raw),('修复前',before),('修复后',raster)]:
                preview=composite(image)
                panel=Image.new('RGB',(900,preview.height+42),'#f5f6f8')
                ImageDraw.Draw(panel).text((16,8),title,font=font,fill='#4b5563')
                panel.paste(preview,((900-preview.width)//2,40))
                rows.append(panel)
            comparison=Image.new('RGB',(900,sum(p.height for p in rows)),'white')
            y=0
            for panel in rows:
                comparison.paste(panel,(0,y));y+=panel.height
            comparison.save(out/'before_after.png')
            source_alpha=raw[:,:,3]/255.
            kernel=np.ones((3,3),np.uint8)
            band=cv2.dilate((source_alpha>0).astype(np.uint8),kernel)>0
            band &= cv2.erode((source_alpha>.99).astype(np.uint8),kernel)==0
            summary={
                'sample':name,'scope':'One built-in transparent-lettering sample; complete service call, fresh conversion.',
                'edge_band_pixels':int(band.sum()),
                'before_edge_alpha_mae':float(np.abs(before[:,:,3]/255.-source_alpha)[band].mean()),
                'after_edge_alpha_mae':float(np.abs(raster[:,:,3]/255.-source_alpha)[band].mean()),
                'before_nodes':json.loads((baseline/'result.json').read_text(encoding='utf-8'))['stats']['node_total'],
                'after_nodes':result.stats['node_total'],
            }
            (out/'before_after_metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'result.json').write_text(json.dumps({'stats':result.stats,'metrics':result.metrics,'timings':result.timings},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'stats':result.stats,'shape_iou':result.metrics.get('shape_iou'),'nodes':result.metrics.get('node_per_glyph'),'timings':result.timings},ensure_ascii=False))

if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else 'baseline')
