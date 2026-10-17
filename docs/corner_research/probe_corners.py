"""Controlled geometry probe; calls existing fitters without changing application code."""
from pathlib import Path
import sys
import json
import time
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from app.pipeline.svgutil import fit_contour, contour_to_path_d, mark_corners
from app.pipeline.direct import fit_mask_paths
from app.pipeline.native_render import render_svg, shutdown
import vtracer

OUT = Path(__file__).parent / 'probe'
W, H, SCALE = 224, 176, 4
CASES = {
    'rectangle': [(36,36),(188,36),(188,140),(36,140)],
    'concave_L': [(36,28),(86,28),(86,90),(190,90),(190,140),(36,140)],
    'acute_tip': [(28,140),(111,22),(194,140),(125,118),(111,154),(97,118)],
    'small_notch': [(28,36),(104,36),(104,42),(110,42),(110,36),(196,36),(196,140),(28,140)],
}

def svg(d):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><path fill="black" fill-rule="evenodd" d="{d}"/></svg>'

def pixels(text, scale=1):
    if scale != 1:
        if 'viewBox=' not in text:
            text = text.replace('<svg ',f'<svg viewBox="0 0 {W} {H}" ',1)
        text = re.sub(r'\bwidth="[^"]*"', f'width="{W*scale}"', text, count=1)
        text = re.sub(r'\bheight="[^"]*"', f'height="{H*scale}"', text, count=1)
    return render_svg(text, background=(255,255,255,255))

def ink(p):
    return p[:,:,:3].mean(axis=2)<128

def iou(a,b,roi=None):
    if roi is not None:
        a,b=a[roi],b[roi]
    return float(np.logical_and(a,b).sum()/max(1,np.logical_or(a,b).sum()))

def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    thumbnails=[]
    for name, vertices in CASES.items():
        pts=np.array(vertices,dtype=float)
        d='M '+' L '.join(f'{x},{y}' for x,y in pts)+' Z'
        reference=svg(d)
        source=pixels(reference)
        Image.fromarray(source).save(OUT/f'{name}_input.png')
        mask=ink(source).astype(np.uint8)*255
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contour=max(contours,key=cv2.contourArea)
        truth=ink(pixels(reference,SCALE))
        yy,xx=np.indices(truth.shape)
        roi=np.zeros(truth.shape,bool)
        for x,y in pts:
            roi|=(xx/SCALE-x)**2+(yy/SCALE-y)**2 <= 8**2
        versions=[('Reference',reference, len(pts),None)]
        for label,budget,smooth in [('Contour 200',200,True),('Contour 12',12,True),('No smoothing',200,False)]:
            begin=time.perf_counter()
            fitted=fit_contour(contour,tolerance=.55,corner_angle=62,max_nodes=budget,smooth=smooth,snap_deg=0)
            result=svg(contour_to_path_d(fitted))
            versions.append((label,result,len(fitted.segs),1000*(time.perf_counter()-begin)))
            if label=='Contour 200':
                rows.append({'case':name,'diagnostic':'all_simplified_vertices_marked_as_corners','value':bool(mark_corners(fitted.points,62).all()),'vertices':len(fitted.points)})
        begin=time.perf_counter()
        paths,nodes=fit_mask_paths(mask,.55)
        versions.append(('Direct .55',svg(' '.join(paths)),nodes,1000*(time.perf_counter()-begin)))
        begin=time.perf_counter()
        traced=vtracer.convert_raw_image_to_svg((OUT/f'{name}_input.png').read_bytes(),'png',colormode='binary',mode='spline',filter_speckle=0,corner_threshold=60,length_threshold=4,splice_threshold=45,path_precision=3)
        versions.append(('VTracer 60',traced,None,1000*(time.perf_counter()-begin)))
        cells=[]
        for label,result,nodes,millis in versions:
            tag=label.replace(' ','_')
            (OUT/f'{name}_{tag}.svg').write_text(result,encoding='utf-8')
            image=pixels(result,SCALE)
            estimate=ink(image)
            total,local=iou(truth,estimate),iou(truth,estimate,roi)
            if label!='Reference':
                rows.append({'case':name,'method':label,'global_iou':round(total,5),'corner_roi_iou':round(local,5),'segments_or_reported_nodes':nodes,'fit_ms_excluding_render':round(millis,2)})
            cell=Image.new('RGB',(280,284),'white')
            drawing=ImageDraw.Draw(cell)
            drawing.text((8,8),label,fill='black')
            drawing.text((8,26),f'IoU {total:.3f} | corner {local:.3f}',fill='black')
            cell.paste(Image.fromarray(image).convert('RGB').resize((224,176)),(28,50))
            # Magnified uppermost selected corner; antialiasing is preserved in reference.
            x,y=pts[1] if name=='acute_tip' else pts[0]
            crop=Image.fromarray(image).convert('RGB').crop((int((x-10)*SCALE),int((y-10)*SCALE),int((x+10)*SCALE),int((y+10)*SCALE)))
            cell.paste(crop.resize((52,52)),(112,228))
            cells.append(cell)
        thumbnails.append((name,cells))
    board=Image.new('RGB',(6*280, len(thumbnails)*312),'#e5e7eb')
    draw=ImageDraw.Draw(board)
    for index,(name,cells) in enumerate(thumbnails):
        draw.text((8,index*312+5),name,fill='black')
        for col,cell in enumerate(cells):
            board.paste(cell,(col*280,index*312+24))
    board.save(OUT/'comparison.png')
    focus=Image.new('RGB',(4*280,284),'white')
    for col,index in enumerate([0,1,4,5]):
        focus.paste(thumbnails[0][1][index],(col*280,0))
    focus.save(OUT/'rectangle_focus.png')
    (OUT/'results.json').write_text(json.dumps({'scope':'Isolated fitters on controlled synthetic polygons; not complete API or commercial-product benchmark. Reference is exact geometry rendered at 4x. Corner ROI radius=8 source pixels.','source_size':[W,H],'render_scale':SCALE,'rows':rows},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    shutdown()

if __name__=='__main__':
    main()
