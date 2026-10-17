"""All-sample, source-referenced comparisons. Run from the project root."""
from pathlib import Path
import sys, json, time, hashlib, traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.native_render import render_svg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'eval/results/full_folder_fidelity'

def save_rgba(path, rgba):
    Image.fromarray(rgba.astype('uint8'), 'RGBA').save(path)

def composite(a, bg=255):
    a = a.astype(float)
    return a[:,:,:3]*a[:,:,3:4]/255 + bg*(1-a[:,:,3:4]/255)

def errors(source, output):
    # Both light and dark backing; no alpha-threshold silhouette proxy.
    border = np.r_[source[0,:,:3], source[-1,:,:3], source[:,0,:3], source[:,-1,:3]]
    bg = np.median(border,axis=0)
    backgrounds = (0,255) if np.any(source[:,:,3]<255) else (bg,)
    diffs = [np.abs(composite(source,b)-composite(output,b)) for b in backgrounds]
    d = np.maximum.reduce(diffs)
    active = (np.linalg.norm(source[:,:,:3].astype(float)-bg,axis=2)>5) | (source[:,:,3]<250)
    if active.any():
        import cv2
        active = cv2.dilate(active.astype('uint8'),np.ones((5,5),'uint8'))>0
    else:
        active = np.ones(source.shape[:2],bool)
    return {'mae':float(d.mean()), 'active_mae':float(d[active].mean()),
            'max_error':float(d.max()), 'p99_active':float(np.percentile(d[active],99)),
            'pixels_over_8':int((d.max(2)>8).sum()),
            'exact_visible_pixels':bool(d.max()==0)}

def baseline():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest = {x['file']:x for x in json.loads((ROOT/'samples/manifest.json').read_text('utf-8'))}
    rows=[]
    for p in sorted((ROOT/'samples/images').glob('*.png')):
        dest=OUT/p.stem;dest.mkdir(exist_ok=True)
        row={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        try:
            item=manifest[p.name]
            result=convert_bytes(p.name,p.read_bytes(),VectorizeOptions.from_preset('balanced'),
                                 with_preview=True,strict_input=False,glyph_count=item['glyphs'])
            if result.error: raise RuntimeError(result.error)
            (dest/'baseline.svg').write_text(result.svg,encoding='utf-8')
            (dest/'source.png').write_bytes(result.source_png)
            output=render_svg(result.svg);save_rgba(dest/'baseline.png',output)
            source=np.array(Image.open(dest/'source.png').convert('RGBA'))
            row.update(baseline=errors(source,output),baseline_stats=result.stats,
                       baseline_ms=result.elapsed_ms,transform=result.metrics['source_transform'])
        except Exception as exc:
            row['error']=str(exc);traceback.print_exc()
        rows.append(row)
        (OUT/'baseline.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(p.name,row.get('baseline',{}).get('active_mae'),row.get('error',''),flush=True)
    return rows

def candidates():
    import vtracer
    new = hasattr(vtracer,'Config')
    name = 'v1' if new else 'v0'
    records=[]
    for p in sorted((ROOT/'samples/images').glob('*.png')):
        dest=OUT/p.stem
        source=np.array(Image.open(dest/'source.png').convert('RGBA'))
        for hierarchy in (('stacked','cutout','watershed') if new else ('stacked','cutout')):
            key=name+'_'+hierarchy
            started=time.perf_counter()
            try:
                if new:
                    cfg=vtracer.Config(hierarchical='cutout' if hierarchy=='watershed' else hierarchy,
                                       mode='spline',filter_speckle=0,color_precision=8,
                                       layer_difference=1,path_precision=4,optimize=0)
                    if hierarchy=='watershed':
                        cfg.clustering='watershed';cfg.watershed_detail=256
                    svg=cfg.convert_bytes((dest/'source.png').read_bytes())
                else:
                    svg=vtracer.convert_raw_image_to_svg((dest/'source.png').read_bytes(),'png',
                        colormode='color',hierarchical=hierarchy,mode='spline',
                        filter_speckle=0,color_precision=8,layer_difference=1,
                        corner_threshold=60,length_threshold=3.5,max_iterations=10,
                        splice_threshold=45,path_precision=4)
                (dest/(key+'.svg')).write_text(svg,encoding='utf-8')
                rgba=render_svg(svg);save_rgba(dest/(key+'.png'),rgba)
                row={'file':p.name,'candidate':key,'ms':1000*(time.perf_counter()-started),
                     'bytes':len(svg.encode()),'metrics':errors(source,rgba)}
            except BaseException as exc:
                row={'file':p.name,'candidate':key,'error':str(exc)}
            records.append(row)
            (OUT/(name+'.json')).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
            print(p.name,key,row.get('metrics',{}).get('active_mae'),row.get('error',''),flush=True)

def final_outputs():
    from app.pipeline.fidelity import source_error
    records=[]
    baseline_rows=json.loads((OUT/('final.json' if '--smooth-only' in sys.argv else 'baseline.json')).read_text('utf-8'))
    wanted=set(sys.argv[sys.argv.index('--only')+1].split(',')) if '--only' in sys.argv else None
    for row in baseline_rows:
        if wanted is not None and row['file'] not in wanted:
            records.append(row)
            continue
        p=ROOT/'samples/images'/row['file'];dest=OUT/p.stem
        source=np.array(Image.open(p).convert('RGBA'))
        # Recompute baseline with corrected same-background comparison.
        cropped=np.array(Image.open(dest/'source.png').convert('RGBA'))
        old=np.array(Image.open(dest/'baseline.png').convert('RGBA'))
        row['baseline']=errors(cropped,old)
        transform=row['transform']
        if transform['rotation'] is None:
            if np.all(source[:,:,3]==255):
                old_full=np.full_like(source,255)
                old_full[:,:,:3]=source[0,0,:3]
                a=old[:,:,3:4].astype(float)/255
                old=old.copy()
                old[:,:,:3]=np.rint(old[:,:,:3]*a+source[0,0,:3]*(1-a)).astype('uint8')
                old[:,:,3]=255
            else:
                old_full=np.zeros_like(source)
            x0,y0,x1,y1=transform['crop_box'] or (0,0,source.shape[1],source.shape[0])
            old_full[y0:y1,x0:x1]=old
            row['baseline_native']=source_error(source,old_full)
            save_rgba(dest/'baseline-native.png',old_full)
        for mode in ('fidelity','pixel_exact'):
            if '--smooth-only' in sys.argv and mode=='pixel_exact': continue
            started=time.perf_counter()
            try:
                result=convert_bytes(p.name,p.read_bytes(),VectorizeOptions.from_preset(mode),
                                     with_preview=True,strict_input=False)
                if result.error: raise RuntimeError(result.error)
                (dest/(mode+'.svg')).write_text(result.svg,encoding='utf-8')
                # Independently re-render SAVED SVG, not a cached preview.
                pixels=render_svg((dest/(mode+'.svg')).read_text('utf-8'))
                save_rgba(dest/(mode+'.png'),pixels)
                metrics=source_error(source,pixels)
                row[mode]={'metrics':metrics,'bytes':len(result.svg.encode('utf-8')),
                           'ms':result.elapsed_ms,'stats':result.stats}
                print(p.name,mode,round(metrics['active_mae'],4),metrics['premultiplied_max_error'],flush=True)
            except Exception as exc:
                row[mode]={'error':str(exc)};traceback.print_exc()
            row['source_hash_unchanged']=hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            (OUT/'final.json').write_text(json.dumps(records+[row],ensure_ascii=False,indent=2),encoding='utf-8')
        records.append(row)
    return records

if __name__=='__main__':
    if '--final' in sys.argv or '--smooth-only' in sys.argv:
        final_outputs()
    elif '--v1' in sys.argv:
        sys.path.insert(0,str(ROOT/'.tools/vtracer1'))
        candidates()
    elif '--v0' in sys.argv:
        candidates()
    else:
        baseline()
