"""Publish retained original-source comparisons and portable SVG deliverables."""
from pathlib import Path
import json, html, shutil, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'eval/results/full_folder_fidelity'
DELIVER=ROOT.parent/'全量验证_2026-10-17'
FONT='C:/Windows/Fonts/msyh.ttc'

def font(size):return ImageFont.truetype(FONT,size)

def main():
    rows=json.loads((OUT/'final.json').read_text('utf-8'))
    count=len(rows)
    assert count and all((ROOT/'samples/images'/row['file']).is_file() for row in rows)
    DELIVER.mkdir(exist_ok=True)
    for name in ('保真曲线','原像素路径','对比图'):(DELIVER/name).mkdir(exist_ok=True)
    cards=[];sheets=[]
    for i,row in enumerate(rows):
        p=ROOT/'samples/images'/row['file'];dest=OUT/p.stem
        source=Image.open(p).convert('RGBA');arr=np.array(source)
        border=np.r_[arr[0,:,:3],arr[-1,:,:3],arr[:,0,:3],arr[:,-1,:3]]
        if np.any(arr[:,:,3]<255):active=arr[:,:,3]>0
        else:active=np.linalg.norm(arr[:,:,:3].astype(float)-np.median(border,axis=0),axis=2)>5
        y,x=np.nonzero(active)
        box=(max(0,int(x.min())-12),max(0,int(y.min())-12),min(source.width,int(x.max())+13),min(source.height,int(y.max())+13))
        images=[source]
        base=dest/'baseline-native.png'
        images.append(Image.open(base).convert('RGBA') if base.exists() else None)
        images.extend(Image.open(dest/(key+'.png')).convert('RGBA') for key in ('fidelity','pixel_exact'))
        names=['原图','旧版（原背景合成）','新版保真曲线' if row['fidelity']['stats']['representation']!='pixel_grid' else '新版：像素路径回退','原像素路径']
        cw,ch=600,260
        sheet=Image.new('RGB',(cw*2,(ch+42)*2+50),'#edf0f5');draw=ImageDraw.Draw(sheet)
        draw.text((12,8),row['file'],font=font(23),fill='#1e293b')
        for j,(im,label) in enumerate(zip(images,names)):
            ox=(j%2)*cw;oy=(j//2)*(ch+42)+50
            draw.text((ox+10,oy+4),label,font=font(19),fill='#475569')
            if im is None:
                draw.text((ox+10,oy+65),'旧版旋转并裁掉背景，无法按原坐标等同比较',font=font(18),fill='#b45309')
                continue
            crop=im.crop(box)
            rgba=np.array(crop);yy,xx=np.indices(rgba.shape[:2]);v=np.where((xx//12+yy//12)%2,232,246).astype('uint8')
            board=Image.fromarray(np.dstack([v,v,v,np.full_like(v,255)]),'RGBA')
            painted=Image.alpha_composite(board,crop).convert('RGB')
            painted.thumbnail((cw-20,ch-12),Image.Resampling.LANCZOS)
            sheet.paste(painted,(ox+(cw-painted.width)//2,oy+38+(ch-38-painted.height)//2))
        sheet.save(dest/'comparison.jpg',quality=94)
        shutil.copyfile(dest/'comparison.jpg',DELIVER/'对比图'/(p.stem+'.jpg'))
        # Detail at 4x is intentionally nearest-neighbour for the two raster
        # previews; SVG files remain available for actual curve inspection.
        if p.stem=='11_3d':
            detail=Image.new('RGB',(1280,690),'white');dd=ImageDraw.Draw(detail)
            detail_box=(300,445,450,520)
            for j,(im,label) in enumerate(zip(images,names)):
                dd.text(((j%2)*640+10,(j//2)*345+10),label,font=font(20),fill='black')
                detail.paste(im.crop(detail_box).convert('RGB').resize((600,300),Image.Resampling.NEAREST),((j%2)*640+10,(j//2)*345+40))
            detail.save(DELIVER/'立体字_4倍细节.png')
        for mode,name in [('fidelity','保真曲线'),('pixel_exact','原像素路径')]:
            shutil.copyfile(dest/(mode+'.svg'),DELIVER/name/(p.stem+'.svg'))
        f=row['fidelity']['metrics'];e=row['pixel_exact']['metrics'];old=row.get('baseline_native',{}).get('active_mae')
        old_label=f'{old:.3f}' if old is not None else '不可同坐标比较'
        cards.append(f'''<article id="{p.stem}"><h2>{i+1:02d} · {html.escape(row['file'])}</h2>
<p>旧版主体误差 {old_label} → 新版 {f['active_mae']:.3f}/255 · 最差像素 {f['max_error']:.2f}/255 · 差异&gt;8 的像素 {f['pixels_over_8']:,} · 像素路径预乘 RGBA 最大误差 {e['premultiplied_max_error']}</p>
<img loading="lazy" src="{p.stem}/comparison.jpg" alt="原图、旧版、新版和原像素路径对比">
<p><a href="{p.stem}/fidelity.svg" target="_blank">打开新版 SVG</a> · <a download="{p.stem}.svg" href="{p.stem}/fidelity.svg">下载新版 SVG</a> · <a download="{p.stem}.svg" href="{p.stem}/pixel_exact.svg">下载原像素路径</a></p>
<p class="note">表示：{row['fidelity']['stats']['representation']}；新版 {row['fidelity']['bytes']/1048576:.2f} MB，像素路径 {row['pixel_exact']['bytes']/1048576:.2f} MB。原图 SHA-256 验证未修改。</p></article>''')
        sheets.append(sheet)
    for start in range(0,count,5):
        page=Image.new('RGB',(1200,654*min(5,count-start)),'white')
        for j,s in enumerate(sheets[start:start+5]):page.paste(s,(0,j*654))
        page.save(DELIVER/f'总览_{start+1:02d}-{min(start+5,count):02d}.jpg',quality=91)
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>25 张逐图保真验证</title>
<style>body{max-width:1260px;margin:36px auto;padding:0 22px;background:#f3f5f9;color:#182335;font:16px/1.7 "Microsoft YaHei",sans-serif}header,article{background:white;padding:24px;border-radius:14px;margin:20px 0;border:1px solid #dbe1e9}h1{font-size:29px}h2{font-size:20px}img{max-width:100%;height:auto;border-radius:8px}a{color:#2459b7}.note{color:#64748b;font-size:14px}.warn{padding:16px;background:#fff3d9;border-left:4px solid #e7ab2b}</style>
<header><h1>25 张图片 · 原图对照验证</h1><p>2026-10-17 · 25 张旧管线 + 125 个算法配置结果 + 50 份最终 SVG。所有原图 SHA-256 保持不变。</p>
<p class="warn"><b>尚未实现“平滑曲线、少节点、任意放大完全无失真”。</b>新版曲线保留更多细节，仍有局部误差。原像素路径在原始分辨率的 8 位预乘 RGBA 回渲中 25/25 完全一致，但放大仍保留像素网格。透明样本直通 RGBA 合成会有不足 1/255 的整数舍入差异。</p>
<p>保真模式保留整个原画布、原背景和透明度，不做去背景、校斜、去噪或节点压缩。2026-10-18 已移除复杂背景样本；保留的历史合成图仅供诊断，未通过纯白底业务素材准入验收。</p>
<p>黑底和白底逐像素误差取较大值；主体均值剔除大面积空白的稀释作用，同时列出最差像素和显著差异数量。原图无独立矢量真值，因此这些数值不能证明恢复了原设计的曲线。</p>
<p><a href="final.json">完整原始指标 JSON</a> · <a href="research.md">方法调研与运行边界</a></p></header>'''+''.join(cards)+'</html>'
    document=document.replace('25 张',f'{count} 张').replace('25/25',f'{count}/{count}').replace('50 份最终 SVG',f'{count*2} 份最终 SVG').replace('125 个算法配置结果','历史算法配置结果')
    (OUT/'report.html').write_text(document,encoding='utf-8')
    portable=document
    for row in rows:
        stem=Path(row['file']).stem
        portable=portable.replace(stem+'/comparison.jpg','对比图/'+stem+'.jpg')
        portable=portable.replace(stem+'/fidelity.svg','保真曲线/'+stem+'.svg')
        portable=portable.replace(stem+'/pixel_exact.svg','原像素路径/'+stem+'.svg')
    (DELIVER/'逐图对比.html').write_text(portable,encoding='utf-8')
    shutil.copyfile(OUT/'final.json',DELIVER/'final.json')
    if (OUT/'research.md').exists():shutil.copyfile(OUT/'research.md',DELIVER/'research.md')
    with zipfile.ZipFile(DELIVER/f'全部{count}张_SVG与报告.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in DELIVER.rglob('*'):
            if p.is_file() and p.suffix!='.zip':archive.write(p,p.relative_to(DELIVER))
    print('published',DELIVER,'SVG',len(list(DELIVER.rglob('*.svg'))))

if __name__=='__main__':main()
