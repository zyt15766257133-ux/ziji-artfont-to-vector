"""Development probe; operates only on the saved baseline source."""
from pathlib import Path
import sys, json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import cv2
from scipy.optimize import least_squares
from PIL import Image
from app.pipeline.direct import fit_mask_paths
from app.pipeline.svgutil import SvgDocument, SvgLayer
from app.pipeline.native_render import render_svg

out = Path(__file__).resolve().parent/'results/shadow_quality'
rgb = np.asarray(Image.open(out/'source.png').convert('RGB')).astype(float)
density = (255-rgb.mean(2))/239
body = (density > .65).astype(float)
outside = cv2.dilate(body.astype(np.uint8), np.ones((5,5),np.uint8)) == 0
yy,xx = np.where(outside & (density > .015))
yy,xx = yy[::3],xx[::3]
def predict(p, mask=body):
    dx,dy,sigma,opacity=p
    blurred=cv2.GaussianBlur(mask,(0,0),sigma)
    return cv2.warpAffine(blurred,np.float32([[1,0,dx],[0,1,dy]]),(body.shape[1],body.shape[0]))*opacity
fit=least_squares(lambda p: (predict(p)[yy,xx]-density[yy,xx]),[8,8,5,.4],
                  bounds=([-40,-40,1,.05],[40,40,30,.65]),diff_step=.015,max_nfev=50)
print(fit.x, np.abs(fit.fun).mean())
shadow=predict(fit.x)
coverage=np.clip((density-shadow)/(1-shadow),0,1)
coverage[coverage<.03]=0
for tol in [.25,.45,.65,1.0]:
    paths,nodes=fit_mask_paths(np.uint8(coverage*255),tol,geometric=True)
    doc=SvgDocument(body.shape[1],body.shape[0])
    dx,dy,sigma,opacity=fit.x
    doc.defs=[f'<filter id="inferred-shadow" filterUnits="userSpaceOnUse" x="0" y="0" width="{doc.width}" height="{doc.height}" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="{sigma:.3f}"/><feOffset dx="{dx:.3f}" dy="{dy:.3f}"/></filter>']
    doc.layers=[SvgLayer('#101010',paths,label='shadow',opacity=float(opacity),filter_id='inferred-shadow'),SvgLayer('#101010',paths,label='lettering')]
    svg=doc.to_svg()
    (out/f'probe_{tol}.svg').write_text(svg,encoding='utf-8')
    pixels=render_svg(svg)
    Image.fromarray(pixels).save(out/f'probe_{tol}.png')
    alpha=pixels[:,:,3:4]/255
    composite=pixels[:,:,:3]*alpha+255*(1-alpha)
    print(tol,nodes,float(np.abs(composite-rgb).mean()))
