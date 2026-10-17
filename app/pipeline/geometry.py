"""Corner-separated cubic fitting for clean, antialiased outline coverage.

Smooth recursive splits share a tangent; true corners keep independent tangents.
The tolerance bounds sample residuals, not a formal continuous Hausdorff distance.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .curvefit import _unit
from .svgutil import mark_corners


def _tangent(p, start):
    q = p if start else p[::-1]
    d = np.linalg.norm(q-q[0], axis=1)
    k = min(len(q)-1, max(1, int(np.searchsorted(d, 2.5))))
    return _unit(q[k]-q[0])


def _fit(p, tol, left=None, right=None, depth=0):
    if len(p) < 2:
        return []
    chord = p[-1]-p[0]
    length = np.linalg.norm(chord)
    if len(p) == 2:
        return [(p[0], None, None, p[-1])]
    left = _tangent(p, True) if left is None else left
    right = _tangent(p, False) if right is None else right
    if length > 1e-8:
        distance = np.abs(chord[0]*(p[:,1]-p[0,1])-chord[1]*(p[:,0]-p[0,0]))/length
        projection = (p-p[0])@chord/(length*length)
        if (distance.max() <= min(.32, tol) and projection.min() >= -1e-6
                and projection.max() <= 1+1e-6
                and left@_unit(chord) > .995 and right@_unit(-chord) > .995):
            return [(p[0], None, None, p[-1])]
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if u[-1] < 1e-9:
        return []
    u /= u[-1]
    best = None
    for _ in range(5):
        b0,b1,b2,b3 = (1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3
        matrix = np.stack([b1[:,None]*left,b2[:,None]*right],axis=2).reshape(-1,2)
        rhs = (p-(b0+b1)[:,None]*p[0]-(b2+b3)[:,None]*p[-1]).reshape(-1)
        handles = np.linalg.lstsq(matrix,rhs,rcond=None)[0]
        if (handles < max(length,1)*1e-4).any() or (handles > max(length,1)*2).any():
            handles[:] = length/3
        c1,c2 = p[0]+handles[0]*left,p[-1]+handles[1]*right
        q = b0[:,None]*p[0]+b1[:,None]*c1+b2[:,None]*c2+b3[:,None]*p[-1]
        error = np.linalg.norm(q-p,axis=1)
        if best is None or error.max() < best[0]:
            best = (float(error.max()),int(np.argmax(error)),c1,c2)
        if best[0] <= tol:
            return [(p[0],best[2],best[3],p[-1])]
        # Newton reparameterization corrects uneven raster boundary sampling.
        qp = 3*((1-u)**2)[:,None]*(c1-p[0])+6*(u*(1-u))[:,None]*(c2-c1)+3*(u*u)[:,None]*(p[-1]-c2)
        qpp = 6*(1-u)[:,None]*(c2-2*c1+p[0])+6*u[:,None]*(p[-1]-2*c2+c1)
        den = np.sum(qp*qp+(q-p)*qpp,axis=1)
        delta = np.divide(np.sum((q-p)*qp,axis=1),den,out=np.zeros_like(u),where=np.abs(den)>1e-10)
        new_u = np.clip(u-delta,0,1)
        if np.any(np.diff(new_u) <= 0):
            break
        u = new_u
    if depth >= 24:
        return [(p[i],None,None,p[i+1]) for i in range(len(p)-1)]
    split = max(1,min(len(p)-2,best[1]))
    tangent = _unit(p[min(len(p)-1,split+2)]-p[max(0,split-2)])
    return (_fit(p[:split+1],tol,left,-tangent,depth+1)
            + _fit(p[split:],tol,tangent,right,depth+1))


def fit_geometric_ring(points, tolerance=.4, corner_angle=62, snap_deg=0):
    p = np.asarray(points,dtype=float).reshape(-1,2)
    if len(p)<3:
        return []
    # Detect corners at a scale larger than the extraction stair step. The
    # original coverage contour, not this coarse polygon, remains the fit target.
    coarse = cv2.approxPolyDP(p.astype(np.float32), .8, True).reshape(-1,2)
    anchors = [int(np.argmin(np.sum((p-v)**2,axis=1)))
               for v in coarse[mark_corners(coarse,corner_angle)]]
    true_corners=set(anchors)
    lines={}
    indexes=[int(np.argmin(np.sum((p-v)**2,axis=1))) for v in coarse]
    for i,a in enumerate(indexes):
        b=indexes[(i+1)%len(indexes)]
        span=p[a:b+1] if b>a else np.vstack([p[a:],p[:b+1]])
        chord=p[b]-p[a];length=np.linalg.norm(chord)
        if length<10:
            continue
        distance=np.abs(chord[0]*(span[:,1]-p[a,1])-chord[1]*(span[:,0]-p[a,0]))/length
        if distance.max()>min(.25,tolerance):
            continue
        lines[a]=b
        anchors.extend([a,b])
        # Axis snapping is bounded by subpixel evidence, not merely by an angle
        # setting that could flatten intentional slopes in calligraphy.
        axis=int(np.argmin(np.abs(chord)))
        if snap_deg>0 and abs(chord[axis])<=min(.3,length*np.sin(np.deg2rad(snap_deg))):
            value=(p[a,axis]+p[b,axis])/2
            if np.max(np.abs(span[:,axis]-value))<=.25:
                p[a,axis]=p[b,axis]=value
    if len(set(anchors))<2:
        a=anchors[0] if anchors else int(np.argmin(p[:,0]))
        b=int(np.argmax(np.sum((p-p[a])**2,axis=1)))
        anchors=[a,b]
    anchors=sorted(set(anchors))
    result=[]
    for i,a in enumerate(anchors):
        b=anchors[(i+1)%len(anchors)]
        if lines.get(a)==b:
            result.append((p[a],None,None,p[b]))
            continue
        span=p[a:b+1] if b>a else np.vstack([p[a:],p[:b+1]])
        # Uniform arclength sampling prevents long horizontal pixel runs from
        # receiving less fitting weight than densely sampled diagonal runs.
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(span,axis=0),axis=1))]
        samples=np.linspace(0,arc[-1],max(2,int(arc[-1]*2)+1))
        dense=np.column_stack([np.interp(samples,arc,span[:,j]) for j in range(2)])
        padded=np.vstack([2*dense[0]-dense[1:5][::-1],dense,2*dense[-1]-dense[-5:-1][::-1]])
        offset=min(4,len(dense)-1)
        smoothed=gaussian_filter1d(padded,1.1,axis=0)[offset:offset+len(dense)]
        smoothed[0],smoothed[-1]=dense[0],dense[-1]
        span=smoothed
        left=None if a in true_corners else _unit(p[(a+2)%len(p)]-p[(a-2)%len(p)])
        right=None if b in true_corners else _unit(p[(b-2)%len(p)]-p[(b+2)%len(p)])
        incoming=next((start for start,end in lines.items() if end==a),None)
        if a not in true_corners and incoming is not None:
            left=_unit(p[a]-p[incoming])
        if b not in true_corners and b in lines:
            right=_unit(p[b]-p[lines[b]])
        result.extend(_fit(span,tolerance,left,right))
    return result
