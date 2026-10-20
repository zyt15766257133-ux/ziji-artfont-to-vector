"""Approximate flat-color Gaussian SVG effects as editable vector opacity bands."""
from __future__ import annotations
from copy import deepcopy
import xml.etree.ElementTree as ET
import numpy as np
from .direct import fit_mask_paths
from .native_render import render_svg

NS = 'http://www.w3.org/2000/svg'


def expand_filters(svg: str, levels=16) -> str:
    root = ET.fromstring(svg)
    ET.register_namespace('', NS)
    # Separate our linked source+shadow into a shadow-only layer and the body.
    # Otherwise the solid body is incorrectly quantized as part of the halo.
    filters = {n.get('id'): n for n in root.iter() if n.tag == f'{{{NS}}}filter'}
    for container in list(root.iter()):
        for node in list(container):
            ref = node.get('filter', '')
            definition = filters.get(ref[5:-1]) if ref.startswith('url(#') and ref.endswith(')') else None
            if (node.get('id') != 'lettering-compound' or definition is None
                    or not any(n.tag == f'{{{NS}}}feMergeNode' and n.get('in') == 'SourceGraphic'
                               for n in definition.iter())):
                continue
            shadow_filter = deepcopy(definition)
            shadow_id = definition.get('id')+'-shadow-only'
            shadow_filter.set('id', shadow_id)
            for child in list(shadow_filter):
                if child.tag == f'{{{NS}}}feMerge':
                    shadow_filter.remove(child)
            root.find(f'{{{NS}}}defs').append(shadow_filter)
            shadow, body = deepcopy(node), deepcopy(node)
            shadow.set('filter', f'url(#{shadow_id})')
            shadow.set('id', 'export-shadow-source')
            body.attrib.pop('filter')
            wrapper = ET.Element(f'{{{NS}}}g')
            wrapper.extend([shadow, body])
            index = list(container).index(node)
            container.remove(node)
            container.insert(index, wrapper)
    parent = {child: node for node in root.iter() for child in node}
    filtered = [node for node in root.iter() if node.get('filter')]
    for node in filtered:
        if any(ancestor in filtered for ancestor in _ancestors(node, parent)):
            continue
        # Generated shadow groups carry their own paint attributes.
        single = ET.Element(root.tag, root.attrib)
        for child in root:
            if child.tag.rsplit('}',1)[-1] == 'defs':
                single.append(deepcopy(child))
        detached = deepcopy(node)
        for key in ('fill', 'fill-rule', 'stroke'):
            if key not in detached.attrib:
                for ancestor in _ancestors(node, parent):
                    if key in ancestor.attrib:
                        detached.set(key, ancestor.get(key))
                        break
        single.append(detached)
        pixels = render_svg(ET.tostring(single, encoding='unicode'))
        alpha = pixels[:, :, 3].astype(float)/255
        visible = alpha > .005
        if not visible.any():
            parent[node].remove(node)
            continue
        colors = pixels[:, :, :3][alpha > max(.2, float(alpha.max())*.8)]
        color = np.median(colors,axis=0)
        reconstruction = np.abs(pixels[:, :, :3].astype(float)-color)*alpha[:, :, None]
        if reconstruction[visible].mean()/255 > .01:
            raise ValueError('当前 PDF 滤镜展开仅支持纯色阴影，彩色滤镜请保留 SVG')
        replacement = ET.Element(f'{{{NS}}}g', {'id':node.get('id','effect'), 'data-name':'editable-shadow'})
        previous = 0.
        for level in np.linspace(.02, max(.02,float(alpha.max())), levels):
            mask = alpha >= level
            if mask.sum() < 4:
                continue
            paths, _ = fit_mask_paths(mask, .8)
            opacity = (level-previous)/max(1e-9,1-previous)
            group = ET.SubElement(replacement, f'{{{NS}}}g', {
                'fill':'#%02x%02x%02x'%tuple(color.astype(int)),
                'opacity':f'{opacity:.6f}', 'fill-rule':'evenodd'})
            for path in paths:
                ET.SubElement(group,f'{{{NS}}}path',{'d':path})
            previous=level
        container=parent[node];index=list(container).index(node)
        container.remove(node);container.insert(index,replacement)
    for node in list(root.iter()):
        for child in list(node):
            if child.tag.rsplit('}',1)[-1]=='filter':node.remove(child)
    return ET.tostring(root,encoding='unicode')


def _ancestors(node,parent):
    while node in parent:
        node=parent[node]
        yield node
