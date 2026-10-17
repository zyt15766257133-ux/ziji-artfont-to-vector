"""Independent vector references for crisp outlines separated from soft shadows."""
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest
from app.pipeline.native_render import render_svg
from app.pipeline.preprocess import preprocess
from app.pipeline.shadow_model import shadow_document
from app.pipeline.svgutil import SvgDocument, SvgLayer
from app.pipeline.vectorize import VectorizeOptions
from app.pipeline.service import convert_bytes

ROOT = Path(__file__).resolve().parents[1]


def _reference(dx,dy,sigma):
    path='M40 35H180V175H40Z M57 52V158H163V52Z M215 170L245 37L262 45L233 178Z M280 40C365 40 365 170 280 170L280 155C343 155 343 55 280 55Z'
    body=SvgDocument(400,220,layers=[SvgLayer('#181818',[path])])
    full=SvgDocument(400,220,layers=[SvgLayer('#181818',[path],opacity=.42,filter_id='test-shadow'),body.layers[0]])
    full.defs=[f'<filter id="test-shadow" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB"><feGaussianBlur stdDeviation="{sigma}"/><feOffset dx="{dx}" dy="{dy}"/></filter>']
    pixels=render_svg(full.to_svg())
    alpha=pixels[:,:,3:4]/255.
    rgb=np.uint8(np.rint(pixels[:,:,:3]*alpha+255*(1-alpha)))
    return preprocess(rgb[:,:,::-1]),body


@pytest.mark.parametrize('dx,dy,sigma',[(9,7,4),(-8,6,3),(0,-9,5)])
def test_offset_shadow_is_inferred_without_thinning_or_losing_holes(dx,dy,sigma):
    prepared,truth=_reference(dx,dy,sigma)
    result=shadow_document(prepared,VectorizeOptions(glyph_count=3))
    assert result is not None
    doc,stats=result
    assert abs(stats['shadow']['dx']-dx)<.65
    assert abs(stats['shadow']['dy']-dy)<.65
    body=SvgDocument(doc.width,doc.height,layers=[doc.layers[-1]])
    actual=render_svg(body.to_svg())[:,:,3]/255.
    expected=prepared.align_mask(render_svg(truth.to_svg())[:,:,3])/255.
    assert .98 < actual.sum()/expected.sum() < 1.02
    intersection=np.logical_and(actual>.5,expected>.5).sum()
    union=np.logical_or(actual>.5,expected>.5).sum()
    assert intersection/union > .97
    assert stats['node_total']<=1500
    assert '<image' not in doc.to_svg() and '<text' not in doc.to_svg()


def test_shadow_modes_and_unsupported_colours_keep_existing_engine():
    prepared,_=_reference(9,7,4)
    for opts in [VectorizeOptions(shadow_layers='off'),VectorizeOptions(shadow_style='steps'),
                 VectorizeOptions(smooth=False),VectorizeOptions(spot_colors=['#000000'])]:
        assert shadow_document(prepared,opts) is None
    colorful=prepared.rgb_raw.copy();colorful[60:110,60:160]=[220,30,30]
    assert shadow_document(replace(prepared,rgb_raw=colorful),VectorizeOptions()) is None


def test_reported_shadow_sample_uses_crisp_body_and_independent_shadow():
    result=convert_bytes('12_shadow.png',(ROOT/'samples/images/12_shadow.png').read_bytes(),
                         VectorizeOptions(glyph_count=4),strict_input=False,with_preview=True)
    assert result.stats.get('source_shadow_model')
    assert result.stats['body_edge_coverage_mae'] < .065
    assert .99 < result.stats['body_area_ratio'] < 1.01
    assert result.stats['node_total'] < 1000
    root=ET.fromstring(result.svg)
    groups=[n for n in root if n.tag.endswith('}g')]
    assert groups[0].get('filter')
    assert groups[-1].get('filter') is None
    assert result.source_png and result.preview_png
    assert result.metrics['comparison_basis']=='source_background_composite_contrast_20'
    assert result.metrics['shape_iou']>.94


def test_translucent_shadow_metric_uses_same_background_on_both_sides():
    from app.pipeline.metrics import evaluate
    prepared,_=_reference(9,7,4)
    candidate=shadow_document(prepared,VectorizeOptions(glyph_count=3))[0]
    metrics=evaluate(prepared,candidate.to_svg(),glyph_count=3)
    assert metrics['comparison_basis']=='source_background_composite_contrast_20'
    assert metrics['shape_iou']>.96
    assert metrics['alpha_shape_iou']<metrics['shape_iou']-.15
