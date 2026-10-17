"""Actual source pixels, alpha and API semantics, independent of silhouette masks."""
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.fidelity import pixel_paths, source_error
from app.pipeline.native_render import render_svg
from app.pipeline.service import convert_bytes
from app.pipeline.vectorize import VectorizeOptions, vectorize_bytes


def test_exact_cells_keep_isolated_pixels_holes_and_random_alpha():
    rng=np.random.default_rng(54)
    source=rng.integers(0,256,(31,43,4),dtype=np.uint8)
    source[3:17,8:24]=[17,203,54,255]
    source[7:12,11:19]=0
    snapshot=source.copy()
    svg,stats=pixel_paths(source)
    output=render_svg(svg)
    np.testing.assert_array_equal(source,snapshot)
    metric=source_error(source,output)
    assert metric['premultiplied_max_error']<=1,metric
    assert metric['pixels_over_8']==0
    tags={e.tag.rsplit('}',1)[-1] for e in ET.fromstring(svg).iter()}
    assert tags<={'svg','title','desc','g','path'}
    assert 'data:image' not in svg
    assert stats['representation']=='pixel_grid'


def test_flat_opaque_image_is_one_rectangle_and_exact():
    source=np.full((29,57,4),[32,78,143,255],dtype=np.uint8)
    svg,stats=pixel_paths(source)
    assert stats['rectangles']==1
    np.testing.assert_array_equal(render_svg(svg),source)


def test_metric_cannot_hide_opacity_or_local_errors_in_large_blank_canvas():
    source=np.full((300,300,4),255,dtype=np.uint8)
    source[145:155,145:155,:3]=0
    damaged=source.copy();damaged[147:153,147:153,:3]=255
    result=source_error(source,damaged)
    assert result['mae']<1
    assert result['active_mae']>20
    assert result['max_error']==255
    invisible=source.copy();invisible[:,:,3]=0
    assert source_error(source,invisible)['max_error']==255


def test_service_and_byte_entry_preserve_source_canvas_and_background():
    source=np.full((37,49,4),[77,55,133,255],dtype=np.uint8)
    source[9:12,19:22]=[0,0,0,255]
    buf=io.BytesIO();Image.fromarray(source).save(buf,format='PNG')
    opts=VectorizeOptions.from_preset('pixel_exact')
    result=convert_bytes('sample.png',buf.getvalue(),opts,False,False)
    assert not result.error
    assert result.size=={'width':49,'height':37}
    assert result.metrics['source_transform']['crop_box'] is None
    assert result.metrics['source_fidelity']['native_grid_exact']
    other=vectorize_bytes(buf.getvalue(),opts)
    np.testing.assert_array_equal(render_svg(other['svg']),source)


def test_api_curve_preset_keeps_3d_detail_and_reports_approximation():
    p=Path(__file__).resolve().parents[1]/'samples/images/11_3d.png'
    with TestClient(app) as client:
        res=client.post('/api/convert',files={'file':(p.name,p.read_bytes(),'image/png')},
                        data={'preset':'fidelity','with_preview':'false'})
    assert res.status_code==200,res.text
    result=res.json()
    assert result['size']=={'width':1000,'height':1000}
    assert result['stats']['representation']=='approximate_curves'
    assert result['stats']['node_budget_applied'] is False
    assert result['metrics']['source_fidelity']['active_mae']<2
    assert result['metrics']['source_fidelity']['pixels_over_8']<5000
    assert result['metrics']['source_fidelity']['native_grid_exact'] is False
    root=ET.fromstring(result['svg'])
    assert any('C' in e.get('d','') for e in root.iter())
    assert not any(e.tag.endswith('image') for e in root.iter())


def test_smoother_candidate_does_not_replace_brush_texture():
    p=Path(__file__).resolve().parents[1]/'samples/images/14_brush_feibai.png'
    result=convert_bytes(p.name,p.read_bytes(),VectorizeOptions.from_preset('fidelity'),False,False)
    # A smooth but inaccurate watershed candidate erases the fine brush marks.
    assert result.metrics['source_fidelity']['active_mae']<3
    assert result.stats['algorithm']=='color-cluster'
