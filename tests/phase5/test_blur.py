import io
import pytest
from PIL import Image
from .conftest import CONSENT,add_reference,upload

def flat(value=127):
    stream=io.BytesIO();Image.new('RGB',(200,200),(value,value,value)).save(stream,format='PNG')
    return ('fictional-blurry-frame',stream.getvalue(),'image/png')

def analyze(ui,photo=None,query='allow_blurry_expression=true'):
    return ui.client.post('/api/v1/demo/analyze-face?'+query,headers=ui.headers,data=CONSENT,files={'image':photo or flat()})

def test_blur_relaxation_requires_explicit_header_before_detection(ui):
    assert analyze(ui).status_code==403
    assert ui.context.bundle.detector.calls==0

def test_blurry_expression_succeeds_warns_and_never_matches_identity(ui):
    add_reference(ui.gallery,name='Fictional quality reference')
    ui.headers['x-faceattend-blur-opt-in']='1'
    response=analyze(ui,query='allow_blurry_expression=true&include_candidates=true&include_face_box=true')
    assert response.status_code==200
    face=response.json()['faces'][0]
    assert face['input_quality']['status']=='BLUR_GATE_BYPASSED' and 'unreliable' in face['input_quality']['warning']
    assert face['visible_expression_prediction']['score_calibrated'] is False
    assert face['identity_diagnostic']['possible_candidates']==[] and face['identity_diagnostic']['verified'] is False
    assert face['identity_diagnostic']['reason']=='BLURRY_FRAME_IDENTITY_MATCHING_SKIPPED'
    assert 'face_box' in face and 'display_name' not in response.text
    assert ui.context.bundle.recognizer.calls==0 and ui.context.bundle.detector.calls==1
    assert ui.context.bundle.config.min_blur_variance==60

def test_original_expression_and_preview_blur_gate_stays_strict(ui):
    ui.headers['x-faceattend-blur-opt-in']='1'
    assert analyze(ui,query='').status_code==422
    result=ui.client.post('/api/v1/demo/preview-face',headers=ui.headers,data=CONSENT,files={'image':flat()})
    assert result.status_code==422

@pytest.mark.parametrize('brightness,status',[(0,400),(255,422)])
def test_blur_override_does_not_allow_black_or_overexposed_frames(ui,brightness,status):
    ui.headers['x-faceattend-blur-opt-in']='1'
    response=analyze(ui,flat(brightness))
    assert response.status_code==status and 'visible_expression_prediction' not in response.text

@pytest.mark.parametrize('mode',['multiple','over_clipped'])
def test_blur_override_does_not_allow_bystanders_or_invalid_geometry(ui,mode):
    ui.headers['x-faceattend-blur-opt-in']='1';ui.context.bundle.detector.mode=mode
    assert analyze(ui).status_code==422

def test_clear_expression_override_keeps_candidate_matching_and_no_blur_warning(ui):
    add_reference(ui.gallery,name='Fictional quality reference');ui.headers['x-faceattend-blur-opt-in']='1'
    response=analyze(ui,upload(),query='allow_blurry_expression=true&include_candidates=true')
    assert response.status_code==200
    face=response.json()['faces'][0]
    assert face['identity_diagnostic']['candidate_count']==1 and 'input_quality' not in face
    assert ui.context.bundle.recognizer.calls==1
