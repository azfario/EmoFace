import pytest
from faceattend_analysis.config import AnalysisSettings
from .conftest import query,add_reference,CONSENT,upload

def test_default_off_and_invalid_enable_fail_closed(monkeypatch):
    for value in ('false','TRUE','1','invalid'):
        monkeypatch.setenv('FACEATTEND_FACE_ANALYSIS_ENABLED',value)
        assert not AnalysisSettings.from_env().enabled
    monkeypatch.setenv('FACEATTEND_FACE_ANALYSIS_ENABLED','true');assert AnalysisSettings.from_env().enabled

def test_unauthorized_no_body_inference_or_name(operator):
    operator.client.cookies.clear();response=query(operator,params={'include_candidates':'true'})
    assert response.status_code==401 and operator.context.bundle.detector.calls==0
    assert 'faces' not in response.json()

@pytest.mark.parametrize('header',['x-faceattend-csrf','x-faceattend-expression-opt-in'])
def test_required_csrf_expression_optin_before_models(operator,header):
    operator.headers.pop(header);assert query(operator).status_code==403
    assert operator.expression.calls==0 and operator.context.bundle.detector.calls==0

@pytest.mark.parametrize('field',['adult_consent','research_opt_in'])
def test_optout_no_classification(operator,field):
    response=operator.client.post('/api/v1/demo/analyze-face',headers=operator.headers,
        data=dict(CONSENT,**{field:'false'}),files={'image':upload()})
    assert response.status_code==403 and operator.expression.calls==0

def test_candidate_names_require_additional_optin(operator):
    operator.headers.pop('x-faceattend-candidate-opt-in')
    assert query(operator,params={'include_candidates':'true'}).status_code==403
    assert operator.expression.calls==0

def test_anonymous_expression_works_without_gallery_and_no_names(operator,monkeypatch):
    operator.app.state.enrollment.service=None
    response=query(operator);assert response.status_code==200
    value=response.json()['faces'][0]
    assert value['identity_diagnostic']['possible_candidates']==[]
    assert value['visible_expression_prediction']['not_a_mood_or_diagnosis']
    assert 'identity_id' not in response.text and 'display_name' not in response.text

def test_no_names_vectors_labels_in_logs_or_persistence(operator,caplog):
    add_reference(operator.gallery,name='Fictional protected reference')
    before=dict(operator.gallery.repository.rows)
    response=query(operator,params={'include_candidates':'true'});assert response.status_code==200
    assert 'Fictional protected reference' in response.text
    assert 'embedding' not in response.text and 'template_locator' not in response.text
    assert 'Fictional protected reference' not in caplog.text and 'VISIBLE_EXPRESSION_PREDICTED' not in caplog.text
    assert operator.gallery.repository.rows==before
    assert response.headers['cache-control']=='no-store'
    assert '/api/v1/demo/analyze-face' not in operator.app.openapi()['paths']

@pytest.mark.parametrize('mode',['multiple','clipped','exception'])
def test_invalid_face_no_expression_names(operator,mode):
    operator.context.bundle.detector.mode=mode
    response=query(operator,params={'include_candidates':'true'})
    assert response.status_code in (422,503) and 'visible_expression_prediction' not in response.text
    assert operator.expression.calls==0

def test_model_gate_no_overlapping_analysis(operator):
    operator.context.models.gate.acquire()
    try:assert query(operator).status_code==503
    finally:operator.context.models.gate.release()

def test_disable_setting_refuses(operator):
    operator.app.state.face_analysis=AnalysisSettings()
    assert query(operator).status_code==503 and operator.expression.calls==0

def test_expression_dependency_unready(operator):
    operator.context.models.expression_service=None
    assert query(operator).status_code==503

@pytest.mark.parametrize('count',['0','6','bad','1.1'])
def test_invalid_top_k_no_processing(operator,count):
    assert query(operator,params={'top_k':count}).status_code==422 and operator.expression.calls==0

def test_session_revoked_before_response_hides_results(operator,monkeypatch):
    original=operator.expression.predict
    def logout(rgb):operator.auth.logout();return original(rgb)
    monkeypatch.setattr(operator.expression,'predict',logout)
    response=query(operator)
    assert response.status_code==401 and 'visible_expression_prediction' not in response.text

def test_cross_origin_refused(operator):
    operator.headers['origin']='https://outside.invalid'
    assert query(operator).status_code==403 and operator.expression.calls==0

def test_original_full_image_expression_endpoint_retains_contract(operator):
    response=operator.client.post('/api/v1/expression/predict',data=CONSENT,files={'image':upload()})
    assert response.status_code==200 and response.json()['confidence']==0.91
    assert 'crop_version' not in response.text and operator.expression.calls==0

def test_gallery_read_failure_returns_no_partial_expression_or_names(operator):
    add_reference(operator.gallery,name='Fictional private reference');operator.gallery.files.key=None
    response=query(operator,params={'include_candidates':'true'})
    assert response.status_code==503
    assert 'visible_expression_prediction' not in response.text and 'Fictional private reference' not in response.text

def test_timed_out_worker_cleans_up_before_gate_release(operator,monkeypatch):
    import time
    from dataclasses import replace
    operator.app.state.settings=replace(operator.context.settings,inference_timeout_seconds=0.005)
    original=operator.expression.predict
    def slow(rgb):time.sleep(0.03);return original(rgb)
    monkeypatch.setattr(operator.expression,'predict',slow)
    response=query(operator)
    assert response.status_code==504 and response.json()['status']=='INFERENCE_TIMEOUT'
    assert not operator.context.models.gate.locked() and operator.expression.calls==1
    assert 'visible_expression_prediction' not in response.text
