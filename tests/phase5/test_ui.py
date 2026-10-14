import pytest
from faceattend_ui.integration import UISettings,CSP
from .conftest import preview,add_reference,CONSENT,upload

def test_default_off_and_invalid_flags(monkeypatch):
    for flag in ('false','TRUE','1','garbage'):
        monkeypatch.setenv('FACEATTEND_FRONTEND_ENABLED',flag);assert not UISettings.from_env().enabled

def test_local_assets_csp_no_store_no_external_resources(ui):
    for path in ('/app/','/app/assets/test.js'):
        result=ui.client.get(path)
        assert result.status_code==200 and result.headers['content-security-policy']==CSP
        assert result.headers['cache-control']=='no-store' and result.headers['referrer-policy']=='no-referrer'
        assert result.headers['permissions-policy']=='camera=(self), microphone=(), geolocation=(), display-capture=()'
    assert ui.client.get('/app/../models/faceattend_best.pt').status_code==404
    assert ui.client.get('/app/.env').status_code==404

@pytest.mark.parametrize('headers',[{'host':'evil.invalid'},{'origin':'https://outside.invalid'}])
def test_static_and_status_keep_local_admission(ui,headers):
    assert ui.client.get('/app/',headers=headers).status_code==403
    assert ui.client.get('/api/v1/demo/ui-status',headers=headers).status_code==403

def test_coarse_status_no_names_ids_secrets_or_vectors(ui):
    add_reference(ui.gallery,name='Fictional private reference')
    ui.client.cookies.clear();response=ui.client.get('/api/v1/demo/ui-status')
    assert response.status_code==200 and response.json()['fixture_mode'] is True
    assert response.json()['anonymous_totals'] is None and not response.json()['telemetry_enabled']
    assert all(word not in response.text for word in ('Fictional private reference','identity_id','csrf_token','key','embedding','template_locator'))

def test_unauthorized_preview_no_processing(ui):
    ui.client.cookies.clear();assert preview(ui).status_code==401
    assert ui.context.bundle.detector.calls==0

def test_default_preview_no_box_or_expression_or_embedding(ui):
    response=preview(ui);assert response.status_code==200
    value=response.json();assert value['face_count']==1 and value['verified'] is False and value['threshold'] is None
    assert 'face_box' not in response.text and 'visible_expression_prediction' not in response.text
    assert ui.context.bundle.recognizer.calls==0

def test_outline_requires_optin_and_coordinates_match_original_frame(ui):
    ui.headers.pop('x-faceattend-box-opt-in')
    assert preview(ui,params={'include_face_box':'true'}).status_code==403
    ui.headers['x-faceattend-box-opt-in']='1'
    value=preview(ui,params={'include_face_box':'true'}).json()['faces'][0]['face_box']
    assert value=={'x':25,'y':25,'width':150,'height':150,'image_width':200,'image_height':200}

def test_combined_outline_optin_preserves_prediction_contract(ui):
    response=ui.client.post('/api/v1/demo/analyze-face?include_face_box=true',headers=ui.headers,data=CONSENT,files={'image':upload()})
    assert response.status_code==200 and response.json()['faces'][0]['face_box']['width']==150
    ui.headers.pop('x-faceattend-box-opt-in')
    assert ui.client.post('/api/v1/demo/analyze-face?include_face_box=true',headers=ui.headers,data=CONSENT,files={'image':upload()}).status_code==403

def test_candidates_without_expression_still_unknown(ui):
    add_reference(ui.gallery,name='Fictional reference')
    response=preview(ui,params={'include_candidates':'true'})
    assert response.status_code==200 and 'visible_expression_prediction' not in response.text
    identity=response.json()['faces'][0]['identity_diagnostic']
    assert identity['candidate_count']==1 and identity['verified'] is False and identity['identity_status']=='Unknown (unverified)'

@pytest.mark.parametrize('mode',['multiple','over_clipped','exception'])
def test_preview_qc_and_bystander_failures(ui,mode):
    ui.context.bundle.detector.mode=mode
    response=preview(ui,params={'include_face_box':'true'})
    assert response.status_code in (422,503) and 'face_box' not in response.text

def test_no_face_no_outline(ui):
    ui.context.bundle.detector.mode='none';response=preview(ui,params={'include_face_box':'true'})
    assert response.json()['face_count']==0 and response.json()['faces']==[]

def test_preview_session_rechecked_after_work(ui,monkeypatch):
    original=ui.context.bundle.detector.detect
    def logout(*args,**kwargs):ui.auth.logout();return original(*args,**kwargs)
    monkeypatch.setattr(ui.context.bundle.detector,'detect',logout)
    assert preview(ui).status_code==401

def test_old_openapi_contract_and_frontend_disabled(ui):
    assert len(ui.app.openapi()['paths'])==6
    assert '/api/v1/demo/ui-status' not in ui.app.openapi()['paths']
    ui.app.state.ui=UISettings()
    assert preview(ui).status_code==503
