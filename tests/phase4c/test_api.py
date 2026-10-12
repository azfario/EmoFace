import json
import pytest
from faceattend_candidates.config import CandidateSettings
from .conftest import query,add_reference,upload,CONSENT

def test_disabled_by_default_and_bad_flags_fail_closed(monkeypatch):
    for flag in ('false','TRUE','1','malformed'):
        monkeypatch.setenv('FACEATTEND_CANDIDATES_ENABLED',flag)
        assert not CandidateSettings.from_env().enabled
    monkeypatch.setenv('FACEATTEND_CANDIDATES_ENABLED','true')
    assert CandidateSettings.from_env().enabled

def test_unauthorized_candidate_request_no_image_or_catalog_processing(candidate_operator):
    op=candidate_operator;op.client.cookies.clear()
    response=query(op)
    assert response.status_code==401 and op.context.bundle.detector.calls==0
    assert 'possible_candidates' not in response.json()

@pytest.mark.parametrize('header',['x-faceattend-csrf','x-faceattend-candidate-opt-in'])
def test_csrf_and_explicit_search_opt_in_before_upload(candidate_operator,header):
    op=candidate_operator;op.headers.pop(header)
    response=query(op)
    assert response.status_code==403 and op.context.bundle.detector.calls==0

@pytest.mark.parametrize('field',['adult_consent','research_opt_in'])
def test_probe_consent_refusal_before_inference(candidate_operator,field):
    op=candidate_operator;fields=dict(CONSENT,**{field:'false'})
    result=op.client.post('/api/v1/demo/candidates',headers=op.headers,data=fields,files={'image':upload()})
    assert result.status_code==403 and op.context.bundle.detector.calls==0

@pytest.mark.parametrize('top_k',['0','6','-1','wrong','1.5'])
def test_invalid_candidate_count_sanitized(candidate_operator,top_k):
    response=query(candidate_operator,params={'top_k':top_k})
    assert response.status_code==422 and top_k not in response.text

def test_candidate_names_private_no_vectors_logs_or_schema_change(candidate_operator,caplog):
    op=candidate_operator;add_reference(op.gallery,name='Fictional private display')
    response=query(op);result=response.json()
    assert response.status_code==200 and result['possible_candidates'][0]['display_name']=='Fictional private display'
    assert result['possible_candidates'][0]['label']=='Possible match (unverified)'
    assert response.headers['cache-control']=='no-store'
    assert all(token not in response.text for token in ('embedding','template_locator','sample_hashes',str(op.gallery.root)))
    assert 'Fictional private display' not in caplog.text
    assert '/api/v1/demo/candidates' not in op.app.openapi()['paths']

def test_existing_shared_inference_gate(candidate_operator):
    op=candidate_operator;op.context.models.gate.acquire()
    try:assert query(op).status_code==503
    finally:op.context.models.gate.release()

def test_disabled_feature_no_lookup(candidate_operator):
    op=candidate_operator;op.app.state.candidates=CandidateSettings()
    assert query(op).status_code==503 and op.context.bundle.detector.calls==0

def test_same_origin_local_admission(candidate_operator):
    op=candidate_operator;op.headers['origin']='https://outside.invalid'
    assert query(op).status_code==403 and op.context.bundle.detector.calls==0

def test_unknown_no_face_still_never_verified(candidate_operator):
    op=candidate_operator;op.context.bundle.detector.mode='none'
    result=query(op).json()
    assert result['identity_status']=='Unknown (unverified)' and result['verified'] is False

def test_bystander_image_refused_without_candidate_names(candidate_operator):
    op=candidate_operator;add_reference(op.gallery,name='Fictional protected label')
    op.context.bundle.detector.mode='multiple'
    response=query(op)
    assert response.status_code==422 and 'Fictional protected label' not in response.text

def test_logout_during_inference_cannot_return_names(candidate_operator,monkeypatch):
    op=candidate_operator;add_reference(op.gallery,name='Fictional protected label')
    original=op.context.bundle.recognizer.get_feat
    def logout(image):op.auth.logout();return original(image)
    monkeypatch.setattr(op.context.bundle.recognizer,'get_feat',logout)
    response=query(op)
    assert response.status_code==401 and 'Fictional protected label' not in response.text

def test_request_failure_does_not_log_payload_path(candidate_operator,caplog):
    op=candidate_operator;op.context.bundle.detector.mode='exception'
    response=query(op)
    assert response.status_code==503 and 'SECRET' not in response.text and 'SECRET' not in caplog.text
