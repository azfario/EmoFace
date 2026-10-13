"""Password-free local workspace: real routes with fictional images/storage."""
from types import SimpleNamespace
import pytest
from faceattend_enrollment.auth import LocalAccess,COOKIE
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.errors import EnrollmentError
from tests.phase4b.conftest import fields,files
from .conftest import CONSENT,upload

BASE='/api/v1/demo'

@pytest.fixture
def local(ui):
    ui.auth=LocalAccess();ui.app.state.enrollment.auth=ui.auth
    ui.client.cookies.clear()
    result=ui.client.post(BASE+'/session',headers={'x-faceattend-login':'1'},json={})
    assert result.status_code==200
    ui.headers={**ui.headers,'x-faceattend-csrf':result.json()['csrf_token']}
    return ui

def test_local_workspace_enrollment_lifecycle_without_password(local):
    assert local.client.get(BASE+'/ui-status').json()['login_required'] is False
    result=local.client.post(BASE+'/enrollments',headers=local.headers,data=fields(),files=files())
    assert result.status_code==200
    identity=result.json()['identity_id']
    assert local.client.get(BASE+'/enrollments').json()['enrollments'][0]['identity_id']==identity
    assert local.client.get(BASE+'/enrollments/'+identity).status_code==200
    assert local.client.post(BASE+'/enrollments/'+identity+'/reenroll',headers=local.headers,data=fields(),files=files()).status_code==200
    assert local.client.post(BASE+'/enrollments/'+identity+'/revoke',headers=local.headers).status_code==200
    assert local.client.delete(BASE+'/enrollments/'+identity,headers=local.headers).status_code==200
    assert local.client.get(BASE+'/enrollments').json()['enrollments']==[]

@pytest.mark.parametrize('endpoint',['preview-face','analyze-face'])
def test_local_live_analysis_keeps_consent_csrf_and_gpu_gate(local,endpoint):
    url=BASE+'/'+endpoint+'?live_camera=true&include_face_box=true'
    headers={**local.headers,'x-faceattend-live-opt-in':'1'}
    def post(h=headers,data=CONSENT):return local.client.post(url,headers=h,data=data,files={'image':upload()})
    assert post(h={k:v for k,v in headers.items() if k!='x-faceattend-csrf'}).status_code==403
    assert post(data={'adult_consent':'false','research_opt_in':'true'}).status_code==403
    assert local.context.bundle.detector.calls==0
    result=post();assert result.status_code==200 and result.json()['faces'][0]['face_box']
    local.context.models.gate.acquire()
    try:assert post().status_code==503
    finally:local.context.models.gate.release()

@pytest.mark.parametrize('headers',[{}, {'x-faceattend-login':'1','origin':'https://outside.invalid'}, {'x-faceattend-login':'1','host':'outside.invalid'}])
def test_local_session_cannot_be_opened_by_cross_origin_forms(ui,headers):
    ui.app.state.enrollment.auth=LocalAccess();ui.client.cookies.clear()
    assert ui.client.post(BASE+'/session',headers=headers,json={}).status_code==403

def test_local_session_rejects_password_and_extra_fields(local):
    for value in ({'password':'PRIVATE_SENTINEL'},{'extra':True}):
        response=local.client.post(BASE+'/session',headers={'x-faceattend-login':'1'},json=value)
        assert response.status_code==422 and 'PRIVATE_SENTINEL' not in response.text

def test_tabs_sharing_a_cookie_keep_their_csrf_token(local):
    token=local.client.cookies.get(COOKIE)
    response=local.client.post(BASE+'/session',headers={'x-faceattend-login':'1'},json={})
    assert response.status_code==200 and response.json()['csrf_token']==local.headers['x-faceattend-csrf']
    assert local.client.cookies.get(COOKIE)==token
    assert local.client.delete(BASE+'/session',headers=local.headers).status_code==200
    assert local.client.get(BASE+'/enrollments').status_code==401

def test_local_session_cannot_skip_cookie_or_csrf(local):
    assert local.client.post(BASE+'/enrollments',data=fields(),files=files()).status_code==403
    local.client.cookies.clear()
    assert local.client.get(BASE+'/enrollments').status_code==401
    assert local.client.post(BASE+'/enrollments',headers=local.headers,data=fields(),files=files()).status_code==401
    assert local.context.bundle.detector.calls==0

def test_local_sessions_expire_logout_independently_and_bound_rates():
    now=[0.0];auth=LocalAccess(clock=lambda:now[0])
    def open_request():
        token,csrf=auth.open_session()
        return SimpleNamespace(scope={'headers':[]},cookies={COOKIE:token},headers={'x-faceattend-csrf':csrf})
    first=open_request();second=open_request();auth.logout(first)
    with pytest.raises(EnrollmentError):auth.authorize(first)
    auth.authorize(second)
    for _ in range(30):auth.authorize(second,mutation=True,live=True)
    with pytest.raises(EnrollmentError) as caught:auth.authorize(second,mutation=True,live=True)
    assert caught.value.code=='RATE_LIMITED'
    for _ in range(20):auth.authorize(second,mutation=True)
    with pytest.raises(EnrollmentError):auth.authorize(second,mutation=True)
    for _ in range(3):open_request()
    with pytest.raises(EnrollmentError):open_request()
    now[0]=900
    with pytest.raises(EnrollmentError):auth.authorize(second)
    third=open_request();auth.authorize(third);auth.logout()
    with pytest.raises(EnrollmentError):auth.authorize(third)

def test_env_defaults_to_password_free_but_optional_auth_is_explicit(private_root,monkeypatch):
    gallery=private_root/'gallery';gallery.mkdir()
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_ENABLED','true')
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_DATABASE_URL','postgresql+psycopg://fictional:fixture@127.0.0.1/test')
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_GALLERY_DIR',str(gallery))
    monkeypatch.delenv('FACEATTEND_OPERATOR_AUTH_FILE',raising=False)
    monkeypatch.delenv('FACEATTEND_OPERATOR_LOGIN_REQUIRED',raising=False)
    assert EnrollmentSettings.from_env().login_required is False
    monkeypatch.setenv('FACEATTEND_OPERATOR_LOGIN_REQUIRED','true')
    with pytest.raises(EnrollmentError):EnrollmentSettings.from_env()
    monkeypatch.setenv('FACEATTEND_OPERATOR_LOGIN_REQUIRED','unexpected')
    with pytest.raises(EnrollmentError):EnrollmentSettings.from_env()
