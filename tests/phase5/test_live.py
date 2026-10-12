from types import SimpleNamespace
import pytest
from contextlib import contextmanager
from faceattend_enrollment.auth import OperatorAuth,password_record,COOKIE
from faceattend_enrollment.errors import EnrollmentError
from .conftest import CONSENT,upload

@contextmanager
def refuses(code):
    with pytest.raises(EnrollmentError) as caught:yield
    assert caught.value.code==code

def operator():
    clock=[0.0];auth=OperatorAuth(password_record('Fictional live operator password'),clock=lambda:clock[0])
    token,csrf=auth.login('Fictional live operator password')
    request=SimpleNamespace(scope={'headers':[]},cookies={COOKIE:token},headers={'x-faceattend-csrf':csrf})
    return auth,request,clock

def test_live_bucket_keeps_enrollment_and_login_limits():
    auth,request,clock=operator()
    for i in range(40):
        clock[0]=i/20;auth.authorize(request,mutation=True,live=True)
    assert len(auth.rates['mutation'])==0
    for _ in range(20):auth.authorize(request,mutation=True)
    with refuses('RATE_LIMITED'):auth.authorize(request,mutation=True)
    assert len(auth.rates['login'])==1
    clock[0]=2;auth.authorize(request,mutation=True,live=True)

def test_live_minute_budget_and_recovery():
    auth,request,clock=operator()
    for i in range(1200):
        clock[0]=i/20;auth.authorize(request,mutation=True,live=True)
    clock[0]=59.999
    with refuses('RATE_LIMITED'):auth.authorize(request,mutation=True,live=True)
    clock[0]=60;auth.authorize(request,mutation=True,live=True)

def test_live_burst_csrf_expiry_and_logout():
    auth,request,clock=operator()
    request.headers={}
    with refuses('CSRF_REQUIRED'):auth.authorize(request,mutation=True,live=True)
    request.headers={'x-faceattend-csrf':auth.session[1]}
    for _ in range(30):auth.authorize(request,mutation=True,live=True)
    with refuses('RATE_LIMITED'):auth.authorize(request,mutation=True,live=True)
    clock[0]=900
    with refuses('OPERATOR_AUTH_REQUIRED'):auth.authorize(request,mutation=True,live=True)
    auth.logout();clock[0]=0
    with refuses('OPERATOR_AUTH_REQUIRED'):auth.authorize(request,mutation=True,live=True)

@pytest.mark.parametrize('endpoint',['preview-face','analyze-face'])
def test_live_route_requires_explicit_optin_auth_csrf_and_ui(ui,endpoint):
    url=f'/api/v1/demo/{endpoint}?live_camera=true'
    def post(headers,data=CONSENT):return ui.client.post(url,headers=headers,data=data,files={'image':upload()})
    assert post(ui.headers).status_code==403
    headers={**ui.headers,'x-faceattend-live-opt-in':'1'}
    assert post({k:v for k,v in headers.items() if k!='x-faceattend-csrf'}).status_code==403
    assert post(headers,data={'adult_consent':'false','research_opt_in':'true'}).status_code==403
    assert ui.context.bundle.detector.calls==0
    from faceattend_ui.integration import UISettings
    ui.app.state.ui=UISettings()
    assert post(headers).status_code==503
    ui.client.cookies.clear();assert post(headers).status_code==401
    assert ui.context.bundle.detector.calls==0

@pytest.mark.parametrize('endpoint',['preview-face','analyze-face'])
def test_more_than_twenty_live_frames_preserve_single_gpu_gate(ui,endpoint):
    start=ui.auth.clock();clock=[start];ui.auth.clock=lambda:clock[0]
    headers={**ui.headers,'x-faceattend-live-opt-in':'1'}
    url=f'/api/v1/demo/{endpoint}?live_camera=true&include_face_box=true'
    for i in range(40):
        clock[0]=start+i/20
        response=ui.client.post(url,headers=headers,data=CONSENT,files={'image':upload()})
        assert response.status_code==200
        assert response.json()['faces'][0]['face_box']['width']==150
        assert response.json()['verified'] is False and response.json()['threshold'] is None
    assert ui.auth.rates['mutation']==[]
    calls=ui.context.bundle.detector.calls
    ui.context.models.gate.acquire()
    try:
        clock[0]+=1
        assert ui.client.post(url,headers=headers,data=CONSENT,files={'image':upload()}).status_code==503
        assert ui.context.bundle.detector.calls==calls
    finally:ui.context.models.gate.release()
