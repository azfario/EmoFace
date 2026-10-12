import asyncio,threading,io,json
from dataclasses import replace
from uuid import UUID,uuid4
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import pytest
from faceattend_enrollment.auth import COOKIE
from .conftest import fields,files,enroll_http,PASSWORD

BASE='/api/v1/demo/enrollments'

@pytest.mark.parametrize('method,path',[('get',BASE),('post',BASE),('get',BASE+'/'+str(uuid4())),
 ('post',BASE+'/'+str(uuid4())+'/revoke'),('post',BASE+'/'+str(uuid4())+'/reenroll'),('delete',BASE+'/'+str(uuid4()))])
def test_unauthorized_management_never_processes_uploads(operator,method,path):
    operator.client.cookies.clear()
    response=getattr(operator.client,method)(path)
    assert response.status_code==401 and not operator.gallery.repository.rows
    assert operator.context.bundle.detector.calls==0

def test_login_cookie_attributes_csrf_logout_and_safe_schema(operator):
    response=operator.client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
    cookie=response.headers['set-cookie']
    assert 'HttpOnly' in cookie and 'SameSite=strict' in cookie and 'Max-Age=900' in cookie
    assert COOKIE in operator.client.cookies
    assert len(operator.client.get('/openapi.json').json()['paths'])==6
    assert operator.client.post(BASE,data=fields(),files=files()).status_code==403
    csrf={'x-faceattend-csrf':response.json()['csrf_token']}
    assert operator.client.delete('/api/v1/demo/session',headers=csrf).status_code==200
    assert operator.client.get(BASE).status_code==401

@pytest.mark.parametrize('kind',['header_missing','wrong_password','too_large','extra','bad_json'])
def test_login_errors_do_not_echo_secrets(operator,kind,caplog):
    header={'x-faceattend-login':'1'}
    if kind=='header_missing': result=operator.client.post('/api/v1/demo/session',json={'password':PASSWORD})
    elif kind=='wrong_password': result=operator.client.post('/api/v1/demo/session',headers=header,json={'password':'PRIVATE_SECRET'})
    elif kind=='too_large': result=operator.client.post('/api/v1/demo/session',headers=header,json={'password':'PRIVATE_SECRET'*500})
    elif kind=='extra': result=operator.client.post('/api/v1/demo/session',headers=header,json={'password':PASSWORD,'private':'PRIVATE_SECRET'})
    else: result=operator.client.post('/api/v1/demo/session',headers=dict(header,**{'content-type':'application/json'}),content=b'PRIVATE_SECRET not json')
    assert result.status_code>=400 and 'PRIVATE_SECRET' not in result.text+caplog.text and PASSWORD not in result.text

def test_enrollment_list_status_revoke_delete_and_no_sensitive_output(operator,caplog):
    result=enroll_http(operator)
    assert result.status_code==200
    value=result.json();identity=value['identity_id']
    assert value['status']=='active' and value['verified'] is False and value['threshold'] is None
    assert 'template_locator' not in value and 'embedding' not in value and 'sample_hashes' not in value
    assert operator.client.get(BASE).json()['enrollments'][0]==value
    assert operator.client.get(BASE+'/'+identity).json()==value
    assert operator.client.post(BASE+'/'+identity+'/inspect-reference',headers=operator.csrf).status_code==200
    assert operator.client.post(BASE+'/'+identity+'/revoke',headers=operator.csrf).json()['status']=='revoked'
    assert operator.client.post(BASE+'/'+identity+'/inspect-reference',headers=operator.csrf).status_code==422
    assert operator.client.delete(BASE+'/'+identity,headers=operator.csrf).json()=={'status':'deleted'}
    assert operator.client.get(BASE).json()['enrollments']==[]
    assert 'Fictional Adult' not in caplog.text

@pytest.mark.parametrize('field',['adult_consent','research_opt_in','enrollment_consent'])
def test_refused_consent_does_not_enroll(operator,field):
    response=enroll_http(operator,**{field:'false'})
    assert response.status_code==403 and operator.context.bundle.detector.calls==0
    assert not operator.gallery.repository.rows

@pytest.mark.parametrize('n',[0,1,2,6])
def test_reference_count_bounds(operator,n):
    response=operator.client.post(BASE,headers=operator.csrf,data=fields(),files=files(n))
    assert response.status_code>=400 and operator.context.bundle.detector.calls==0

@pytest.mark.parametrize('kind',['duplicate','corrupt','black','extra_field','invalid_retention','invalid_name'])
def test_invalid_enrollment_uploads(operator,kind):
    data=fields();images=files()
    if kind=='duplicate': images=[images[0]]*3
    elif kind=='corrupt': images[0]=('enrollment_images',('private',b'bad','image/png'))
    elif kind=='black':
        stream=io.BytesIO();Image.new('RGB',(200,200)).save(stream,format='PNG')
        images[0]=('enrollment_images',('private',stream.getvalue(),'image/png'))
    elif kind=='extra_field': data['private']='PRIVATE_SECRET'
    elif kind=='invalid_retention': data['retention_days']='31'
    else: data['display_name']='PRIVATE_SECRET/unsafe'
    response=operator.client.post(BASE,headers=operator.csrf,data=data,files=images)
    assert response.status_code>=400 and 'PRIVATE_SECRET' not in response.text
    assert not operator.gallery.repository.rows and not list(operator.gallery.files.staging.iterdir())

def test_non_cuda_model_rejected_before_template_or_metadata(operator):
    operator.context.bundle.detector.session.providers=['CPUExecutionProvider']
    response=enroll_http(operator)
    assert response.status_code==503 and response.json()['status']=='CUDA_UNAVAILABLE'
    assert not operator.gallery.repository.rows

def test_same_origin_policy_applies_to_new_admin_routes(operator):
    assert operator.client.get(BASE,headers={'origin':'https://evil.example'}).status_code==403
    assert operator.client.post('/api/v1/demo/session',headers={'origin':'null','x-faceattend-login':'1'},json={'password':PASSWORD}).status_code==403

def test_pagination_is_bounded_and_non_identifying(operator):
    assert operator.client.get(BASE+'?page=5').status_code==422
    assert operator.client.get(BASE+'?page=0').status_code==422

def test_explicit_reenrollment_same_uuid_and_no_old_ciphertext(operator):
    response=enroll_http(operator);identity=response.json()['identity_id']
    old=operator.gallery.repository.get(UUID(identity)).template_locator
    result=operator.client.post(BASE+'/'+identity+'/reenroll',headers=operator.csrf,data=fields(),files=files())
    assert result.status_code==200 and result.json()['identity_id']==identity
    assert not operator.gallery.files.final(old).exists()

def test_original_inference_gate_is_shared(operator):
    operator.context.models.gate.acquire()
    try:
        response=enroll_http(operator)
        assert response.status_code==503 and response.json()['status']=='ENROLLMENT_BUSY'
    finally: operator.context.models.gate.release()

def test_cancel_waits_for_native_enrollment_and_releases_gate(operator,monkeypatch):
    import httpx
    entered,release=threading.Event(),threading.Event()
    original=operator.gallery.service.enroll
    def delayed(*args,**kwargs):
        entered.set();assert release.wait(5);return original(*args,**kwargs)
    monkeypatch.setattr(operator.gallery.service,'enroll',delayed)
    cookie=operator.client.cookies.get(COOKIE)
    async def run():
        transport=httpx.ASGITransport(app=operator.app,client=('127.0.0.1',1))
        async with httpx.AsyncClient(transport=transport,base_url='http://127.0.0.1',cookies={COOKIE:cookie}) as client:
            task=asyncio.create_task(client.post(BASE,headers=operator.csrf,data=fields(),files=files()))
            for _ in range(100):
                if entered.is_set():break
                await asyncio.sleep(.01)
            assert entered.is_set();task.cancel();await asyncio.sleep(.02)
            try: assert operator.context.models.gate.locked()
            finally:release.set()
            with pytest.raises(asyncio.CancelledError):await task
            assert not operator.context.models.gate.locked()
            # Cancellation is not a claim that an already-started operation was undone.
            assert next(iter(operator.gallery.repository.rows.values())).status=='active'
    asyncio.run(run())
