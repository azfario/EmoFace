from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import asyncio
import io
import json
import os
import threading
import time
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient
from faceattend import CLASS_NAMES
from faceattend_api.config import Settings
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.errors import APIError
from faceattend_api.main import create_app
from faceattend_api.services.image_input import decode, Part, read_parts
from faceattend_api.services.temporary_compare import request_reference
from face_verification.template_store import TemplateStore
from .conftest import CONSENT, photo, upload, comparison_files

def post(context, path='face/detect', **kwargs):
    return context.client.post('/api/v1/'+path, data=CONSENT, files={'image': upload()}, **kwargs)

def test_health_and_public_model_info(context):
    assert context.client.get('/health').json()['gpu_ready'] is True  # Injected mock readiness.
    info = context.client.get('/api/v1/models/info').json()
    assert info['classifier_labels'] == CLASS_NAMES
    assert info['model_readiness'] == {'face':'ready','expression':'ready'}
    assert 'private' not in json.dumps(info)

@pytest.mark.parametrize('expression_enabled,face,expression,code,gpu', [
    (True,False,False,503,False), (True,True,False,503,False),
    (False,True,False,200,True), (False,False,False,503,False)])
def test_truthful_readiness(context, expression_enabled, face, expression, code, gpu):
    models = ModelDependencies(face_models=context.bundle if face else None, load_on_start=False)
    app = create_app(replace(context.settings, expression_enabled=expression_enabled), models)
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1',1)) as client:
        result = client.get('/health')
        assert result.status_code == code and result.json()['gpu_ready'] is gpu

@pytest.mark.parametrize('path,status', [('face/detect','OK'),('face/embedding-info','EMBEDDING_GENERATED'),('expression/predict','VISIBLE_EXPRESSION_PREDICTED')])
def test_valid_safe_output(context, path, status):
    response = post(context, path)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['status'] == status
    assert not any(k in data for k in ['embedding', 'bbox', 'landmarks', 'aligned', 'claimed_id'])
    assert response.headers['cache-control'] == 'no-store'
    if path.endswith('embedding-info'):
        assert data['dimension'] == 512 and data['l2_norm'] == pytest.approx(1)
    if path.startswith('expression'):
        assert data['label'] == CLASS_NAMES[3] and data['confidence_calibrated'] is False
        assert 'not internal mood' in data['interpretation']

@pytest.mark.parametrize('mode,status', [('none','NO_FACE'),('multiple','MULTIPLE_FACES'),('clipped','LOW_QUALITY_IMAGE')])
@pytest.mark.parametrize('path', ['face/detect','face/embedding-info','expression/predict'])
def test_quality_failures_distinct(context, mode, status, path):
    context.bundle.detector.mode = mode
    response = post(context, path)
    assert response.status_code == 422 and response.json()['status'] == status
    assert context.bundle.recognizer.calls == 0

@pytest.mark.parametrize('field,value', [('adult_consent','false'),('research_opt_in','false'),('adult_consent','True')])
def test_explicit_consent(context, field, value):
    data = dict(CONSENT, **{field:value})
    response = context.client.post('/api/v1/face/detect', data=data, files={'image': upload()})
    assert response.status_code == 403 and response.json()['status'] == 'CONSENT_REQUIRED'
    assert context.bundle.detector.calls == 0

@pytest.mark.parametrize('mime,content,status', [('text/plain',b'x','UNSUPPORTED_MEDIA_TYPE'),
    ('image/png',b'\xff\xd8\xffBAD','UNSUPPORTED_MEDIA_TYPE'),
    ('image/png',b'garbage','UNSUPPORTED_MEDIA_TYPE'),
    ('image/png',b'\x89PNG\r\n\x1a\nBAD','INVALID_IMAGE')])
def test_encoding_before_inference(context, mime, content, status):
    response = context.client.post('/api/v1/face/detect', data=CONSENT, files={'image':('private',content,mime)})
    assert response.json()['status'] == status and response.status_code in {400,415}
    assert context.bundle.detector.calls == 0

def test_jpeg_allowlist(context):
    assert context.client.post('/api/v1/face/detect', data=CONSENT, files={'image':upload(kind='JPEG')}).status_code == 200

def test_black_image(context):
    stream=io.BytesIO()
    Image.new('RGB',(200,200)).save(stream,format='PNG')
    response=context.client.post('/api/v1/face/detect',data=CONSENT,files={'image':('x',stream.getvalue(),'image/png')})
    assert response.json()['status']=='BLACK_FRAME_REJECTED' and context.bundle.detector.calls==0

def test_file_and_request_limits(context):
    settings=replace(context.settings,max_image_bytes=1000,max_request_bytes=150_000)
    app=create_app(settings,context.models)
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        response=client.post('/api/v1/face/detect',data=CONSENT,files={'image':upload()})
        assert response.status_code==413
        response=client.post('/api/v1/face/detect',content=b'x'*150_001,headers={'content-type':'multipart/form-data; boundary=a'})
        assert response.status_code==413
    assert context.bundle.detector.calls==0

def test_pixel_limit_before_cv_decode(context, monkeypatch):
    import faceattend_api.services.image_input as module
    called=[]
    monkeypatch.setattr(module.cv2,'imdecode',lambda *a:called.append(1))
    with pytest.raises(APIError) as exc:
        decode(Part(mime='image/png',is_file=True,data=bytearray(photo())),replace(context.settings,max_pixels=100))
    assert exc.value.code=='PAYLOAD_TOO_LARGE' and called==[]

def test_animated_image_rejected(context):
    stream=io.BytesIO()
    images=[Image.new('RGB',(200,200),(i,100,100)) for i in (50,90)]
    images[0].save(stream,format='PNG',save_all=True,append_images=images[1:])
    response=context.client.post('/api/v1/face/detect',data=CONSENT,files={'image':('x',stream.getvalue(),'image/png')})
    assert response.status_code==400 and context.bundle.detector.calls==0

@pytest.mark.parametrize('headers', [{'host':'evil.example'},{'origin':'https://evil.example'}, {'origin':'null'}, {'host':'127.0.0.1@evil.example'}, {'host':'127.0.0.1/evil'}])
def test_host_origin_policy(context,headers):
    response=context.client.get('/health',headers=headers)
    assert response.status_code==403 and 'access-control-allow-origin' not in response.headers

def test_same_origin_allowed_no_cors(context):
    response=context.client.get('/health',headers={'origin':'http://127.0.0.1'})
    assert response.status_code==200 and 'access-control-allow-origin' not in response.headers
    assert context.client.options('/api/v1/face/detect',headers={'origin':'https://other'}).status_code==403

def test_nonloopback_client_rejected(context):
    with TestClient(context.app,base_url='http://127.0.0.1',client=('192.0.2.1',1)) as client:
        assert client.get('/health').status_code==403

@pytest.mark.parametrize('field', ['host','port','max_pixels','max_image_bytes','max_request_bytes'])
def test_invalid_settings(context,field):
    values={'host':'0.0.0.0','port':80,'max_pixels':10_000_001,'max_image_bytes':9*1024*1024,'max_request_bytes':41*1024*1024}
    with pytest.raises(ValueError):
        replace(context.settings,**{field:values[field]})

def test_private_storage_cannot_be_project(context):
    with pytest.raises(ValueError):
        replace(context.settings,private_parent=context.settings.root/'data/private')

def test_missing_and_extra_fields(context):
    assert context.client.post('/api/v1/face/detect',files={'image':upload()}).status_code==422
    assert context.client.post('/api/v1/face/detect',data=dict(CONSENT,extra='private'),files={'image':upload()}).status_code==422

def test_gpu_fails_closed(context):
    context.bundle.detector.session.providers=['CPUExecutionProvider']
    response=post(context)
    assert response.status_code==503 and response.json()['status']=='CUDA_UNAVAILABLE'
    assert context.bundle.detector.calls==0
    assert context.client.get('/health').json()['gpu_ready'] is False

def test_expression_unavailable(context):
    context.models.expression_service=None
    assert post(context,'expression/predict').status_code==503

def test_cuda_unavailable_at_startup_no_model_loading(context,monkeypatch):
    import torch
    import faceattend_api.dependencies as module
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    def forbidden(*args,**kwargs):
        raise AssertionError('Must not initialize/download a model when CUDA is absent')
    monkeypatch.setattr(module,'initialize_models',forbidden)
    models=ModelDependencies()
    app=create_app(context.settings,models)
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        response=client.get('/health')
        assert response.status_code==503 and response.json()['gpu_ready'] is False
        assert post(context).status_code==200  # The separately injected mock service still works.
        response=client.post('/api/v1/face/detect',data=CONSENT,files={'image':upload()})
        assert response.status_code==503 and response.json()['status']=='CUDA_UNAVAILABLE'

@pytest.mark.parametrize('n',[0,1,2,6])
def test_compare_reference_count(context,n):
    response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files(n))
    assert response.status_code==422 and context.bundle.detector.calls==0
    assert list(context.private.iterdir())==[]

def test_compare_selected_only_cleanup_encryption(context,monkeypatch):
    loaded=[]
    original_load=TemplateStore.load
    def load(store,user_id):
        loaded.append(user_id)
        encrypted=store._path(user_id).read_bytes()
        assert b'embedding' not in encrypted and b'volunteer_demo' not in encrypted
        return original_load(store,user_id)
    monkeypatch.setattr(TemplateStore,'load',load)
    response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files())
    assert response.status_code==200,response.text
    data=response.json()
    assert loaded==['volunteer_demo'] and data['cosine_similarity']>.99
    assert data['operational_verification']=={'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED'}
    assert 'claimed_id' not in data and list(context.private.iterdir())==[]

@pytest.mark.parametrize('case', ['duplicate_probe','duplicate_reference','invalid_id','enrollment_failure'])
def test_comparison_rejections_clean_up(context,case):
    files=comparison_files(probe_seed=1 if case=='duplicate_probe' else 20)
    if case=='duplicate_reference':
        files[1]=files[0]
    if case=='enrollment_failure':
        context.bundle.detector.mode='none'
    selected='Real Name' if case=='invalid_id' else 'volunteer_demo'
    response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id=selected),files=files)
    assert response.status_code==422 and list(context.private.iterdir())==[]

@pytest.mark.parametrize('error',[RuntimeError('private secret'),asyncio.CancelledError()])
def test_reference_cleanup_exception_and_cancellation(context,error,monkeypatch):
    monkeypatch.setenv('FACEATTEND_TEMPLATE_KEY','OPERATOR_SECRET_UNCHANGED')
    audit={}
    images=[np.random.default_rng(i).integers(60,190,(200,200,3),dtype=np.uint8) for i in range(3)]
    with pytest.raises(type(error)):
        with request_reference('volunteer_demo',images,context.bundle,context.private,audit=audit):
            raise error
    assert all(audit.values()) and list(context.private.iterdir())==[]
    assert os.environ['FACEATTEND_TEMPLATE_KEY']=='OPERATOR_SECRET_UNCHANGED'

def test_private_error_and_logs(context,caplog):
    context.bundle.detector.mode='exception'
    response=post(context)
    assert response.status_code==503
    @context.app.get('/unexpected')
    def fail():
        raise RuntimeError('SECRET C:\\private\\portrait.jpg vector=[1,2,3]')
    response=context.client.get('/unexpected')
    assert response.status_code==500 and response.json()['status']=='INTERNAL_ERROR'
    assert 'SECRET' not in response.text+caplog.text and 'portrait.jpg' not in response.text+caplog.text

def test_no_operational_verify_or_gallery(context):
    paths=context.client.get('/openapi.json').json()['paths']
    assert '/api/v1/face/verify' not in paths and len(paths)==6
    assert context.client.get('/docs').status_code==200
    schema=paths['/api/v1/face/compare-diagnostic']['post']['requestBody']['content']['multipart/form-data']['schema']
    assert schema['properties']['enrollment_images']['maxItems']==5
    html=context.client.get('/docs').text
    assert 'cdn.jsdelivr' not in html and '/docs-assets/swagger-ui-bundle.js' in html
    assert context.client.get('/docs-assets/swagger-ui-bundle.js').status_code==200

def test_one_active_request_no_queue(context):
    entered,release=threading.Event(),threading.Event()
    original=context.models.expression_service.predict
    def wait(image, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(image, **kwargs)
    context.models.expression_service.predict=wait
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(post,context,'expression/predict')
        assert entered.wait(5)
        try:
            assert post(context).json()['status']=='SERVICE_BUSY'
            assert context.client.get('/health').status_code==200
        finally:
            release.set()
        assert future.result().status_code==200
    assert not context.models.gate.locked()

def test_inference_deadline_waits_for_cleanup(context):
    original=context.models.expression_service.predict
    def slow(image, **kwargs):
        time.sleep(.03)
        return original(image, **kwargs)
    context.models.expression_service.predict=slow
    context.app.state.settings=replace(context.settings,inference_timeout_seconds=.001)
    response=post(context,'expression/predict')
    assert response.status_code==504 and not context.models.gate.locked()

def test_chunked_limits_and_timeout(context):
    class Stream:
        headers={'content-type':'multipart/form-data; boundary=test'}
        async def stream(self):
            yield b'--test\r\n'
            yield b'x'*100
    with pytest.raises(APIError) as exc:
        asyncio.run(read_parts(Stream(),replace(context.settings,max_request_bytes=20)))
    assert exc.value.code=='PAYLOAD_TOO_LARGE'
    class Slow(Stream):
        async def stream(self):
            await asyncio.sleep(.02)
            yield b'x'
    with pytest.raises(APIError) as exc:
        asyncio.run(read_parts(Slow(),replace(context.settings,upload_timeout_seconds=.001)))
    assert exc.value.code=='REQUEST_TIMEOUT'

def test_http_task_cancellation_keeps_gate_until_reference_cleanup(context,monkeypatch):
    import httpx
    import faceattend_api.services.temporary_compare as module
    entered,release=threading.Event(),threading.Event()
    original=module.process_frame
    def delayed(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'process_frame',delayed)
    async def run():
        transport=httpx.ASGITransport(app=context.app,client=('127.0.0.1',1))
        async with httpx.AsyncClient(transport=transport,base_url='http://127.0.0.1') as client:
            task=asyncio.create_task(client.post('/api/v1/face/compare-diagnostic',
                data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files()))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            assert entered.is_set()
            task.cancel()
            await asyncio.sleep(.02)
            try:
                assert context.models.gate.locked() and list(context.private.glob('phase3_session_*'))
                busy=await client.post('/api/v1/face/detect',data=CONSENT,files={'image':upload()})
                assert busy.status_code==503 and busy.json()['status']=='SERVICE_BUSY'
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not context.models.gate.locked() and list(context.private.iterdir())==[]
    asyncio.run(run())

def test_expression_rgb_decode_matches_phase1_exif_behavior():
    import cv2
    from faceattend_api.services.expression import rgb_input
    image=Image.new('RGB',(200,300),(70,90,120))
    exif=Image.Exif()
    exif[274]=6
    stream=io.BytesIO()
    image.save(stream,format='JPEG',exif=exif)
    encoded=stream.getvalue()
    bgr=cv2.imdecode(np.frombuffer(encoded,dtype=np.uint8),cv2.IMREAD_COLOR)
    actual=rgb_input(bgr,encoded)
    with Image.open(io.BytesIO(encoded)) as original:
        expected=original.convert('RGB')
    assert actual.size==(200,300) and bgr.shape[:2]==(200,300)
    np.testing.assert_array_equal(np.asarray(actual),np.asarray(expected))
