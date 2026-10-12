import asyncio, threading
from unittest.mock import MagicMock
import pytest
from tests.phase3.conftest import CONSENT, upload, comparison_files
from faceattend_db.instrumentation import Telemetry
from faceattend_db.config import DBSettings

def enable(context):
    service=Telemetry(DBSettings())
    service.status='ready'
    service.repository=MagicMock()
    context.app.state.telemetry=service
    return service

def post(context,path='face/detect',**kwargs):
    return context.client.post('/api/v1/'+path,data=CONSENT,files={'image':upload()},**kwargs)

def test_disabled_db_health_independent_of_models(context):
    assert context.client.get('/health/db').json()=={'status':'disabled'}
    assert context.client.get('/health').status_code==200
    assert post(context).status_code==200

@pytest.mark.parametrize('path,operation',[('face/detect','face_detect'),('face/embedding-info','embedding_info'),('expression/predict','expression_predict')])
def test_success_only_safe_scalars_off_event_loop(context,path,operation):
    baseline=post(context,path).json()
    service=enable(context)
    threads=[]
    service.repository.increment_daily.side_effect=lambda *args: threads.append(threading.current_thread().name)
    result=post(context,path)
    assert result.status_code==200 and result.json()==baseline
    args=service.repository.increment_daily.call_args.args
    assert len(args)==3 and args[:2]==(operation,'completed') and type(args[2]) is int and args[2]%10==0
    assert threads==['AnyIO worker thread']

@pytest.mark.parametrize('mode',['none','multiple','clipped'])
def test_quality_rejections_broad_only(context,mode):
    service=enable(context); context.bundle.detector.mode=mode
    assert post(context).status_code==422
    assert service.repository.increment_daily.call_args.args[:2]==('face_detect','quality_rejected')

def test_ml_error_counts_processing_error(context):
    service=enable(context); context.bundle.detector.mode='exception'
    assert post(context).status_code==503
    assert service.repository.increment_daily.call_args.args[:2]==('face_detect','processing_error')

@pytest.mark.parametrize('case',['consent','origin','malformed','black','invalid_id','extra'])
def test_rejections_before_ml_never_persist(context,case):
    service=enable(context)
    if case=='consent':
        response=context.client.post('/api/v1/face/detect',data={'adult_consent':'false','research_opt_in':'true'},files={'image':upload()})
    elif case=='origin': response=post(context,headers={'Origin':'https://evil.example'})
    elif case=='malformed': response=context.client.post('/api/v1/face/detect',data=CONSENT,files={'image':('x',b'bad','image/png')})
    elif case=='black':
        import io
        from PIL import Image
        stream=io.BytesIO(); Image.new('RGB',(200,200)).save(stream,format='PNG')
        response=context.client.post('/api/v1/face/detect',data=CONSENT,files={'image':('x',stream.getvalue(),'image/png')})
    elif case=='invalid_id': response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id='Real Name'),files=comparison_files())
    else: response=context.client.post('/api/v1/face/detect',data=dict(CONSENT,unexpected='private'),files={'image':upload()})
    assert response.status_code>=400
    service.repository.increment_daily.assert_not_called()

def test_compare_no_identifying_or_biometric_values_persist(context):
    service=enable(context)
    response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files())
    assert response.status_code==200
    assert response.json()['operational_verification']=={'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED'}
    args=service.repository.increment_daily.call_args.args
    assert args[:2]==('compare_diagnostic','completed') and type(args[2]) is int and len(args)==3
    assert list(context.private.iterdir())==[]

@pytest.mark.parametrize('path',['face/detect','face/embedding-info','expression/predict'])
def test_database_failure_preserves_result_and_sanitizes(context,path,caplog):
    expected=post(context,path).json()
    service=enable(context)
    service.repository.increment_daily.side_effect=RuntimeError('SECRET postgres://owner:password portrait vector')
    response=post(context,path)
    assert response.status_code==200 and response.json()==expected
    assert service.status=='unavailable' and 'SECRET' not in caplog.text+response.text
    assert context.client.get('/health').status_code==200

def test_unexpected_adapter_failure_is_fail_open(context):
    service=enable(context)
    service.record=MagicMock(side_effect=RuntimeError('SECRET'))
    assert post(context).status_code==200

def test_coarse_health_and_original_openapi_unchanged(context,monkeypatch):
    service=enable(context)
    monkeypatch.setattr('faceattend_db.instrumentation.runtime_ready',lambda _:False)
    response=context.client.get('/health/db')
    assert response.status_code==503 and response.json()=={'status':'unavailable'}
    assert len(context.client.get('/openapi.json').json()['paths'])==6
    assert set(context.client.get('/health').json())=={'status','model_readiness','gpu_ready','gpu_required'}

def test_busy_database_slot_drops_metric_without_queue(context):
    service=enable(context)
    service.gate.acquire()
    try:
        assert post(context).status_code==200
        service.repository.increment_daily.assert_not_called()
    finally: service.gate.release()
