"""Explicit opt-in only: authorized private stills, local CUDA, no accuracy evaluation."""
import json
import io
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from starlette.testclient import TestClient
from face_verification import initialize_models, load_config
from face_verification.gpu import inspect_profile
from face_verification.research_demo import process_frame
from faceattend import CLASS_NAMES
from faceattend_api.config import Settings, ROOT, private_base
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_api.services.expression import ExpressionService
from faceattend_api.services.temporary_compare import request_reference

pytestmark = pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE3_GPU_TESTS') != '1',
    reason='Opt-in local CUDA integration on previously authorized images, disabled by default.')

@pytest.fixture(scope='module')
def real_api():
    parent = private_base()
    sources = parent / 'batch_03/captures'
    paths = [sources / f'volunteer_006_capture_{n:02}.jpg' for n in (1,2,7,3)]
    assert all(p.is_file() for p in paths), 'Authorized private inputs unavailable; do not fabricate a passing result.'
    assert torch.cuda.is_available()
    models = initialize_models(load_config(root=ROOT), profile=True)
    expression = ExpressionService(ROOT/'models/faceattend_best.pt')
    deps = ModelDependencies(face_models=models, expression_service=expression, load_on_start=False)
    deps.device_label = torch.cuda.get_device_name(0)
    settings = Settings(private_parent=parent)
    app = create_app(settings, deps)
    receipt = {'scope': 'New Phase 3 opt-in HTTP/adapter tests; not calibration or held-out expression evaluation',
               'private_images_saved': False, 'vectors_returned': False, 'model_id': models.model_id,
               'gpu': deps.device_label, 'results': {}}
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1',1)) as client:
        yield client, models, expression, paths, parent, receipt
        receipt['gpu_profiles'] = {name: inspect_profile(model.session.end_profiling()) for name,model in
                                   [('detector',models.detector),('recognizer',models.recognizer)]}
        receipt['temporary_directories_remaining'] = len(list(parent.glob('phase3_session_*')))
        assert receipt['temporary_directories_remaining'] == 0
        (ROOT/'reports/phase3/gpu_integration.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')

def upload(path):
    return ('authorized-private-input.jpg',path.read_bytes(),'image/jpeg')

CONSENT = {'adult_consent':'true','research_opt_in':'true'}

def test_actual_cuda_face_http_metadata(real_api):
    client,models,expression,paths,parent,receipt = real_api
    for endpoint,expected in [('detect','OK'),('embedding-info','EMBEDDING_GENERATED')]:
        response = client.post('/api/v1/face/'+endpoint,data=CONSENT,files={'image':upload(paths[3])})
        assert response.status_code==200,response.json().get('status')
        result=response.json()
        assert result['status']==expected and 'embedding' not in result
        if endpoint=='embedding-info':
            assert result['dimension']==512 and result['l2_norm']==pytest.approx(1,abs=1e-5)
        receipt['results'][endpoint]=result

def test_actual_comparison_matches_phase2a_adapter(real_api):
    client,models,expression,paths,parent,receipt = real_api
    selected='volunteer_006'
    files=[('enrollment_images',upload(p)) for p in paths[:3]]+[('probe',upload(paths[3]))]
    response=client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id=selected),files=files)
    assert response.status_code==200,response.json().get('status')
    result=response.json()
    assert result['operational_verification']=={'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED'}
    references=[cv2.imread(str(p)) for p in paths[:3]]
    probe=cv2.imread(str(paths[3]))
    with request_reference(selected,references,models,parent) as store:
        baseline=process_frame(probe,models,adult_consent=True,identity_consent=True,claimed_user_id=selected,store=store)
    assert baseline.status=='DIAGNOSTIC_COMPARISON_COMPLETED'
    assert result['cosine_similarity']==pytest.approx(baseline.cosine_similarity,abs=1e-6)
    assert not list(parent.glob('phase3_session_*'))
    receipt['results']['compare-diagnostic']=result
    receipt['adapter_parity_absolute_difference']=abs(result['cosine_similarity']-baseline.cosine_similarity)
    del references,probe

def test_actual_m1_cuda_visible_expression(real_api):
    client,models,expression,paths,parent,receipt = real_api
    assert expression.metadata['class_names']==CLASS_NAMES
    assert next(expression.model.parameters()).device.type=='cuda'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
        # A synthetic forward, separate from any held-out dataset evaluation.
        synthetic=expression.predict(np.full((224,224,3),127,dtype=np.uint8))
        torch.cuda.synchronize()
    cuda_events=sum(e.device_type==torch.autograd.DeviceType.CUDA for e in profile.events())
    assert cuda_events>0
    response=client.post('/api/v1/expression/predict',data=CONSENT,files={'image':upload(paths[3])})
    assert response.status_code==200,response.json().get('status')
    result=response.json()
    assert result['label'] in CLASS_NAMES and result['confidence_calibrated'] is False
    with Image.open(io.BytesIO(paths[3].read_bytes())) as original:
        tensor=expression.transform(original.convert('RGB')).unsqueeze(0).to(expression.device)
    with torch.inference_mode():
        expected=torch.softmax(expression.model(tensor),dim=1)[0]
    expected_index=int(expected.argmax().item())
    assert result['label']==CLASS_NAMES[expected_index]
    assert result['confidence']==pytest.approx(float(expected[expected_index].item()),abs=1e-7)
    receipt['results']['expression-predict']=result
    receipt['expression_cuda_forward_events']=cuda_events
    receipt['expression_parameter_device']='cuda:0'
    receipt['expression_original_preprocessing_parity']=True
    receipt['expression_preprocessing']={'input_size':expression.metadata['input_size'],
        'normalization':expression.metadata['normalization'],'channel_order':'RGB','full_decoded_image':True}
