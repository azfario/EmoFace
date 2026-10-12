"""Explicit local CUDA/crop functionality; no held-out dataset or labelled accuracy."""
from pathlib import Path
from types import SimpleNamespace
import json,os
from uuid import UUID
import numpy as np
import torch
import pytest
from cryptography.fernet import Fernet
from starlette.testclient import TestClient
from sqlalchemy import text
from face_verification import initialize_models,load_config
from faceattend import CLASS_NAMES
from faceattend_api.config import ROOT,Settings
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.services.expression import ExpressionService
from faceattend_api.main import create_app
from faceattend_enrollment.auth import password_record
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_candidates.config import CandidateSettings
from faceattend_analysis.config import AnalysisSettings
from faceattend_analysis.crop import rgb_crop
from faceattend_analysis.expression import CropExpression
from faceattend_api.errors import APIError
from tests.phase4b.test_postgres import live_database,db
from tests.phase4b.test_gpu import profile_for_receipt
from tests.phase4c.test_gpu import accepted
from .test_crop import face
from .conftest import PASSWORD

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4D_GPU_TESTS')!='1',reason='Explicit existing CUDA environment opt-in required.')

@pytest.fixture(scope='module')
def runtime():
    assert torch.cuda.is_available()
    bundle=initialize_models(load_config(root=ROOT),profile=True)
    # Synthetic model compute makes a GPU-only run honest without adult inputs.
    bundle.detector.detect(np.full((640,640,3),127,np.uint8),input_size=(640,640),max_num=0)
    bundle.recognizer.get_feat(np.full((112,112,3),127,np.uint8))
    expression=ExpressionService(ROOT/'models/faceattend_best.pt')
    receipt={'scope':'crop input/strict CUDA and consenting-adult functionality only; no accuracy or diagnosis',
        'held_out_dataset_accessed':False,'source_names_paths_images_vectors_saved':False,
        'individual_expression_labels_scores_saved':False,'camera_opened':False,'candidate_threshold':None,
        'adult_probe_requests':0,'crop_rgb_m1_input_parity':False,'groups_qc':[]}
    try:yield SimpleNamespace(bundle=bundle,expression=expression,crop=CropExpression(expression),receipt=receipt)
    finally:
        receipt['cuda_profiles']={name:profile_for_receipt(name,model.session.end_profiling(),receipt['adult_probe_requests']>0)
            for name,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]}
        (ROOT/'reports/phase4d/gpu_functional.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')

def test_actual_m1_crop_input_normalization_logits_and_cuda(runtime,monkeypatch):
    rgb=rgb_crop(np.full((140,120,3),(17,53,201),np.uint8),face())
    inputs=[];outputs=[]
    first=runtime.expression.model.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
    last=runtime.expression.model.register_forward_hook(lambda module,args,result:outputs.append(result.detach().clone()))
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            result=runtime.crop.predict(rgb);torch.cuda.synchronize()
        expected=runtime.expression.transform(rgb).unsqueeze(0).to(runtime.expression.device)
        torch.testing.assert_close(inputs[-1],expected,rtol=0,atol=0)
        assert inputs[-1].shape==(1,3,224,224) and inputs[-1].device.type=='cuda'
        assert outputs[-1].shape==(1,7) and outputs[-1].device.type=='cuda'
        scores=torch.softmax(outputs[-1],dim=1)[0]
        assert result['label']==CLASS_NAMES[int(scores.argmax().item())]
        assert result['softmax_score']==pytest.approx(float(scores.max().item()),abs=1e-7)
        events=sum(event.device_type==torch.autograd.DeviceType.CUDA for event in profile.events())
        assert events>0
        runtime.receipt.update(expression_cuda_events=events,expression_parameter_device='cuda:0',expression_input_device='cuda:0',
            expression_logits_device='cuda:0',expression_logits_shape=[1,7],crop_rgb_m1_input_parity=True)
    finally:first.remove();last.remove();inputs.clear();outputs.clear();rgb.close()
    # Wrong forward contracts fail safely; changes exist only in RAM and are restored.
    with monkeypatch.context() as patch:
        patch.setattr(runtime.expression.model,'forward',lambda tensor:torch.zeros((1,8),device=runtime.expression.device))
        from PIL import Image
        with pytest.raises(APIError) as error:runtime.crop.predict(Image.new('RGB',(80,80)))
        assert error.value.code=='MODEL_NOT_READY'

@pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4D_ADULT_IMAGE_TESTS')!='1',reason='Explicit consenting-adult/isolated-DB opt-in required.')
def test_actual_adult_combined_anonymous_empty_and_crop_domain(db,private_root,runtime):
    folder=Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private/phase4c_test_inputs'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['adult_consent'] is True and manifest['separate_probe_sessions_user_confirmed'] is True
    settings=Settings(root=ROOT,private_parent=private_root)
    gallery=private_root/'gallery';gallery.mkdir()
    key=private_root/'disposable.key';key.write_bytes(Fernet.generate_key())
    auth=private_root/'disposable.secret';auth.write_text(json.dumps(password_record(PASSWORD)))
    config=EnrollmentSettings(enabled=True,database_url=db.runtime_url,gallery_dir=gallery,key_file=key,auth_file=auth)
    deps=EnrollmentDependencies(config)
    models=ModelDependencies(face_models=runtime.bundle,expression_service=runtime.expression,load_on_start=False)
    app=create_app(settings,models,enrollment_dependencies=deps,candidate_settings=CandidateSettings(enabled=True),analysis_settings=AnalysisSettings(enabled=True))
    selected=[];ids=[];uncertainty=[]
    try:
        for index,group in enumerate(manifest['groups'],1):
            refs,ref_checks=accepted(folder,group['enrollment'],settings,runtime.bundle)
            probes,probe_checks=accepted(folder,group['probes'],settings,runtime.bundle)
            runtime.receipt['groups_qc'].append({'anonymous_group_index':index,'enrollment_passed':len(refs),'probe_passed':len(probes),
                'enrollment_supplied':len(ref_checks),'probe_supplied':len(probe_checks)})
            if len(refs)<3 or not probes:raise AssertionError('Supplied group lacks accepted references/probe; private input details withheld.')
            selected.append((refs[:3],probes[0]))
        with db.owner.connect() as c:before_aggregate=c.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()
        with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
            assert login.status_code==200
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-expression-opt-in':'1','x-faceattend-candidate-opt-in':'1'}
            for group,(refs,probe) in zip(manifest['groups'],selected):
                response=client.post('/api/v1/demo/enrollments',headers=headers,
                    data={'display_name':group['chosen_test_pseudonym'],'adult_consent':'true','research_opt_in':'true','enrollment_consent':'true','retention_days':'30'},
                    files=[('enrollment_images',('private-input',data,mime)) for data,mime in refs])
                if response.status_code!=200:raise AssertionError('Enrollment failed; private response withheld.')
                ids.append(UUID(response.json()['identity_id']))
            records=[db.repository.get(identity) for identity in ids]
            for refs,(data,mime) in selected:
                response=client.post('/api/v1/demo/analyze-face?include_candidates=true',headers=headers,
                    data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
                if response.status_code!=200:raise AssertionError('Combined diagnostics failed; private response withheld.')
                result=response.json();assert result['face_count']==1 and result['not_a_mood_or_diagnosis'] is True
                item=result['faces'][0];identity=item['identity_diagnostic'];expression=item['visible_expression_prediction']
                assert identity['verified'] is False and identity['threshold'] is None and identity['candidate_count']==3
                assert identity['identity_status']=='Unknown (unverified)'
                assert expression['label'] in CLASS_NAMES and expression['score_calibrated'] is False and expression['domain_accuracy_validated'] is False
                assert 0<=expression['softmax_score']<=1 and 0<=expression['softmax_entropy_nats']<=float(np.log(7))+1e-6
                assert 'embedding' not in response.text and 'template_locator' not in response.text and response.headers['cache-control']=='no-store'
                uncertainty.append((expression['softmax_score'],expression['softmax_entropy_nats']))
                runtime.receipt['adult_probe_requests']+=1
            assert [db.repository.get(identity) for identity in ids]==records
            # Expression opt-out never runs classification, even with a valid session.
            data,mime=selected[0][1];without=dict(headers);without.pop('x-faceattend-expression-opt-in')
            response=client.post('/api/v1/demo/analyze-face',headers=without,data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
            assert response.status_code==403 and 'visible_expression_prediction' not in response.text
            for identity in ids:assert client.delete(f'/api/v1/demo/enrollments/{identity}',headers=headers).status_code==200
            ids.clear()
            # Unenrolled voluntary subject still gets anonymous expression after gallery deletion.
            for candidates in (False,True):
                response=client.post('/api/v1/demo/analyze-face',params={'include_candidates':str(candidates).lower()},headers=headers,
                    data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
                if response.status_code!=200:raise AssertionError('Anonymous/empty-gallery analysis failed; private response withheld.')
                result=response.json()['faces'][0]
                assert result['identity_diagnostic']['candidate_count']==0 and result['visible_expression_prediction']['label'] in CLASS_NAMES
                assert 'identity_id' not in response.text and 'display_name' not in response.text
                runtime.receipt['adult_probe_requests']+=1
            client.delete('/api/v1/demo/session',headers=headers)
        with db.owner.connect() as c:assert c.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()==before_aggregate
        assert db.repository.list()==[] and not list((gallery/'templates').iterdir())
        runtime.receipt.update(identity_metadata_unchanged_by_analysis=True,anonymous_aggregate_rows_unchanged=True,
            all_disposable_registrations_deleted=True,anonymous_empty_gallery_functionality=True,
            pilot_uncertainty={'samples':len(uncertainty),'softmax_top_score_min':min(s[0] for s in uncertainty),'softmax_top_score_max':max(s[0] for s in uncertainty),
                'entropy_nats_min':min(s[1] for s in uncertainty),'entropy_nats_max':max(s[1] for s in uncertainty),'interpretation':'descriptive uncalibrated values only; not correctness or mental-state evidence'})
    finally:
        selected.clear();uncertainty.clear()
        if ids:
            cleanup=EnrollmentDependencies(config);cleanup.start()
            try:
                for identity in ids:cleanup.require_service().delete(identity)
            finally:cleanup.close()
