"""Actual local UI API functionality; consenting adult inputs, never accuracy."""
import json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import text
from starlette.testclient import TestClient
from face_verification import initialize_models,load_config
from faceattend_api.config import ROOT,Settings
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_api.services.expression import ExpressionService
from faceattend_enrollment.auth import password_record
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_candidates.config import CandidateSettings
from faceattend_analysis.config import AnalysisSettings
from faceattend_ui.integration import UISettings
from tests.phase4b.test_postgres import live_database,db
from tests.phase4b.test_gpu import profile_for_receipt
from tests.phase4c.test_gpu import accepted
from tests.phase4b.conftest import PASSWORD

pytestmark=pytest.mark.skipif(any(os.environ.get(name)!='1' for name in (
    'FACEATTEND_RUN_PHASE5_GPU_TESTS','FACEATTEND_RUN_PHASE5_POSTGRES_TESTS','FACEATTEND_RUN_PHASE5_ADULT_IMAGE_TESTS')),
    reason='Explicit CUDA, isolated PostgreSQL and consenting-adult opt-ins required.')

def test_actual_ui_preview_optional_outline_and_expression_no_history(db,private_root):
    bundle=initialize_models(load_config(root=ROOT),profile=True)
    expression=ExpressionService(ROOT/'models/faceattend_best.pt')
    settings=Settings(root=ROOT,private_parent=private_root)
    gallery=private_root/'ui-gallery';gallery.mkdir()
    key=private_root/'ui-disposable.key';key.write_bytes(Fernet.generate_key())
    auth=private_root/'ui-disposable.secret';auth.write_text(json.dumps(password_record(PASSWORD)))
    config=EnrollmentSettings(enabled=True,database_url=db.runtime_url,gallery_dir=gallery,key_file=key,auth_file=auth)
    deps=EnrollmentDependencies(config)
    app=create_app(settings,ModelDependencies(face_models=bundle,expression_service=expression,load_on_start=False),
        enrollment_dependencies=deps,candidate_settings=CandidateSettings(enabled=True),analysis_settings=AnalysisSettings(enabled=True),
        ui_settings=UISettings(enabled=True))
    folder=Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private/phase4c_test_inputs'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['adult_consent'] and manifest['separate_probe_sessions_user_confirmed']
    group=manifest['groups'][0]
    refs,_=accepted(folder,group['enrollment'],settings,bundle)
    probes,_=accepted(folder,group['probes'],settings,bundle)
    assert len(refs)>=3 and probes,'Acceptable consenting-adult inputs required; private details withheld.'
    data,mime=probes[0];receipt={'scope':'local UI API functionality only; no accuracy or mental-state evaluation',
        'held_out_dataset_accessed':False,'camera_opened':False,'private_names_paths_images_vectors_labels_scores_saved':False}
    try:
        with db.owner.connect() as connection:
            aggregates=connection.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()
        with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
            assert client.get('/app/').status_code==200
            status=client.get('/api/v1/demo/ui-status').json()
            assert status['registration_database']=='ready' and status['operator_auth_ready'] and not status['fixture_mode']
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
            assert login.status_code==200
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-box-opt-in':'1',
                'x-faceattend-candidate-opt-in':'1','x-faceattend-expression-opt-in':'1'}
            response=client.post('/api/v1/demo/enrollments',headers=headers,
                data={'display_name':'Fictional UI CUDA reference','adult_consent':'true','research_opt_in':'true','enrollment_consent':'true','retention_days':'1'},
                files=[('enrollment_images',('reference',blob,kind)) for blob,kind in refs[:3]])
            assert response.status_code==200,'Enrollment failed; private response withheld.'
            identity=response.json()['identity_id'];before=client.get('/api/v1/demo/enrollments').json()
            consent={'adult_consent':'true','research_opt_in':'true'}
            def analyze(path,payload=None):
                result=client.post(path,headers=headers,data=consent,files={'image':('probe',*(payload or (data,mime)))})
                assert result.status_code==200,'UI analysis failed; private response withheld.'
                assert result.headers['cache-control']=='no-store'
                assert 'embedding' not in result.text and 'template_locator' not in result.text
                value=result.json();assert value['face_count']==1 and value['verified'] is False and value['threshold'] is None
                return value['faces'][0]
            bare=analyze('/api/v1/demo/preview-face')
            assert 'face_box' not in bare and 'visible_expression_prediction' not in bare
            outlined=analyze('/api/v1/demo/preview-face?include_face_box=true&include_candidates=true')
            box=outlined['face_box'];assert box['width']>0 and box['height']>0
            assert box['x']+box['width']<=box['image_width'] and box['y']+box['height']<=box['image_height']
            assert outlined['identity_diagnostic']['candidate_count']==1 and 'visible_expression_prediction' not in outlined
            combined=analyze('/api/v1/demo/analyze-face?include_face_box=true&include_candidates=true')
            assert combined['face_box']==box and not combined['visible_expression_prediction']['score_calibrated']
            assert not combined['visible_expression_prediction']['domain_accuracy_validated']
            # Controlled in-memory blur verifies the requested exception with real CUDA.
            import cv2
            import numpy as np
            from faceattend_analysis.quality import expression_detection
            from face_verification.schemas import Status
            decoded=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            blurred=None
            for sigma in (2,4,6,8):
                candidate=cv2.GaussianBlur(decoded,(0,0),sigma)
                detected,bypassed=expression_detection(candidate,bundle,True)
                if detected.status==Status.OK and bypassed:blurred=candidate;break
            assert blurred is not None,'Could not construct the bounded blurry probe; private details withheld.'
            ok,encoded=cv2.imencode('.png',blurred);assert ok
            payload=(encoded.tobytes(),'image/png')
            refused=client.post('/api/v1/demo/analyze-face',headers=headers,data=consent,files={'image':('probe',*payload)})
            assert refused.status_code==422 and refused.json()['status']=='LOW_QUALITY_IMAGE'
            headers['x-faceattend-blur-opt-in']='1'
            relaxed=analyze('/api/v1/demo/analyze-face?allow_blurry_expression=true&include_candidates=true',payload)
            assert relaxed['input_quality']['status']=='BLUR_GATE_BYPASSED'
            assert relaxed['identity_diagnostic']['possible_candidates']==[]
            assert relaxed['identity_diagnostic']['reason']=='BLURRY_FRAME_IDENTITY_MATCHING_SKIPPED'
            assert relaxed['visible_expression_prediction']['score_calibrated'] is False
            assert bundle.config.min_blur_variance==60
            receipt.update(real_blurry_expression_functionality=True,strict_default_blur_refusal=True,blurry_identity_search_skipped=True)
            # Exercise the live-only opt-in through unchanged CUDA models, without camera access.
            from time import perf_counter
            headers['x-faceattend-live-opt-in']='1'
            elapsed=[]
            for _ in range(3):
                started=perf_counter()
                live=analyze('/api/v1/demo/analyze-face?live_camera=true&allow_blurry_expression=true&include_face_box=true')
                elapsed.append(round((perf_counter()-started)*1000,2))
                assert live['face_box']==box and live['visible_expression_prediction']['score_calibrated'] is False
            receipt.update(real_live_analysis_api=True,sequential_live_requests=3,warm_local_image_request_ms=elapsed,
                timing_scope='Sequential local-image TestClient requests with CUDA profiling; not browser-camera FPS.')
            assert client.get('/api/v1/demo/enrollments').json()==before
            assert client.delete(f'/api/v1/demo/enrollments/{identity}',headers=headers).status_code==200
            assert client.delete('/api/v1/demo/session',headers=headers).status_code==200
        with db.owner.connect() as connection:
            assert connection.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()==aggregates
        assert db.repository.list()==[] and not list((gallery/'templates').iterdir())
        receipt.update(preview_without_expression=True,explicit_outline_geometry=True,optional_unverified_candidates=True,
            combined_uncalibrated_expression=True,metadata_and_telemetry_unchanged=True,all_disposable_registrations_deleted=True)
    finally:
        cleanup=EnrollmentDependencies(config);cleanup.start()
        try:
            for row in cleanup.require_service().repository.list():cleanup.require_service().delete(row.identity_id)
        finally:cleanup.close()
        receipt['cuda_profiles']={name:profile_for_receipt(name,model.session.end_profiling(),True)
            for name,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]}
        (ROOT/'reports/phase5/gpu_functional.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        refs.clear();probes.clear()
