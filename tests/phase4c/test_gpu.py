"""Opt-in adult separate-session functionality, strict CUDA, isolated PostgreSQL."""
from pathlib import Path
import json,os
from uuid import UUID
import pytest
from cryptography.fernet import Fernet
from starlette.testclient import TestClient
from face_verification import initialize_models,load_config
from face_verification.detector import detect_single_face
from face_verification.schemas import Status
from faceattend_api.config import Settings,ROOT
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_api.services.image_input import Part,decode
from faceattend_enrollment.auth import password_record
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_candidates.config import CandidateSettings
from tests.phase4b.test_postgres import live_database,db
from tests.phase4b.test_gpu import profile_for_receipt
from tests.phase4b.conftest import PASSWORD

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4C_ADULT_IMAGE_TESTS')!='1',reason='Explicit consenting adult separate-session GPU/isolated DB opt-in required.')

def accepted(folder,files,settings,bundle):
    selected=[];checks=[]
    for index,filename in enumerate(files,1):
        data=(folder/filename).read_bytes();mime='image/png' if data[:8]==b'\x89PNG\r\n\x1a\n' else 'image/jpeg'
        part=Part(name='image',mime=mime,is_file=True,data=bytearray(data))
        try:face=detect_single_face(decode(part,settings),bundle)
        finally:part.data.clear()
        checks.append({'input_index':index,'status':face.status.value,'quality':face.quality})
        if face.status==Status.OK:selected.append((data,mime))
    return selected,checks

def check_diagnostic(response,expected_count):
    if response.status_code!=200:raise AssertionError('Candidate request failed; private body withheld.')
    value=response.json()
    assert value['identification_status']=='UNCALIBRATED_DIAGNOSTIC'
    assert value['verified'] is False and value['threshold'] is None
    assert value['identity_status']=='Unknown (unverified)' and value['candidate_count']==expected_count
    for candidate in value['possible_candidates']:
        assert candidate['verified'] is False and candidate['label']=='Possible match (unverified)'
        assert -1<=candidate['cosine_similarity']<=1
        assert 'embedding' not in candidate and 'template_locator' not in candidate
    assert response.headers['cache-control']=='no-store'
    return value

def test_separate_session_gallery_ranking_unknown_and_empty(db,private_root):
    folder=Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private/phase4c_test_inputs'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['adult_consent'] is True and manifest['persistent_enrollment_consent'] is True
    assert manifest['separate_probe_sessions_user_confirmed'] is True and len(manifest['groups'])==3
    bundle=initialize_models(load_config(root=ROOT),profile=True)
    receipt={'scope':'small supplied consenting-adult separate-session functionality only',
        'separate_sessions_basis':'user confirmation; not independently authenticated capture dates',
        'calibration_or_population_evaluation':False,'threshold':None,'verified':False,
        'source_names_paths_images_vectors_saved':False,'camera_opened':False,'groups':[],'candidate_requests_completed':0}
    settings=Settings(root=ROOT,private_parent=private_root,expression_enabled=False)
    gallery=private_root/'gallery';gallery.mkdir()
    key=private_root/'disposable.key';key.write_bytes(Fernet.generate_key())
    auth=private_root/'disposable.secret';auth.write_text(json.dumps(password_record(PASSWORD)))
    config=EnrollmentSettings(enabled=True,database_url=db.runtime_url,gallery_dir=gallery,key_file=key,auth_file=auth)
    dependencies=EnrollmentDependencies(config)
    selected=[];ids=[]
    try:
        for index,group in enumerate(manifest['groups'],1):
            references,reference_checks=accepted(folder,group['enrollment'],settings,bundle)
            probes,probe_checks=accepted(folder,group['probes'],settings,bundle)
            receipt['groups'].append({'anonymous_group_index':index,'enrollment_qc':reference_checks,'probe_qc':probe_checks,
                'enrollment_passed':len(references),'probe_passed':len(probes),'enrolled_images':0,'selected_probe_index':None})
            if len(references)<3 or not probes:
                raise AssertionError(f'Anonymous group {index} lacks three accepted references or one accepted separate-session probe; private inputs withheld.')
            receipt['groups'][-1]['selected_probe_index']=next(c['input_index'] for c in probe_checks if c['status']==Status.OK.value)
            selected.append((references[:3],probes[0]))
        models=ModelDependencies(face_models=bundle,load_on_start=False)
        app=create_app(settings,models,enrollment_dependencies=dependencies,candidate_settings=CandidateSettings(enabled=True))
        with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
            assert login.status_code==200
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-candidate-opt-in':'1'}
            for index,(group,(refs,probe)) in enumerate(zip(manifest['groups'],selected)):
                response=client.post('/api/v1/demo/enrollments',headers=headers,
                    data={'display_name':group['chosen_test_pseudonym'],'adult_consent':'true','research_opt_in':'true','enrollment_consent':'true','retention_days':'30'},
                    files=[('enrollment_images',('private-input',data,mime)) for data,mime in refs])
                if response.status_code!=200:raise AssertionError('Actual enrollment failed; private body withheld.')
                ids.append(UUID(response.json()['identity_id']));receipt['groups'][index]['enrolled_images']=3
            for refs,(data,mime) in selected:
                response=client.post('/api/v1/demo/candidates?top_k=3',headers=headers,
                    data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
                value=check_diagnostic(response,3)
                assert value['eligible_reference_count']==3
                receipt['candidate_requests_completed']+=1
            # This adult is absent from the eligible gallery after revocation.
            response=client.post(f'/api/v1/demo/enrollments/{ids[0]}/revoke',headers=headers)
            assert response.status_code==200
            data,mime=selected[0][1]
            response=client.post('/api/v1/demo/candidates',headers=headers,
                data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
            value=check_diagnostic(response,2)
            assert all(item['identity_id']!=str(ids[0]) for item in value['possible_candidates'])
            receipt['candidate_requests_completed']+=1
            receipt['gallery_absent_subject_remained_unknown']=True
            for identity in ids:assert client.delete(f'/api/v1/demo/enrollments/{identity}',headers=headers).status_code==200
            ids.clear()
            response=client.post('/api/v1/demo/candidates',headers=headers,
                data={'adult_consent':'true','research_opt_in':'true'},files={'image':('private-probe',data,mime)})
            value=check_diagnostic(response,0)
            assert value['reason']=='NO_ELIGIBLE_REFERENCES'
            receipt['candidate_requests_completed']+=1
            receipt['empty_gallery_remained_unknown']=True
            client.delete('/api/v1/demo/session',headers=headers)
        assert db.repository.list()==[] and not list((gallery/'templates').iterdir())
        receipt['isolated_registration_deletion']='passed'
    finally:
        selected.clear()
        if ids:
            cleanup=EnrollmentDependencies(config);cleanup.start()
            try:
                for identity in ids:cleanup.require_service().delete(identity)
            finally:cleanup.close()
        receipt['cuda_profiles']={}
        for name,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]:
            receipt['cuda_profiles'][name]=profile_for_receipt(name,model.session.end_profiling(),bool(receipt['candidate_requests_completed']))
        (ROOT/'reports/phase4c/adult_gpu_functional.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
