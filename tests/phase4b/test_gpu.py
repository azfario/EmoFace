"""Explicit consenting-adult enrollment checks; no probe search or camera."""
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID
import os,json,subprocess,sys
import numpy as np
import pytest
from cryptography.fernet import Fernet
from starlette.testclient import TestClient
from face_verification import initialize_models,load_config
from face_verification.detector import detect_single_face
from face_verification.schemas import Status
from face_verification.gpu import inspect_profile
from faceattend_api.config import Settings,ROOT
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_api.services.image_input import Part,decode
from faceattend_enrollment.auth import password_record
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.dependencies import EnrollmentDependencies
from .test_postgres import live_database,db
from .conftest import PASSWORD

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4B_ADULT_IMAGE_TESTS')!='1',reason='Explicit consented adult/GPU/isolated-DB opt-in required.')

def profile_for_receipt(name,path,enrollment_completed):
    events=json.loads(Path(path).read_text())
    nodes=[event for event in events if event.get('cat')=='Node' and event.get('args',{}).get('provider')]
    if name=='recognizer' and not nodes and not enrollment_completed:
        return {'status':'NOT_EXERCISED',
            'reason':'Enrollment stopped before recognition; no computational CUDA evidence from this run.',
            'node_events_by_provider':{},'profile_file':Path(path).name}
    return inspect_profile(path)

@pytest.fixture(scope='module')
def adult_inputs():
    folder=Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private/phase4b_test_inputs'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert manifest['adult_consent'] is True and manifest['persistent_enrollment_consent'] is True
    bundle=initialize_models(load_config(root=ROOT),profile=True)
    receipt={'scope':'supplied adult enrollment functionality only; not calibration/evaluation',
             'source_names_or_paths_saved':False,'images_or_vectors_in_db':False,'camera_opened':False,
             'test_probe_folder_used':False,'groups':[]}
    try:
        yield folder,manifest,bundle,receipt
    finally:
        receipt['cuda_profiles']={}
        completed=any(group.get('enrolled_images',0)>0 for group in receipt['groups'])
        for name,model in [('detector',bundle.detector),('recognizer',bundle.recognizer)]:
            path=model.session.end_profiling()
            receipt['cuda_profiles'][name]=profile_for_receipt(name,path,completed)
        (ROOT/'reports/phase4b/adult_gpu_functional.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')

@pytest.mark.parametrize('group_index',[0,1,2])
def test_consented_adult_registration_survives_fresh_process_then_deletes(db,private_root,adult_inputs,group_index):
    folder,manifest,bundle,receipt=adult_inputs
    if group_index not in manifest.get('selected_group_indices',[0,1,2]):
        receipt['groups'].append({'group_index':group_index+1,'outcome':'Explicitly deferred by user',
            'images_processed':0,'enrolled_images':0,'live_enrollment_retained':False})
        pytest.skip('Explicit user deferral; no images processed or enrollment acceptance claimed.')
    item=manifest['groups'][group_index]
    settings=Settings(root=ROOT,private_parent=private_root,expression_enabled=False)
    eligible=[]
    rejected=0
    quality_checks=[]
    for image_index,filename in enumerate(item['images'],1):
        data=(folder/filename).read_bytes();mime='image/png' if data[:8]==b'\x89PNG\r\n\x1a\n' else 'image/jpeg'
        decoded=decode(Part(name='enrollment_images',mime=mime,is_file=True,data=bytearray(data)),settings)
        detected=detect_single_face(decoded,bundle)
        quality_checks.append({'image_index':image_index,'status':detected.status.value,
            'reason':detected.message or 'Passed all unchanged per-image QC gates.','quality':detected.quality})
        if detected.status==Status.OK:eligible.append((data,mime))
        else:rejected+=1
    # Deterministic first three passing QC; no score-based selection/calibration.
    qc_pass_count=len(eligible)
    if qc_pass_count<3:
        receipt['groups'].append({'group_index':group_index+1,'provided_images':len(item['images']),
            'qc_passed':qc_pass_count,'qc_rejected':rejected,'enrolled_images':0,
            'fresh_process_persistence':'not run','deletion':'not applicable',
            'live_enrollment_retained':False,'outcome':'replacement images required','quality_checks':quality_checks})
        raise AssertionError(f'Only {qc_pass_count} inputs pass unchanged QC; at least 3 required (private inputs withheld).')
    selected=eligible[:3]
    gallery=private_root/'gallery';gallery.mkdir()
    key_file=private_root/'operator-disposable.key';key_file.write_bytes(Fernet.generate_key())
    auth_file=private_root/'operator-disposable.secret';auth_file.write_text(json.dumps(password_record(PASSWORD)))
    config=EnrollmentSettings(enabled=True,database_url=db.runtime_url,gallery_dir=gallery,key_file=key_file,auth_file=auth_file)
    enrollment=EnrollmentDependencies(config)
    models=ModelDependencies(face_models=bundle,load_on_start=False)
    app=create_app(settings,models,enrollment_dependencies=enrollment)
    identity=None
    try:
        with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
            login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
            assert login.status_code==200
            headers={'x-faceattend-csrf':login.json()['csrf_token']}
            fields={'display_name':item['chosen_test_pseudonym'],'adult_consent':'true','research_opt_in':'true','enrollment_consent':'true','retention_days':'30'}
            files=[('enrollment_images',('private-test-input',data,mime)) for data,mime in selected]
            response=client.post('/api/v1/demo/enrollments',headers=headers,data=fields,files=files)
            assert response.status_code==200
            result=response.json();identity=UUID(result['identity_id'])
            assert result['verified'] is False and result['threshold'] is None
            assert 'template_locator' not in result and 'embedding' not in result
            assert len(client.get('/api/v1/demo/enrollments').json()['enrollments'])==1
        # Closed application/dependencies. Child reconstructs DB+key+store from disk.
        env=os.environ.copy()
        env.update(FACEATTEND_ENROLLMENT_ENABLED='true',FACEATTEND_ENROLLMENT_DATABASE_URL=db.runtime_url.render_as_string(hide_password=False),
            FACEATTEND_ENROLLMENT_GALLERY_DIR=str(gallery),FACEATTEND_ENROLLMENT_KEY_FILE=str(key_file),
            FACEATTEND_OPERATOR_AUTH_FILE=str(auth_file),FACEATTEND_RESTART_IDENTITY=str(identity),FACEATTEND_RESTART_MODEL=bundle.model_id)
        child=subprocess.run([sys.executable,str(ROOT/'scripts/phase4b_restart_check.py')],env=env,cwd=ROOT,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=30)
        assert child.returncode==0,'Fresh-process selected-reference validation failed (private details withheld).'
        assert json.loads(child.stdout)['restart_selected_reference']=='passed'
        restarted=EnrollmentDependencies(config);restarted.start()
        try:
            assert restarted.require_service().delete(identity)=={'status':'deleted'}
        finally:restarted.close()
        assert db.repository.list()==[] and not list((gallery/'templates').iterdir())
        receipt['groups'].append({'group_index':group_index+1,'provided_images':len(item['images']),'qc_passed':len(eligible),
            'qc_rejected':rejected,'enrolled_images':3,'fresh_process_persistence':'passed','deletion':'passed',
            'live_enrollment_retained':False,'quality_checks':quality_checks})
    finally:
        if identity is not None:
            # On failure, still revoke/remove the isolated enrollment and ciphertext.
            cleanup=EnrollmentDependencies(config);cleanup.start()
            try:cleanup.require_service().delete(identity)
            finally:cleanup.close()
