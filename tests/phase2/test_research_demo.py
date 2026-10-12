from types import SimpleNamespace
from functools import partial
import json
import numpy as np
import pytest
from cryptography.fernet import Fernet
from face_verification import Config,Status,TemplateStore,VerificationError
from face_verification.schemas import FaceResult
from face_verification.enrollment import enroll_user
from face_verification.research_demo import process_frame,temporary_enrollment

def fake_face(image,models):
    return FaceResult(Status.OK,aligned=image,quality={'synthetic':True})
def fake_vector(face,models):
    return np.array([1.,0.,0.],np.float32)

@pytest.fixture
def models(tmp_path):
    session=SimpleNamespace(get_providers=lambda:['CUDAExecutionProvider'],disable_fallback=lambda:None)
    return SimpleNamespace(config=Config(root=tmp_path),model_id='synthetic_research_model',embedding_dim=3,
        detector=SimpleNamespace(session=session),recognizer=SimpleNamespace(session=session))

@pytest.fixture
def frame():
    return np.full((200,200,3),128,np.uint8)

def test_embedding_diagnostic_has_no_vector_or_acceptance(models,frame):
    result=process_frame(frame,models,adult_consent=True,identity_consent=True,detector=fake_face,embedder=fake_vector)
    value=result.to_dict()
    assert result.status=='EMBEDDING_GENERATED' and result.embedding_dimension==3
    assert value['verified'] is False and value['threshold'] is None
    assert value['operational_reason']=='THRESHOLD_NOT_CALIBRATED'
    assert 'embedding' not in value and json.loads(json.dumps(value))==value

@pytest.mark.parametrize('status',[Status.NO_FACE,Status.MULTIPLE_FACES,Status.LOW_QUALITY_IMAGE,Status.MODEL_ERROR,Status.GPU_UNAVAILABLE])
def test_bad_face_refused(models,frame,status):
    result=process_frame(frame,models,adult_consent=True,identity_consent=True,
        detector=lambda *a:FaceResult(status),embedder=lambda *a:pytest.fail('must not embed rejected face'))
    assert result.status==status.value and not result.verified

def test_consent_before_inference(models,frame):
    assert process_frame(frame,models,detector=lambda *a:pytest.fail('no inference')).status=='CONSENT_REQUIRED'

def test_gpu_fallback_rejected(models,frame):
    models.detector.session=SimpleNamespace(get_providers=lambda:['CPUExecutionProvider'])
    assert process_frame(frame,models,adult_consent=True,identity_consent=True).status=='GPU_UNAVAILABLE'

def test_missing_enrollment(models,frame):
    result=process_frame(frame,models,adult_consent=True,identity_consent=True,claimed_user_id='volunteer_006',store=None)
    assert result.status=='UNKNOWN_ID'

def test_only_selected_reference_and_diagnostic_without_threshold(models,frame,tmp_path):
    store=TemplateStore(tmp_path/'store',key=Fernet.generate_key())
    from datetime import datetime,timezone
    store.save('volunteer_006',[1,0,0],model_id=models.model_id,sample_hashes=['a','b','c'],
        consent={'adult':True,'identity_verification':True,'recorded_at':datetime.now(timezone.utc).isoformat()})
    calls=[]
    original=store.load
    def selected(user):
        calls.append(user)
        return original(user)
    store.load=selected
    result=process_frame(frame,models,adult_consent=True,identity_consent=True,
        claimed_user_id='volunteer_006',store=store,detector=fake_face,embedder=fake_vector)
    assert calls==['volunteer_006'] and result.cosine_similarity==1
    assert result.status=='DIAGNOSTIC_COMPARISON_COMPLETED'
    assert not result.verified and result.operational_reason=='THRESHOLD_NOT_CALIBRATED'

@pytest.mark.parametrize('exception',[False,True])
def test_temporary_enrollment_cleanup_even_on_exception(models,tmp_path,monkeypatch,exception):
    monkeypatch.setenv('FACEATTEND_TEMPLATE_KEY',Fernet.generate_key().decode())
    samples=[np.full((200,200,3),100+i,np.uint8) for i in range(3)]
    audit={}
    enroller=partial(enroll_user,detector=fake_face,embedder=fake_vector)
    def session():
        with temporary_enrollment('volunteer_006',samples,models,private_parent=tmp_path/'private',
            adult_consent=True,identity_consent=True,audit=audit,enroller=enroller) as store:
            assert store.exists('volunteer_006')
            if exception:
                raise RuntimeError('synthetic interruption')
    if exception:
        with pytest.raises(RuntimeError):
            session()
    else:
        session()
    assert audit['templates_removed'] and audit['directory_removed']
    assert not list((tmp_path/'private').rglob('*.fernet'))

def test_missing_operator_key_cleans_directory(models,tmp_path,monkeypatch):
    monkeypatch.delenv('FACEATTEND_TEMPLATE_KEY',raising=False)
    audit={}
    with pytest.raises(VerificationError) as exc:
        with temporary_enrollment('volunteer_006',[],models,private_parent=tmp_path/'private',
            adult_consent=True,identity_consent=True,audit=audit):
            pytest.fail('should not yield')
    assert exc.value.status==Status.KEY_UNAVAILABLE and audit['directory_removed']
