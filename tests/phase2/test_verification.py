import numpy as np
import pytest
from face_verification import verify_claimed_identity, Status
from face_verification.calibration import calibrate_development, save_calibration
from face_verification.schemas import FaceResult
from conftest import fake_detect, fake_embed

def verify(user, image, store, bundle, **kwargs):
    return verify_claimed_identity(user,image,store,bundle,adult_consent=True,identity_consent=True,
        detector=kwargs.pop('detector',fake_detect),embedder=kwargs.pop('embedder',fake_embed),**kwargs)

def calibration_fixture(bundle):
    # Synthetic unit fixture exercising production branches; not exported evidence.
    from test_calibration import synthetic_pairs
    report = calibrate_development(synthetic_pairs(),bundle.model_id,bundle.config,data_kind='real_consented')
    save_calibration(report,bundle.config.path(bundle.config.threshold_file))

def test_uncalibrated_fails_before_inference(enrolled,bundle):
    result = verify('volunteer_001',None,enrolled,bundle,detector=lambda *a: pytest.fail('no uncalibrated inference'))
    assert result.status == Status.THRESHOLD_NOT_CALIBRATED and not result.verified
    assert result.score is None and result.threshold is None

def test_unknown_id(store,bundle):
    assert verify('volunteer_404',None,store,bundle).status == Status.UNKNOWN_ID

def test_consent_required(enrolled,bundle):
    assert verify_claimed_identity('volunteer_001',None,enrolled,bundle).status == Status.CONSENT_REQUIRED

def test_accept_and_reject(enrolled,bundle):
    calibration_fixture(bundle)
    image = np.zeros((200,200,3),np.uint8)
    assert verify('volunteer_001',image,enrolled,bundle).status == Status.VERIFIED
    assert verify('volunteer_001',image,enrolled,bundle,embedder=lambda *a: np.array([0,1,0])).status == Status.NOT_VERIFIED

@pytest.mark.parametrize('status', [Status.NO_FACE,Status.MULTIPLE_FACES,Status.LOW_QUALITY_IMAGE,Status.MODEL_ERROR,Status.GPU_UNAVAILABLE])
def test_bad_face(enrolled,bundle,status):
    calibration_fixture(bundle)
    result = verify('volunteer_001',None,enrolled,bundle,detector=lambda *a: FaceResult(status))
    assert result.status == status and not result.verified

def test_fresh_probe_required(enrolled,bundle,images):
    calibration_fixture(bundle)
    assert verify('volunteer_001',images[0],enrolled,bundle).status == Status.DUPLICATE_IMAGE

def test_model_changed(enrolled,bundle):
    bundle.model_id = 'new_model'
    assert verify('volunteer_001',None,enrolled,bundle).status == Status.MODEL_MISMATCH

def test_no_other_identity_access(enrolled,bundle):
    calibration_fixture(bundle)
    original = enrolled.load
    calls = []
    def load(user):
        calls.append(user)
        return original(user)
    enrolled.load = load
    verify('volunteer_001',np.zeros((200,200,3),np.uint8),enrolled,bundle)
    assert calls == ['volunteer_001']
