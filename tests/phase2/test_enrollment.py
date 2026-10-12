from face_verification import enroll_user, delete_enrollment, Status
from face_verification.enrollment import aggregate_embeddings
from face_verification.schemas import FaceResult, VerificationError
from conftest import fake_detect, fake_embed
import pytest

def test_no_consent(store,bundle,images):
    result = enroll_user('volunteer_001',images,store,bundle,detector=lambda *a: pytest.fail('must not infer without consent'))
    assert result.status == Status.CONSENT_REQUIRED

def test_duplicate(enrolled,bundle,images):
    assert enroll_user('volunteer_001',images,enrolled,bundle,adult_consent=True,identity_consent=True).status == Status.DUPLICATE_ID

def test_reenrollment(enrolled,bundle,images):
    delete_enrollment('volunteer_001',enrolled)
    assert enroll_user('volunteer_001',images,enrolled,bundle,adult_consent=True,identity_consent=True,detector=fake_detect,embedder=fake_embed).status == Status.OK

def test_repeated_image(store,bundle,images):
    assert enroll_user('volunteer_001',[images[0]]*3,store,bundle,adult_consent=True,identity_consent=True,detector=fake_detect,embedder=fake_embed).status == Status.DUPLICATE_IMAGE

def test_too_few(store,bundle,images):
    assert enroll_user('volunteer_001',images[:2],store,bundle,adult_consent=True,identity_consent=True).status == Status.INSUFFICIENT_DATA

def test_inconsistent():
    with pytest.raises(VerificationError) as exc:
        aggregate_embeddings([[1,0],[1,0],[-1,0]])
    assert exc.value.status == Status.INCONSISTENT_ENROLLMENT

def test_bad_face(store,bundle,images):
    assert enroll_user('volunteer_001',images,store,bundle,adult_consent=True,identity_consent=True,
        detector=lambda *a: FaceResult(Status.NO_FACE)).status == Status.NO_FACE
