from types import SimpleNamespace
import numpy as np
import pytest
from face_verification.quality import validate_image, assess_face
from face_verification.detector import detect_single_face
from face_verification.schemas import Status, VerificationError

@pytest.mark.parametrize('image,status', [(None,Status.INVALID_IMAGE), (np.zeros((200,200)),Status.INVALID_IMAGE),
    (np.zeros((200,200,3)),Status.INVALID_IMAGE), (np.zeros((50,50,3),dtype=np.uint8),Status.LOW_QUALITY_IMAGE)])
def test_bad_input(config,image,status):
    with pytest.raises(VerificationError) as exc:
        validate_image(image, config)
    assert exc.value.status == status

def test_blurry_rejected(config):
    with pytest.raises(VerificationError) as exc:
        assess_face(np.full((200,200,3),128,np.uint8), [30,30,170,170], [[65,70],[135,70],[100,100],[75,135],[125,135]], config)
    assert exc.value.status == Status.LOW_QUALITY_IMAGE

@pytest.mark.parametrize('count,status', [(0,Status.NO_FACE),(2,Status.MULTIPLE_FACES)])
def test_face_count(config,count,status):
    session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'], disable_fallback=lambda: None)
    detector = SimpleNamespace(session=session, detect=lambda *a,**kw: (np.zeros((count,5)),None))
    bundle = SimpleNamespace(config=config,detector=detector)
    image = np.random.default_rng(1).integers(0,255,(200,200,3),dtype=np.uint8)
    assert detect_single_face(image,bundle).status == status

def test_multiple_faces_rejected_before_quality(config):
    session = SimpleNamespace(get_providers=lambda: ['CUDAExecutionProvider'], disable_fallback=lambda: None)
    detector = SimpleNamespace(session=session, detect=lambda *a,**kw: (np.array([[10,10,100,100,.9],[110,10,120,20,.7]]),None))
    assert detect_single_face(np.full((200,200,3),128,np.uint8),SimpleNamespace(config=config,detector=detector)).status == Status.MULTIPLE_FACES
