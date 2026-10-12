from types import SimpleNamespace
import numpy as np
import pytest
from cryptography.fernet import Fernet
from face_verification import Config, TemplateStore
from face_verification.schemas import FaceResult, Status

@pytest.fixture
def config(tmp_path):
    return Config(root=tmp_path)

@pytest.fixture
def store(tmp_path):
    return TemplateStore(tmp_path / 'private', key=Fernet.generate_key())

@pytest.fixture
def bundle(config):
    return SimpleNamespace(config=config, model_id='synthetic_test_model', embedding_dim=3)

@pytest.fixture
def images():
    return [np.full((200,200,3), i+100, dtype=np.uint8) for i in range(3)]

def fake_detect(image, bundle):
    return FaceResult(Status.OK, aligned=image)

def fake_embed(face, bundle):
    return np.array([1., 0., 0.], dtype=np.float32)

@pytest.fixture
def enrolled(store, bundle, images):
    from face_verification import enroll_user
    result = enroll_user('volunteer_001', images, store, bundle, adult_consent=True,
        identity_consent=True, detector=fake_detect, embedder=fake_embed)
    assert result.status == Status.OK
    return store
