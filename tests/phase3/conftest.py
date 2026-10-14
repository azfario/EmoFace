from types import SimpleNamespace
import io
import os
from pathlib import Path
import tempfile
import cv2
import numpy as np
from PIL import Image
import pytest
from starlette.testclient import TestClient
from face_verification.config import Config
from faceattend_api.config import Settings
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_api.schemas import Expression

class Session:
    def __init__(self):
        self.providers = ['CUDAExecutionProvider']
    def get_providers(self):
        return self.providers
    def disable_fallback(self):
        pass

class Detector:
    def __init__(self):
        self.session = Session()
        self.mode = 'one'
        self.calls = 0
    def detect(self, image, **kwargs):
        self.calls += 1
        if self.mode == 'exception':
            raise RuntimeError('SECRET uploaded bytes C:\\private\\portrait.jpg')
        count = 0 if self.mode == 'none' else 2 if self.mode == 'multiple' else 1
        boxes = np.array([[25,25,175,175,.99]] * count, dtype=np.float32).reshape(-1,5)
        points = np.array([[[70,75],[130,75],[100,100],[75,135],[125,135]]] * count, dtype=np.float32)
        if self.mode == 'clipped':
            boxes[0,0] = -3
        if self.mode == 'over_clipped':
            boxes[0,0] = -40
        if self.mode == 'padded_landmark':
            points[0,0,0] = 20
        return boxes, points

class Recognizer:
    def __init__(self):
        self.session = Session()
        self.calls = 0
    def get_feat(self, image):
        self.calls += 1
        return np.ones((1,512), dtype=np.float32)

class FakeExpression:
    def predict(self, image, **kwargs):
        return Expression(label='happy', confidence=.91)

@pytest.fixture
def context(tmp_path):
    temporary = tempfile.TemporaryDirectory(prefix='phase3_synthetic_', dir=Path(os.environ['LOCALAPPDATA'])/'Temp')
    private = Path(temporary.name)
    settings = Settings(root=tmp_path/'project', private_parent=private)
    bundle = SimpleNamespace(detector=Detector(), recognizer=Recognizer(), config=Config(root=tmp_path),
        align=lambda image, **kw: cv2.resize(image, (112,112)), model_id='synthetic_test_only', embedding_dim=512)
    models = ModelDependencies(face_models=bundle, expression_service=FakeExpression(), load_on_start=False)
    app = create_app(settings, models)
    with TestClient(app, base_url='http://127.0.0.1', client=('127.0.0.1', 50500)) as client:
        yield SimpleNamespace(settings=settings, models=models, bundle=bundle, app=app, client=client, private=private)
    temporary.cleanup()

def photo(seed=1, size=(200,200), kind='PNG'):
    image = np.random.default_rng(seed).integers(60,190,size=(*size,3), dtype=np.uint8)
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format=kind)
    return stream.getvalue()

def upload(seed=1, kind='PNG'):
    return ('never-log-this-private-filename', photo(seed, kind=kind), 'image/png' if kind == 'PNG' else 'image/jpeg')

CONSENT = {'adult_consent': 'true', 'research_opt_in': 'true'}

def comparison_files(n=3, probe_seed=20):
    return [('enrollment_images', upload(i+1)) for i in range(n)] + [('probe', upload(probe_seed))]
