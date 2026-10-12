from types import SimpleNamespace
import numpy as np
import pytest
from starlette.testclient import TestClient
from faceattend import CLASS_NAMES,MEAN,STD
from faceattend_api.main import create_app
from faceattend_enrollment.auth import OperatorAuth,password_record
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_candidates.config import CandidateSettings
from faceattend_analysis.config import AnalysisSettings
from faceattend_analysis.crop import CROP_VERSION
from faceattend_analysis.service import FaceAnalysis
from faceattend_analysis.expression import WARNING
from tests.phase4c.conftest import context,images,gallery,private_root,MemoryCatalog,add_reference,upload,CONSENT
from tests.phase4b.conftest import PASSWORD

METADATA={'architecture':'mobilenet_v3_small','class_names':CLASS_NAMES,'input_size':[224,224],
    'normalization':{'mean':MEAN,'std':STD}}

class FakeCrop:
    def __init__(self):self.calls=0;self.inputs=[]
    def predict(self,rgb):
        self.calls+=1;self.inputs.append(np.asarray(rgb).copy())
        return {'status':'VISIBLE_EXPRESSION_PREDICTED','label':'neutral','softmax_score':0.2,
            'score_calibrated':False,'softmax_entropy_nats':1.9,'top_score_gap':0.01,
            'not_a_mood_or_diagnosis':True,'device':'CUDA','crop_version':CROP_VERSION,
            'domain_accuracy_validated':False,'warning':WARNING}

@pytest.fixture(autouse=True)
def analysis_env_isolated(monkeypatch):
    monkeypatch.setenv('FACEATTEND_FACE_ANALYSIS_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_CANDIDATES_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_DB_TELEMETRY_ENABLED','false')

@pytest.fixture
def analyzer():return FaceAnalysis(FakeCrop())

@pytest.fixture
def operator(context,gallery,monkeypatch):
    import faceattend_analysis.router as router
    import faceattend_candidates.service as candidates
    fake=FakeCrop();monkeypatch.setattr(router,'CropExpression',lambda existing:fake)
    monkeypatch.setattr(candidates,'Catalog',MemoryCatalog)
    auth=OperatorAuth(password_record(PASSWORD))
    deps=EnrollmentDependencies(service=gallery.service,auth=auth)
    app=create_app(context.settings,context.models,enrollment_dependencies=deps,
        candidate_settings=CandidateSettings(enabled=True),analysis_settings=AnalysisSettings(enabled=True))
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
        yield SimpleNamespace(client=client,app=app,context=context,gallery=gallery,auth=auth,expression=fake,
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-expression-opt-in':'1','x-faceattend-candidate-opt-in':'1'})

def invoke(analyzer,context,images,**kwargs):
    return analyzer.analyze(images[0],context.bundle,adult_consent=True,research_opt_in=True,expression_opt_in=True,**kwargs)

def query(operator,**kwargs):
    return operator.client.post('/api/v1/demo/analyze-face',headers=operator.headers,data=CONSENT,files={'image':upload()},**kwargs)
