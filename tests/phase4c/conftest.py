from datetime import datetime,timezone,timedelta
from uuid import uuid4,UUID
from types import SimpleNamespace
import numpy as np
import pytest
from starlette.testclient import TestClient
from tests.phase3.conftest import context,photo,upload,CONSENT
from tests.phase4b.conftest import private_root,gallery,images,PASSWORD
from faceattend_enrollment.repository import Record
from faceattend_enrollment.models import PURPOSE
from faceattend_enrollment.auth import OperatorAuth,password_record
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_api.main import create_app
from faceattend_candidates.config import CandidateSettings
from faceattend_candidates.service import CandidateService

class MemoryCatalog:
    def __init__(self,repository):self.repository=repository;self.snapshots=0;self.rechecks=0
    def snapshot(self):self.snapshots+=1;return list(self.repository.rows.values())
    def revalidate(self,entries):
        self.rechecks+=1
        return [r for r in entries if self.repository.rows.get(r.identity_id)==r]

@pytest.fixture(autouse=True)
def candidate_config_isolated(monkeypatch):
    monkeypatch.setenv('FACEATTEND_CANDIDATES_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_DB_TELEMETRY_ENABLED','false')

def add_reference(gallery,vector=None,*,name='Fictional Adult',identity=None,model_id='synthetic_test_only',status='active',expires=None,locator=None):
    now=datetime.now(timezone.utc)
    row=Record(identity or uuid4(),name,locator or uuid4().hex,now,PURPOSE,now,
        expires or now+timedelta(days=7),status,model_id)
    gallery.files.store().save(gallery.files.user(row.template_locator),np.ones(512) if vector is None else vector,
        model_id=model_id,sample_hashes=['synthetic_only'],consent={'adult':True,'identity_verification':True,'recorded_at':now.isoformat()})
    gallery.repository.rows[row.identity_id]=row
    return row

@pytest.fixture
def search(gallery):return CandidateService(gallery.service,catalog=MemoryCatalog(gallery.repository))

@pytest.fixture
def candidate_operator(context,gallery,monkeypatch):
    import faceattend_candidates.service as service_module
    monkeypatch.setattr(service_module,'Catalog',MemoryCatalog)
    auth=OperatorAuth(password_record(PASSWORD))
    deps=EnrollmentDependencies(service=gallery.service,auth=auth)
    app=create_app(context.settings,context.models,enrollment_dependencies=deps,candidate_settings=CandidateSettings(enabled=True))
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
        headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-candidate-opt-in':'1'}
        yield SimpleNamespace(client=client,app=app,auth=auth,headers=headers,gallery=gallery,context=context)

def invoke(search,context,images,**kwargs):
    return search.search(images[0],context.bundle,adult_consent=True,research_opt_in=True,search_opt_in=True,**kwargs)

def query(operator,**kwargs):
    return operator.client.post('/api/v1/demo/candidates',headers=operator.headers,data=CONSENT,files={'image':upload()},**kwargs)
