from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime,timezone
import tempfile,json,os
import numpy as np
import pytest
from cryptography.fernet import Fernet
from starlette.testclient import TestClient
from tests.phase3.conftest import context,upload,comparison_files
from faceattend_api.main import create_app
from faceattend_enrollment.auth import OperatorAuth,password_record
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_enrollment.errors import EnrollmentError
from faceattend_enrollment.files import GalleryFiles
from faceattend_enrollment.repository import valid_id
from faceattend_enrollment.service import EnrollmentService

PASSWORD='Fictional test operator password'

class MemoryRepository:
    def __init__(self): self.rows={}
    def get(self,identity_id):
        try: return self.rows[valid_id(identity_id)]
        except KeyError: raise EnrollmentError('ENROLLMENT_NOT_FOUND') from None
    def list(self,page=1):
        if type(page) is not int or not 1<=page<=4: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        return list(self.rows.values())[(page-1)*25:page*25]
    def reserve(self,entry,replace=False):
        if any(r.display_name.casefold()==entry.display_name.casefold() and r.identity_id!=entry.identity_id for r in self.rows.values()):
            raise EnrollmentError('DUPLICATE_DISPLAY_NAME')
        if not replace and len(self.rows)>=100: raise EnrollmentError('GALLERY_FULL')
        self.rows[entry.identity_id]=entry
    def activate(self,identity_id,locator): self.rows[identity_id]=replace(self.get(identity_id),status='active')
    def block(self,identity_id,status):
        row=replace(self.get(identity_id),status=status);self.rows[row.identity_id]=row;return row
    def remove(self,identity_id): return self.rows.pop(identity_id,None) is not None
    def locator_used(self,locator): return any(r.template_locator==locator for r in self.rows.values())
    def expired(self,now): return [r for r in self.rows.values() if r.expires_at<=now]

@pytest.fixture(autouse=True)
def operator_configuration_isolated(monkeypatch):
    monkeypatch.setenv('FACEATTEND_ENROLLMENT_ENABLED','false')
    monkeypatch.setenv('FACEATTEND_DB_TELEMETRY_ENABLED','false')
    for name in ('FACEATTEND_ENROLLMENT_DATABASE_URL','FACEATTEND_ENROLLMENT_KEY_FILE','FACEATTEND_OPERATOR_AUTH_FILE'):
        monkeypatch.delenv(name,raising=False)

@pytest.fixture
def private_root():
    with tempfile.TemporaryDirectory(prefix='faceattend_4b_fixture_',dir=Path(os.environ['LOCALAPPDATA'])/'Temp') as value:
        yield Path(value)

@pytest.fixture
def gallery(private_root):
    root=private_root/'gallery';root.mkdir()
    key=Fernet.generate_key()
    key_file=private_root/'disposable.key';key_file.write_bytes(key)
    repository=MemoryRepository();files=GalleryFiles(root,key=key)
    service=EnrollmentService(repository,files)
    return SimpleNamespace(root=root,key=key,key_file=key_file,files=files,repository=repository,service=service)

@pytest.fixture
def images(): return [np.random.default_rng(i).integers(60,190,(200,200,3),dtype=np.uint8) for i in range(3)]

@pytest.fixture
def operator(context,gallery):
    auth=OperatorAuth(password_record(PASSWORD))
    deps=EnrollmentDependencies(service=gallery.service,auth=auth)
    app=create_app(context.settings,context.models,enrollment_dependencies=deps)
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        result=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
        assert result.status_code==200
        yield SimpleNamespace(client=client,app=app,auth=auth,gallery=gallery,context=context,
                              csrf={'x-faceattend-csrf':result.json()['csrf_token']})

def fields(**updates):
    return dict({'display_name':'Fictional Adult','adult_consent':'true','research_opt_in':'true',
                 'enrollment_consent':'true','retention_days':'30'},**updates)

def files(n=3): return [('enrollment_images',upload(i+1)) for i in range(n)]

def enroll_http(operator,**updates):
    return operator.client.post('/api/v1/demo/enrollments',headers=operator.csrf,data=fields(**updates),files=files())

def enroll_core(gallery,bundle,images,**kwargs):
    return gallery.service.enroll('Fictional Adult',images,bundle,adult_consent=True,research_opt_in=True,enrollment_consent=True,**kwargs)
