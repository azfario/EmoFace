from types import SimpleNamespace
import pytest
from starlette.testclient import TestClient
from faceattend_api.main import create_app
from faceattend_enrollment.auth import OperatorAuth,password_record
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_candidates.config import CandidateSettings
from faceattend_ui.integration import UISettings
from faceattend_analysis.config import AnalysisSettings
from tests.phase4d.conftest import FakeCrop
from tests.phase4c.conftest import context,images,gallery,private_root,MemoryCatalog,add_reference,upload,CONSENT
from tests.phase4b.conftest import PASSWORD

@pytest.fixture(autouse=True)
def ui_env(monkeypatch):
    for name in ('FACEATTEND_FRONTEND_ENABLED','FACEATTEND_ENROLLMENT_ENABLED','FACEATTEND_CANDIDATES_ENABLED','FACEATTEND_FACE_ANALYSIS_ENABLED','FACEATTEND_DB_TELEMETRY_ENABLED'):monkeypatch.setenv(name,'false')

@pytest.fixture
def ui(context,gallery,monkeypatch):
    import faceattend_candidates.service as candidates
    import faceattend_analysis.router as analysis
    monkeypatch.setattr(candidates,'Catalog',MemoryCatalog);monkeypatch.setattr(analysis,'CropExpression',lambda existing:FakeCrop())
    folder=context.settings.root/'frontend/dist';folder.mkdir(parents=True);(folder/'index.html').write_text('<!doctype html><title>Fictional test UI</title>')
    assets=folder/'assets';assets.mkdir();(assets/'test.js').write_text('/* fictional static test asset */')
    auth=OperatorAuth(password_record(PASSWORD));deps=EnrollmentDependencies(service=gallery.service,auth=auth)
    app=create_app(context.settings,context.models,enrollment_dependencies=deps,candidate_settings=CandidateSettings(enabled=True),
        analysis_settings=AnalysisSettings(enabled=True),ui_settings=UISettings(enabled=True,fixture_mode=True))
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        login=client.post('/api/v1/demo/session',headers={'x-faceattend-login':'1'},json={'password':PASSWORD})
        yield SimpleNamespace(client=client,app=app,context=context,gallery=gallery,auth=auth,
            headers={'x-faceattend-csrf':login.json()['csrf_token'],'x-faceattend-box-opt-in':'1','x-faceattend-candidate-opt-in':'1','x-faceattend-expression-opt-in':'1'})

def preview(ui,**kwargs):return ui.client.post('/api/v1/demo/preview-face',headers=ui.headers,data=CONSENT,files={'image':upload()},**kwargs)
