"""Explicit opt-in QA Compose checks; never target the native PostgreSQL cluster.

Set EMOFACE_PHASE6_LIVE=1 and COMPOSE_PROJECT_NAME=emoface_phase6qa.
Supply EMOFACE_MODELS_DIR/EMOFACE_SECRETS_DIR as for Compose. Adult-image
functionality additionally requires a private manifest supplied by the operator,
with adult_consent/persistent_enrollment_consent and references/probe file paths.
No fixture passwords, private input paths or responses are written to receipts.
"""
from pathlib import Path
import json, os, socket, subprocess, time
import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
PROJECT = 'emoface_phase6qa'
BASE = 'http://127.0.0.1:18080'
pytestmark = pytest.mark.skipif(os.environ.get('EMOFACE_PHASE6_LIVE') != '1', reason='Explicit isolated Compose QA opt-in required.')


def compose(*args, code=None):
    if os.environ.get('COMPOSE_PROJECT_NAME') != PROJECT:
        raise RuntimeError('Only the isolated Phase 6 QA project is permitted.')
    result = subprocess.run(['docker', 'compose', '-p', PROJECT, *args], cwd=ROOT,
                            input=code, capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise AssertionError('QA Compose command failed; raw diagnostics withheld.')
    return result.stdout


def inside(code):
    return json.loads(compose('exec', '-T', 'api', 'python', '-', code=code))


def ready():
    deadline = time.monotonic() + 150
    while time.monotonic() < deadline:
        try:
            with httpx.Client(base_url=BASE, trust_env=False, timeout=5) as client:
                r = client.get('/health')
                if r.status_code == 200: return
        except httpx.HTTPError: pass
        time.sleep(1)
    raise AssertionError('QA models did not become ready.')


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE, trust_env=False, timeout=90) as client:
        yield client


@pytest.fixture(scope='session', autouse=True)
def initialized_models():
    ready()


def login(client):
    required=client.get('/api/v1/demo/ui-status').json().get('login_required',True)
    result = client.post('/api/v1/demo/session', headers={'x-faceattend-login':'1'},
                         json={'password':'Fictional Phase 6 operator password'} if required else {})
    assert result.status_code == 200, 'QA login failed; response withheld.'
    return {'x-faceattend-csrf':result.json()['csrf_token']}


def test_proxy_local_headers_static_assets_and_authorization(client):
    assert client.get('/health').status_code == 200
    assert client.get('/api/v1/models/info').json()['model_readiness'] == {'face':'ready','expression':'ready'}
    page = client.get('/app/')
    assert page.status_code == 200 and 'no-store' in page.headers['cache-control']
    assert "connect-src 'self'" in page.headers['content-security-policy']
    assert page.headers['x-frame-options'] == 'DENY'
    for headers in ({'Host':'external.invalid'}, {'Origin':'https://external.invalid'}, {'Origin':'null'}):
        assert client.get('/health', headers=headers).status_code == 403
    assert client.get('/health', headers={'Origin':BASE,'X-Forwarded-For':'8.8.8.8','Forwarded':'for=8.8.8.8'}).status_code == 200
    assert client.get('/api/v1/demo/enrollments').status_code == 401
    assert client.post('/api/v1/demo/session', json={'password':'irrelevant'}).status_code == 403
    headers = login(client)
    assert client.delete('/api/v1/demo/session').status_code == 403
    assert client.delete('/api/v1/demo/session', headers=headers).status_code == 200
    assert client.get('/api/v1/demo/enrollments').status_code == 401


def test_namespace_egress_denied_and_mounts_private():
    value = inside("""
import json,os,socket
def blocked(host,port):
    s=socket.socket();s.settimeout(1)
    try:s.connect((host,port));return False
    except OSError:return True
    finally:s.close()
print(json.dumps({'tcp_denied':blocked('1.1.1.1',443),'dns_denied':blocked('127.0.0.11',53),
 'namespace_matches':open('/netstate/namespace').read().strip()==os.readlink('/proc/self/ns/net'),'uid':os.getuid()}))
""")
    assert value == {'tcp_denied':True,'dns_denied':True,'namespace_matches':True,'uid':10001}
    ids = compose('ps','-aq').split()
    inspected = subprocess.run(['docker','inspect',*ids],capture_output=True,text=True,check=True)
    services = {v['Config']['Labels']['com.docker.compose.service']:v for v in json.loads(inspected.stdout)}
    for name, value in services.items():
        ports = value['HostConfig']['PortBindings'] or {}
        if name == 'frontend': assert ports == {'8080/tcp':[{'HostIp':'127.0.0.1','HostPort':'18080'}]}
        else: assert ports == {}
        devices = value['HostConfig']['DeviceRequests'] or []
        assert bool(devices) == (name == 'api')
    model = next(v for v in services['api']['Mounts'] if v['Destination']=='/app/models')
    assert model['RW'] is False
    assert services['api']['HostConfig']['ReadonlyRootfs']


def test_isolated_database_migrations_roles_and_no_prediction_columns():
    code = """
import json
from docker.runtime import database_url,secret
from sqlalchemy import text,inspect
from faceattend_db.session import make_engine
e=make_engine(database_url('faceattend_migrator',secret('db_owner_password')))
with e.connect() as c:
    versions=[c.execute(text('SELECT version_num FROM '+s+'.alembic_version')).scalar_one() for s in ('public','demo')]
    denied=[not c.execute(text(q)).scalar_one() for q in [
      "SELECT has_schema_privilege('faceattend_runtime','demo','CREATE')",
      "SELECT has_table_privilege('faceattend_runtime','demo.demo_enrolled_identities','TRUNCATE')",
      "SELECT has_table_privilege('faceattend_runtime','public.research_daily_operation_stats','DELETE')"]]
    cols=[v['name'] for v in inspect(c).get_columns('demo_enrolled_identities',schema='demo')]
    print(json.dumps({'versions':versions,'restricted':all(denied),'no_predictions':not any(x in n for n in cols for x in ('emotion','expression','embedding','image','score')),'port':c.execute(text('SHOW port')).scalar_one()}))
e.dispose()
"""
    value = json.loads(compose('run','--rm','--no-deps','migrate','python','-',code=code))
    assert value == {'versions':['0001_daily_stats','0002_consented_enrollment'],'restricted':True,'no_predictions':True,'port':'5432'}


def test_database_outage_fails_closed_and_recovers(client):
    headers = login(client)
    compose('stop','db')
    try:
        assert client.get('/health').status_code == 200  # Original GPU-only health remains independent.
        status = client.get('/api/v1/demo/ui-status').json()
        assert status['registration_database'] == 'unavailable'
        assert client.get('/api/v1/demo/enrollments').status_code == 503
    finally: compose('start','db')
    deadline = time.monotonic()+30
    while time.monotonic()<deadline:
        if client.get('/api/v1/demo/ui-status').json()['registration_database']=='ready':break
        time.sleep(1)
    assert client.get('/api/v1/demo/enrollments').status_code == 200
    assert client.delete('/api/v1/demo/session',headers=headers).status_code == 200


def test_consented_enrollment_cuda_analysis_restart_and_delete(client,record_property):
    manifest_path = os.environ.get('EMOFACE_PHASE6_INPUT_MANIFEST')
    if not manifest_path: pytest.skip('Explicit consenting-adult private manifest required.')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    assert manifest.get('adult_consent') is True and manifest.get('persistent_enrollment_consent') is True
    inputs = [Path(p).read_bytes() for p in manifest['references']]
    probe = Path(manifest['probe']).read_bytes()
    headers = login(client)
    consent = {'adult_consent':'true','research_opt_in':'true'}
    accepted = []
    for data in inputs:
        result = client.post('/api/v1/face/detect', data=consent, files={'image':('input',data,'image/jpeg')})
        if result.status_code == 200: accepted.append(data)
    assert len(accepted)>=3, 'Three unchanged-quality references required; private details withheld.'
    result = client.post('/api/v1/demo/enrollments',headers=headers,
        data={**consent,'display_name':'Fictional Compose QA reference','enrollment_consent':'true','retention_days':'1'},
        files=[('enrollment_images',('reference',v,'image/jpeg')) for v in accepted[:3]])
    assert result.status_code == 200, 'Enrollment failed; private response withheld.'
    identity = result.json()['identity_id']
    try:
        assert result.json()['verified'] is False and result.json()['threshold'] is None
        assert 'embedding' not in result.text and 'template_locator' not in result.text
        encrypted = inside("""
import json
from pathlib import Path
p=Path('/var/lib/faceattend/FaceAttend/phase2/private/phase4b/gallery')
files=list((p/'templates').glob('*.fernet'))
print(json.dumps({'one_encrypted_reference':len(files)==1 and files[0].read_bytes().startswith(b'gAAAA'),
'no_photos':not any(v.suffix.lower() in ('.jpg','.png','.jpeg') for v in p.rglob('*')),
'private_modes':p.stat().st_mode & 0o077 == 0}))
""")
        assert all(encrypted.values())
        compose('restart','api');ready()
        assert client.get('/api/v1/demo/enrollments').status_code == 401
        headers = login(client)
        assert len(client.get('/api/v1/demo/enrollments').json()['enrollments'])==1
        assert client.post(f'/api/v1/demo/enrollments/{identity}/inspect-reference',headers=headers).status_code == 200
        opted = {**headers,'x-faceattend-candidate-opt-in':'1','x-faceattend-expression-opt-in':'1','x-faceattend-box-opt-in':'1'}
        response = client.post('/api/v1/demo/analyze-face?include_candidates=true&include_face_box=true',headers=opted,
                               data=consent,files={'image':('probe',probe,'image/jpeg')})
        assert response.status_code == 200, 'Combined CUDA analysis failed; response withheld.'
        body = response.json()
        assert body['verified'] is False and body['threshold'] is None
        assert body['faces'][0]['identity_diagnostic']['candidate_count']==1
        prediction = body['faces'][0]['visible_expression_prediction']
        assert prediction['score_calibrated'] is False and prediction['domain_accuracy_validated'] is False
        elapsed=[]
        for _ in range(3):
            start=time.perf_counter()
            response=client.post('/api/v1/demo/analyze-face?include_face_box=true',headers=opted,
                                 data=consent,files={'image':('probe',probe,'image/jpeg')})
            assert response.status_code==200, 'CUDA request failed; response withheld.'
            elapsed.append(round((time.perf_counter()-start)*1000,2))
        record_property('sequential_http_image_ms',json.dumps(elapsed))
        record_property('timing_scope','Local HTTP image uploads, not camera FPS or accuracy.')
        # Recreate the namespace through non-destructive down/up, preserving only QA volumes.
        compose('down');compose('up','-d');ready()
        headers = login(client)
        assert len(client.get('/api/v1/demo/enrollments').json()['enrollments'])==1
        assert client.post(f'/api/v1/demo/enrollments/{identity}/inspect-reference',headers=headers).status_code==200
    finally:
        ready();headers = login(client)
        assert client.delete(f'/api/v1/demo/enrollments/{identity}',headers=headers).status_code == 200
        inputs.clear();accepted.clear();probe = b''
    assert client.get('/api/v1/demo/enrollments').json()['enrollments']==[]
    assert inside("from pathlib import Path;import json;print(json.dumps({'empty':not list(Path('/var/lib/faceattend/FaceAttend/phase2/private/phase4b/gallery/templates').iterdir())}))")['empty']
    assert client.delete('/api/v1/demo/session',headers=headers).status_code==200
