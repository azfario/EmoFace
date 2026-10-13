"""Container-only startup/migration adapter; native runtime contracts stay intact."""
from pathlib import Path
import argparse,hashlib,json,os,sys

PRIVATE=Path('/var/lib/faceattend/FaceAttend/phase2/private/phase4b')
ROOT=Path('/app')

def secret(name,maximum=4096):
    path=Path('/run/secrets')/name
    raw=path.read_bytes()
    if not 1<=len(raw)<=maximum:raise ValueError('Invalid secret file.')
    return raw.strip()

def database_url(role,password):
    from sqlalchemy.engine import URL
    if role not in {'faceattend_runtime','faceattend_migrator'}:raise ValueError
    return URL.create('postgresql+psycopg',username=role,password=password.decode('ascii'),host='127.0.0.1',port=5432,database='faceattend_research').render_as_string(hide_password=False)

def prepare():
    from faceattend_enrollment.security import create_private_directory,read_secret
    from faceattend_enrollment.auth import OperatorAuth
    from cryptography.fernet import Fernet
    os.umask(0o077)
    gallery=create_private_directory(PRIVATE/'gallery')
    create_private_directory(PRIVATE)
    folder=create_private_directory(Path('/run/faceattend-private'))
    # Retain only secret fingerprints beside data, never the keys themselves.
    # Refuse implicit rotation, including operator-record changes.
    fingerprints={name:hashlib.sha256(secret(name)).hexdigest() for name in ('gallery_key','operator_auth')}
    marker=PRIVATE/'credential_fingerprints.json'
    if marker.exists():
        if json.loads(read_secret(marker))!=fingerprints:raise ValueError('Explicit key/credential rotation is required.')
    else:
        descriptor=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(descriptor,'w') as stream:json.dump(fingerprints,stream)
    for name,target in [('gallery_key','gallery.key'),('operator_auth','operator.secret')]:
        raw=secret(name)
        path=folder/target
        if not path.exists():
            descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(descriptor,'wb') as stream:stream.write(raw)
        elif read_secret(path)!=raw:raise ValueError('Explicit key/credential rotation is required.')
        if name=='gallery_key':Fernet(raw)
        else:OperatorAuth(json.loads(raw))
    os.environ.update(FACEATTEND_ENROLLMENT_GALLERY_DIR=str(gallery),FACEATTEND_ENROLLMENT_KEY_FILE=str(folder/'gallery.key'),FACEATTEND_OPERATOR_AUTH_FILE=str(folder/'operator.secret'),FACEATTEND_ENROLLMENT_DATABASE_URL=database_url('faceattend_runtime',secret('db_runtime_password')))

def migrate():
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text
    from faceattend_db.session import make_engine
    from faceattend_enrollment.migration import upgrade
    url=database_url('faceattend_migrator',secret('db_owner_password'))
    os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=url
    try:
        command.upgrade(Config(str(ROOT/'alembic.ini')),'head')
        upgrade(url)
        engine=make_engine(url)
        try:
            with engine.begin() as c:
                c.execute(text('GRANT SELECT, INSERT, UPDATE ON public.research_daily_operation_stats TO faceattend_runtime'))
                c.execute(text('GRANT USAGE ON SCHEMA demo TO faceattend_runtime'))
                c.execute(text('GRANT SELECT, INSERT, UPDATE, DELETE ON demo.demo_enrolled_identities TO faceattend_runtime'))
        finally:engine.dispose()
    finally:os.environ.pop('FACEATTEND_MIGRATION_DATABASE_URL',None)
    print('Compose migrations and scoped runtime grants ready.')

def serve():
    # Reject an explicitly disabled device before loading native CUDA libraries.
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1'):
        raise RuntimeError('CUDA explicitly disabled; startup refused.')
    if Path('/netstate/namespace').read_text().strip()!=os.readlink('/proc/self/ns/net'):
        raise RuntimeError('Namespace isolation is not initialized.')
    prepare()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; startup refused.')
    x=torch.ones((32,32),device='cuda');assert (x@x).is_cuda
    import uvicorn
    uvicorn.run('faceattend_api.main:app',host='127.0.0.1',port=8000,workers=1,reload=False,access_log=False,proxy_headers=False,log_level='warning',limit_concurrency=8,timeout_keep_alive=5,h11_max_incomplete_event_size=16384)

def health():
    import urllib.request
    from faceattend_db.session import make_engine
    from faceattend_enrollment.repository import enrollment_runtime_ready
    from faceattend_db.session import sessions
    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/health',headers={'Host':'127.0.0.1:18080'}),timeout=3) as response:
        if response.status!=200:raise RuntimeError
    with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/api/v1/models/info',headers={'Host':'127.0.0.1:18080'}),timeout=3) as response:
        if json.load(response)['model_readiness']!={'face':'ready','expression':'ready'}:raise RuntimeError
    engine=make_engine(database_url('faceattend_runtime',secret('db_runtime_password')))
    try:
        if not enrollment_runtime_ready(sessions(engine)):raise RuntimeError
    finally:engine.dispose()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('serve','migrate','health'));args=parser.parse_args()
    try:globals()[args.command]()
    except Exception:
        print('Compose service unavailable; check local GPU, mounts, permissions and database readiness.',file=sys.stderr)
        sys.exit(1)
