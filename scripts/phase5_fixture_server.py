"""Explicit temporary browser-QA fixture; never a retained production setup."""
from pathlib import Path
import argparse,json,os,sys,tempfile,threading,time
from uuid import uuid4
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
import uvicorn
from cryptography.fernet import Fernet
from faceattend_api.config import Settings
from faceattend_api.dependencies import ModelDependencies
from faceattend_api.main import create_app
from faceattend_db.instrumentation import Telemetry
from faceattend_enrollment.auth import OperatorAuth,password_record
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_enrollment.config import EnrollmentSettings
from faceattend_enrollment.files import GalleryFiles
from faceattend_enrollment.service import EnrollmentService
from faceattend_candidates.config import CandidateSettings
from faceattend_analysis.config import AnalysisSettings
from faceattend_ui.integration import UISettings
from tests.phase4b.conftest import PASSWORD,MemoryRepository

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('fictional','live'),required=True)
    parser.add_argument('--port',type=int,default=8015)
    parser.add_argument('--stop-after',type=int,default=900)
    args=parser.parse_args()
    if args.port!=8015 or not 60<=args.stop_after<=7200:raise SystemExit('Only the isolated local fixture port/lifetime is allowed.')
    # Keep earlier coarse attempts when restarting the disposable manual fixture.
    token=uuid4().hex[:12]
    for field in ('observation','cleanup'):
        previous=ROOT/f'reports/phase5/{args.mode}_fixture_{field}.json'
        if previous.exists():
            (previous.parent/f'{args.mode}_fixture_{field}_{token}.json').write_bytes(previous.read_bytes())
    stop_file=ROOT/f'.cache/phase5_stop_{args.mode}'
    stop_file.unlink(missing_ok=True)
    owner_engine=None;config=None;counts={};last={};started=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='faceattend_phase5_fixture_',dir=Path(os.environ['LOCALAPPDATA'])/'Temp') as private:
        parent=Path(private);gallery=parent/'gallery';gallery.mkdir()
        key=parent/'disposable.key';key.write_bytes(Fernet.generate_key())
        auth_file=parent/'disposable.secret';auth_file.write_text(json.dumps(password_record(PASSWORD)))
        if args.mode=='fictional':
            import cv2
            from types import SimpleNamespace
            from face_verification.config import Config
            from tests.phase3.conftest import Detector,Recognizer,FakeExpression
            from tests.phase4d.conftest import FakeCrop
            import faceattend_analysis.router as expression_router
            import faceattend_candidates.service as candidate_module
            from tests.phase4c.conftest import MemoryCatalog
            candidate_module.Catalog=MemoryCatalog;expression_router.CropExpression=lambda existing:FakeCrop()
            models=ModelDependencies(face_models=SimpleNamespace(detector=Detector(),recognizer=Recognizer(),config=Config(root=ROOT),
                align=lambda image,**kwargs:cv2.resize(image,(112,112)),model_id='synthetic_test_only',embedding_dim=512),expression_service=FakeExpression(),load_on_start=False)
            models.device_label='Fictional GPU fixture (simulated)'
            service=EnrollmentService(MemoryRepository(),GalleryFiles(gallery,key=key.read_bytes()))
            dependencies=EnrollmentDependencies(service=service,auth=OperatorAuth(password_record(PASSWORD)))
        else:
            from sqlalchemy import inspect,text
            from sqlalchemy.engine import make_url
            from faceattend_db.session import make_engine
            from faceattend_enrollment.migration import upgrade
            owner=make_url(os.environ['FACEATTEND_TEST_OWNER_URL']);runtime=make_url(os.environ['FACEATTEND_TEST_RUNTIME_URL'])
            if owner.database!='faceattend_phase4_test' or runtime.database!=owner.database or owner.username!='faceattend_test_owner' or runtime.username!='faceattend_test_runtime' or owner.host!='127.0.0.1' or runtime.host!='127.0.0.1':raise SystemExit('Only the guarded disposable database is allowed.')
            owner_engine=make_engine(owner)
            if 'demo' in inspect(owner_engine).get_schema_names():
                with owner_engine.connect() as connection:
                    if 'demo_enrolled_identities' in inspect(owner_engine).get_table_names(schema='demo') and connection.execute(text('SELECT count(*) FROM demo.demo_enrolled_identities')).scalar_one():raise SystemExit('Unexplained existing test enrollments; fixture refused.')
            upgrade(owner)
            with owner_engine.begin() as connection:
                connection.execute(text('GRANT USAGE ON SCHEMA demo TO faceattend_test_runtime'))
                connection.execute(text('GRANT SELECT,INSERT,UPDATE,DELETE ON demo.demo_enrolled_identities TO faceattend_test_runtime'))
            config=EnrollmentSettings(enabled=True,database_url=runtime,gallery_dir=gallery,key_file=key,auth_file=auth_file)
            dependencies=EnrollmentDependencies(config);models=ModelDependencies()
        settings=Settings(root=ROOT,private_parent=parent,port=args.port)
        app=create_app(settings,models,telemetry=Telemetry(),enrollment_dependencies=dependencies,
            candidate_settings=CandidateSettings(enabled=True),analysis_settings=AnalysisSettings(enabled=True),ui_settings=UISettings(enabled=True,fixture_mode=args.mode=='fictional'))
        class Observer:
            def __init__(self,wrapped):self.wrapped=wrapped
            async def __call__(self,scope,receive,send):
                path=scope.get('path','');operation='analyze-face' if path.endswith('/analyze-face') else 'preview-face' if path.endswith('/preview-face') else 'enroll' if path.endswith('/enrollments') and scope.get('method')=='POST' else 'other'
                status=0;body=bytearray()
                async def observe(message):
                    nonlocal status
                    if message['type']=='http.response.start':status=message['status']
                    if operation in ('analyze-face','preview-face') and message['type']=='http.response.body' and len(body)<65536:body.extend(message.get('body',b''))
                    await send(message)
                try:await self.wrapped(scope,receive,observe)
                finally:
                    if operation!='other':
                        counts[f'{operation}_{status}']=counts.get(f'{operation}_{status}',0)+1
                        if status==200 and body:
                            try:
                                value=json.loads(body);last.clear();last.update(operation=operation,status_code=status,face_count=value.get('face_count'),
                                    expression_prediction_present=any('visible_expression_prediction' in face for face in value.get('faces',[])),
                                    opt_in_outline_present=any('face_box' in face for face in value.get('faces',[])),verified=value.get('verified'),threshold=value.get('threshold'))
                            except Exception:pass
                    body.clear()
                    (ROOT/f'reports/phase5/{args.mode}_fixture_observation.json').write_text(json.dumps({'mode':args.mode,'counts':counts,'last_analysis':last,
                        'source_names_paths_images_labels_scores_saved':False},indent=2),encoding='utf-8')
        server=uvicorn.Server(uvicorn.Config(Observer(app),host='127.0.0.1',port=args.port,workers=1,access_log=False,proxy_headers=False,log_level='warning'))
        def watchdog():
            while not server.should_exit:
                if stop_file.exists() or time.monotonic()-started>=args.stop_after:server.should_exit=True;break
                time.sleep(.25)
        threading.Thread(target=watchdog,daemon=True).start()
        print(json.dumps({'fixture_mode':args.mode,'url':f'http://127.0.0.1:{args.port}/app/','production_setup_created':False}),flush=True)
        try:server.run()
        finally:
            if config is not None:
                cleanup=EnrollmentDependencies(config);cleanup.start()
                try:
                    service=cleanup.require_service()
                    for page in range(1,5):
                        rows=service.repository.list(1)
                        if not rows:break
                        for row in rows:service.delete(row.identity_id)
                finally:cleanup.close()
                from sqlalchemy import text
                with owner_engine.begin() as connection:connection.execute(text('DROP SCHEMA IF EXISTS demo CASCADE'))
                owner_engine.dispose()
            stop_file.unlink(missing_ok=True)
    (ROOT/f'reports/phase5/{args.mode}_fixture_cleanup.json').write_text(json.dumps({'mode':args.mode,'temporary_directory_removed':not parent.exists(),
        'guarded_test_schema_removed':args.mode=='live','production_setup_created':False},indent=2),encoding='utf-8')
if __name__=='__main__':main()
