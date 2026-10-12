"""Opt-in real PostgreSQL semantics; isolated test DB only, no SQLite."""
import os, time
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from alembic import command
from alembic.config import Config
from faceattend_db.config import DBSettings
from faceattend_db.models import MAX_BIGINT
from faceattend_db.session import make_engine, sessions, runtime_ready, DatabaseUnavailable
from faceattend_db.repository import Repository
from faceattend_db.instrumentation import Telemetry
from tests.phase3.conftest import CONSENT, upload, comparison_files

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4_POSTGRES_TESTS')!='1',reason='Explicit isolated PostgreSQL opt-in required.')
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def database():
    owner=make_url(os.environ['FACEATTEND_TEST_OWNER_URL'])
    runtime=make_url(os.environ['FACEATTEND_TEST_RUNTIME_URL'])
    # These tests execute destructive downgrade; strict project test name/roles.
    if (owner.database!='faceattend_phase4_test' or runtime.database!=owner.database
        or owner.username!='faceattend_test_owner' or runtime.username!='faceattend_test_runtime'
        or owner.host!='127.0.0.1' or runtime.host!='127.0.0.1' or owner.port!=runtime.port):
        pytest.fail('Isolated project test database configuration is required.')
    previous=os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL')
    os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=owner.render_as_string(hide_password=False)
    configuration=Config(str(ROOT/'alembic.ini'))
    engine=make_engine(owner); runtime_engine=make_engine(runtime)
    try:
        command.upgrade(configuration,'head')
        assert set(inspect(engine).get_table_names())=={'alembic_version','research_daily_operation_stats'}
        command.downgrade(configuration,'base')
        assert inspect(engine).get_table_names()==['alembic_version']
        command.upgrade(configuration,'head')
        with engine.begin() as c:
            c.execute(text('GRANT SELECT, INSERT, UPDATE ON research_daily_operation_stats TO faceattend_test_runtime'))
        yield SimpleNamespace(owner=engine,runtime=runtime_engine,url=runtime,owner_url=owner,config=configuration,
                              factory=sessions(runtime_engine),repository=Repository(sessions(runtime_engine)))
    finally:
        with engine.begin() as c: c.execute(text('DELETE FROM research_daily_operation_stats'))
        runtime_engine.dispose(); engine.dispose()
        if previous is None: os.environ.pop('FACEATTEND_MIGRATION_DATABASE_URL',None)
        else: os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=previous

@pytest.fixture
def db(database):
    with database.owner.begin() as c: c.execute(text('DELETE FROM research_daily_operation_stats'))
    return database

def test_upgrade_downgrade_and_exact_schema(db):
    inspector=inspect(db.owner)
    assert {c['name'] for c in inspector.get_columns('research_daily_operation_stats')}=={'day_utc','operation','outcome','request_count','total_processing_ms'}
    assert all(not c['nullable'] for c in inspector.get_columns('research_daily_operation_stats'))
    assert inspector.get_pk_constraint('research_daily_operation_stats')['constrained_columns']==['day_utc','operation','outcome']
    assert len(inspector.get_check_constraints('research_daily_operation_stats'))==4
    assert not inspector.get_indexes('research_daily_operation_stats')
    with db.owner.connect() as c:
        assert c.execute(text('SELECT version_num FROM alembic_version')).scalar_one()=='0001_daily_stats'
        assert c.execute(text('SHOW listen_addresses')).scalar_one()=='127.0.0.1'

@pytest.mark.parametrize('operation,outcome,count,ms',[
    ('face_detect','happy',1,1),('students','completed',1,1),('face_detect','completed',-1,1),
    ('face_detect','completed',1,-1),('face_detect','completed',MAX_BIGINT+1,1),
    ('face_detect','completed',1,MAX_BIGINT+1)])
def test_database_constraints_enforced(db,operation,outcome,count,ms):
    with pytest.raises(Exception):
        with db.owner.begin() as c:
            c.execute(text('INSERT INTO research_daily_operation_stats VALUES (:day,:op,:outcome,:count,:ms)'),
                      dict(day=date(1900,1,1),op=operation,outcome=outcome,count=count,ms=ms))
    assert db.repository.get_aggregate_summary(date(1900,1,1),date(1900,1,1))==[]

def test_concurrent_upsert_no_lost_updates(db):
    def run(_): db.repository.increment_daily('face_detect','completed',20,date(1900,1,1))
    with ThreadPoolExecutor(max_workers=8) as pool: list(pool.map(run,range(64)))
    assert db.repository.get_aggregate_summary(date(1900,1,1),date(1900,1,1))==[
        {'day_utc':'1900-01-01','operation':'face_detect','outcome':'completed','request_count':64,'total_processing_ms':1280}]

def test_bigint_overflow_transaction_rolls_back_then_reusable(db):
    with db.owner.begin() as c:
        c.execute(text("INSERT INTO research_daily_operation_stats VALUES ('1900-01-01','face_detect','completed',:max,:max)"),{'max':MAX_BIGINT})
    with pytest.raises(DatabaseUnavailable): db.repository.increment_daily('face_detect','completed',10,date(1900,1,1))
    row=db.repository.get_aggregate_summary(date(1900,1,1),date(1900,1,1))[0]
    assert row['request_count']==MAX_BIGINT and row['total_processing_ms']==MAX_BIGINT
    db.repository.increment_daily('embedding_info','completed',10,date(1900,1,1))
    assert db.runtime.pool.checkedout()==0

def test_retention_boundary_and_unrelated_data(db):
    cutoff=date(1900,2,1)
    for day in (cutoff-timedelta(days=1),cutoff,cutoff+timedelta(days=1)):
        db.repository.increment_daily('face_detect','completed',10,day)
    with db.owner.begin() as c:
        c.execute(text('CREATE TABLE unrelated_synthetic_control (value INTEGER)'))
        c.execute(text('INSERT INTO unrelated_synthetic_control VALUES (7)'))
    try:
        assert Repository(sessions(db.owner)).purge_old_aggregates(cutoff)==1
        assert len(db.repository.get_aggregate_summary(cutoff-timedelta(days=1),cutoff+timedelta(days=1)))==2
        with db.owner.connect() as c: assert c.execute(text('SELECT value FROM unrelated_synthetic_control')).scalar_one()==7
    finally:
        with db.owner.begin() as c: c.execute(text('DROP TABLE unrelated_synthetic_control'))

def test_runtime_privileges_and_no_unrelated_database_access(db):
    assert runtime_ready(db.factory)
    assert not runtime_ready(sessions(db.owner))  # Owner secrets must not run the app.
    for statement in ('CREATE TABLE forbidden (x INT)','DELETE FROM research_daily_operation_stats','TRUNCATE research_daily_operation_stats','SELECT * FROM alembic_version'):
        with pytest.raises(Exception):
            with db.runtime.begin() as c: c.execute(text(statement))
    for name in ('postgres','faceattend_research'):
        engine=make_engine(db.url.set(database=name))
        try:
            with pytest.raises(Exception):
                with engine.connect(): pass
        finally: engine.dispose()

def test_statement_timeout_and_connection_return(db):
    started=time.monotonic()
    with pytest.raises(Exception):
        with db.runtime.begin() as c: c.execute(text('SELECT pg_sleep(3)'))
    assert time.monotonic()-started<3 and db.runtime.pool.checkedout()==0
    assert runtime_ready(db.factory)

@pytest.mark.parametrize('path',['face/detect','face/embedding-info','expression/predict','face/compare-diagnostic'])
def test_consented_api_persists_safe_rows_with_unchanged_output(db,context,path):
    def request():
        if path.endswith('compare-diagnostic'):
            return context.client.post('/api/v1/'+path,data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files())
        return context.client.post('/api/v1/'+path,data=CONSENT,files={'image':upload()})
    expected=request().json()
    service=Telemetry(DBSettings(enabled=True,url=db.url))
    service.start(); context.app.state.telemetry=service
    try:
        assert service.status=='ready' and context.client.get('/health/db').json()=={'status':'ready'}
        response=request()
        assert response.status_code==200 and response.json()==expected
        with db.owner.connect() as c:
            row=c.execute(text('SELECT * FROM research_daily_operation_stats')).mappings().one()
        assert len(row)==5 and row['request_count']==1 and row['outcome']=='completed'
        assert row['total_processing_ms']%10==0
        assert 'volunteer_demo' not in repr(row) and 'happy' not in repr(row)
    finally: service.close()

def test_api_refusal_never_creates_record(db,context):
    service=Telemetry(DBSettings(enabled=True,url=db.url)); service.start(); context.app.state.telemetry=service
    try:
        for consent in ({}, {'adult_consent':'false','research_opt_in':'true'}, {'adult_consent':'true','research_opt_in':'false'}):
            assert context.client.post('/api/v1/face/detect',data=consent,files={'image':upload()}).status_code>=400
        with db.owner.connect() as c: assert c.execute(text('SELECT count(*) FROM research_daily_operation_stats')).scalar_one()==0
    finally: service.close()

def test_outage_does_not_change_compare_or_health(db,context):
    import socket
    with socket.socket() as s:
        s.bind(('127.0.0.1',0)); port=s.getsockname()[1]
    service=Telemetry(DBSettings(enabled=True,url=db.url.set(port=port)))
    service.start(); context.app.state.telemetry=service
    try:
        assert service.status=='unavailable'
        response=context.client.post('/api/v1/face/compare-diagnostic',data=dict(CONSENT,claimed_id='volunteer_demo'),files=comparison_files())
        assert response.status_code==200 and response.json()['operational_verification']=={'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED'}
        assert context.client.get('/health').status_code==200 and context.client.get('/health/db').status_code==503
    finally: service.close()

def test_app_factory_starts_and_closes_actual_db_lifecycle(db,context):
    from starlette.testclient import TestClient
    from faceattend_api.main import create_app
    service=Telemetry(DBSettings(enabled=True,url=db.url))
    app=create_app(context.settings,context.models,telemetry=service)
    with TestClient(app,base_url='http://127.0.0.1',client=('127.0.0.1',1)) as client:
        assert service.status=='ready'
        assert client.get('/health/db').json()=={'status':'ready'}
        assert client.post('/api/v1/face/detect',data=CONSENT,files={'image':upload()}).status_code==200
    assert service.repository is None and service.engine is None
    with db.owner.connect() as c:
        assert c.execute(text('SELECT sum(request_count) FROM research_daily_operation_stats')).scalar_one()==1
