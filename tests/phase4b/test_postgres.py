"""Explicit real PostgreSQL tests in the known disposable DB, demo schema only."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4,UUID
from concurrent.futures import ThreadPoolExecutor
import os
import pytest
from sqlalchemy import inspect,text
from sqlalchemy.engine import make_url
from alembic import command
from alembic.config import Config
from faceattend_db.session import make_engine,sessions,runtime_ready
from faceattend_enrollment.migration import upgrade
from faceattend_enrollment.repository import EnrollmentRepository,enrollment_runtime_ready,Record
from faceattend_enrollment.models import PURPOSE
from faceattend_enrollment.errors import EnrollmentError
from faceattend_enrollment.service import EnrollmentService
from faceattend_enrollment.files import GalleryFiles
from .conftest import enroll_core

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4B_POSTGRES_TESTS')!='1',reason='Explicit isolated PostgreSQL opt-in required.')
ROOT=Path(__file__).resolve().parents[2]

@pytest.fixture(scope='module')
def live_database():
    owner=make_url(os.environ['FACEATTEND_TEST_OWNER_URL']);runtime=make_url(os.environ['FACEATTEND_TEST_RUNTIME_URL'])
    if owner.database!='faceattend_phase4_test' or owner.username!='faceattend_test_owner' or runtime.username!='faceattend_test_runtime' or owner.host!='127.0.0.1' or runtime.host!='127.0.0.1' or owner.database!=runtime.database:
        pytest.fail('Only the established isolated test database/roles are permitted.')
    owner_engine=make_engine(owner);runtime_engine=make_engine(runtime)
    before=set(inspect(owner_engine).get_table_names(schema='public'))
    if 'demo' in inspect(owner_engine).get_schema_names():
        # Never erase unexplained pre-existing test registrations.
        with owner_engine.connect() as c:
            if 'demo_enrolled_identities' in inspect(owner_engine).get_table_names(schema='demo') and c.execute(text('SELECT count(*) FROM demo.demo_enrolled_identities')).scalar_one():
                pytest.fail('Existing test enrollment data must be explicitly cleared by its operator.')
    previous=os.environ.get('FACEATTEND_MIGRATION_DATABASE_URL')
    os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=owner.render_as_string(hide_password=False)
    configuration=Config(str(ROOT/'alembic-phase4b.ini'))
    try:
        upgrade(owner)
        command.downgrade(configuration,'0001_daily_stats')
        assert inspect(owner_engine).get_table_names(schema='demo')==['alembic_version']
        assert set(inspect(owner_engine).get_table_names(schema='public'))==before
        command.upgrade(configuration,'head')
        with owner_engine.begin() as c:
            c.execute(text('GRANT USAGE ON SCHEMA demo TO faceattend_test_runtime'))
            c.execute(text('GRANT SELECT, INSERT, UPDATE, DELETE ON demo.demo_enrolled_identities TO faceattend_test_runtime'))
        yield SimpleNamespace(owner=owner_engine,runtime=runtime_engine,owner_url=owner,runtime_url=runtime,
                              repository=EnrollmentRepository(sessions(runtime_engine)),factory=sessions(runtime_engine))
    finally:
        # Strict guarded disposable DB only. No normal DB downgrade/drop.
        with owner_engine.begin() as c:c.execute(text('DROP SCHEMA IF EXISTS demo CASCADE'))
        runtime_engine.dispose();owner_engine.dispose()
        if previous is None:os.environ.pop('FACEATTEND_MIGRATION_DATABASE_URL',None)
        else:os.environ['FACEATTEND_MIGRATION_DATABASE_URL']=previous

@pytest.fixture
def db(live_database):
    with live_database.owner.begin() as c:c.execute(text('DELETE FROM demo.demo_enrolled_identities'))
    return live_database

def entry(name='Fictional Adult'):
    now=datetime.now(timezone.utc)
    return Record(uuid4(),name,uuid4().hex,now,PURPOSE,now,now+timedelta(days=30),'pending','synthetic_test_only')

def test_migration_exact_schema_and_original_history(db):
    inspector=inspect(db.owner)
    assert set(inspector.get_table_names(schema='public'))=={'alembic_version','research_daily_operation_stats'}
    assert set(inspector.get_table_names(schema='demo'))=={'alembic_version','demo_enrolled_identities'}
    columns=inspector.get_columns('demo_enrolled_identities',schema='demo')
    assert len(columns)==9 and all(not col['nullable'] for col in columns)
    assert len(inspector.get_check_constraints('demo_enrolled_identities',schema='demo'))==6
    with db.owner.connect() as c:
        assert c.execute(text('SELECT version_num FROM public.alembic_version')).scalar_one()=='0001_daily_stats'
        assert c.execute(text('SELECT version_num FROM demo.alembic_version')).scalar_one()=='0002_consented_enrollment'
        assert c.execute(text('SHOW log_error_verbosity')).scalar_one()=='terse'

@pytest.mark.parametrize('kind',['status','purpose','locator','name','expiry','fingerprint'])
def test_database_constraints_enforced(db,kind):
    values=entry()
    changes={'status':{'status':'verified'},'purpose':{'purpose_version':'attendance'},'locator':{'template_locator':'../private'},
             'name':{'display_name':'Invalid/Name'},'expiry':{'expires_at':values.consent_recorded_at+timedelta(days=31)},
             'fingerprint':{'model_policy_fingerprint':'private/path'}}
    bad=replace(values,**changes[kind])
    with pytest.raises(EnrollmentError):db.repository.reserve(bad)
    assert db.repository.list()==[] and db.runtime.pool.checkedout()==0

def test_case_insensitive_name_and_uuid_uniqueness(db):
    value=entry();db.repository.reserve(value)
    with pytest.raises(EnrollmentError) as e:db.repository.reserve(entry('fictional adult'))
    assert e.value.code=='DUPLICATE_DISPLAY_NAME' and len(db.repository.list())==1
    assert db.repository.get(value.identity_id).identity_id==value.identity_id

def test_runtime_least_privilege_and_original_aggregate_permissions(db):
    assert enrollment_runtime_ready(db.factory) and runtime_ready(db.factory)
    assert not enrollment_runtime_ready(sessions(db.owner))
    for sql in ('CREATE TABLE demo.forbidden (x INT)','TRUNCATE demo.demo_enrolled_identities',
                'SELECT * FROM demo.alembic_version','DELETE FROM public.research_daily_operation_stats'):
        with pytest.raises(Exception):
            with db.runtime.begin() as c:c.execute(text(sql))

def test_transaction_rolls_back_duplicate_then_reuses_session(db):
    db.repository.reserve(entry())
    with pytest.raises(EnrollmentError):db.repository.reserve(entry())
    db.repository.reserve(entry('Fictional Other'))
    assert len(db.repository.list())==2 and db.runtime.pool.checkedout()==0

def test_concurrent_name_reservation_never_duplicate(db):
    def run(_):
        try:db.repository.reserve(entry());return 'created'
        except EnrollmentError:return 'rejected'
    with ThreadPoolExecutor(max_workers=4) as pool:outcomes=list(pool.map(run,range(4)))
    assert outcomes.count('created')==1 and len(db.repository.list())==1

def test_gallery_capacity_bound(db):
    for n in range(100):db.repository.reserve(entry(f'Fictional {n}'))
    with pytest.raises(EnrollmentError) as e:db.repository.reserve(entry('Fictional overflow'))
    assert e.value.code=='GALLERY_FULL'
    assert len(db.repository.list(4))==25

def test_real_crypto_db_restart_revoke_reenroll_and_delete(db,gallery,context,images):
    service=EnrollmentService(db.repository,gallery.files)
    value=service.enroll('Fictional Adult',images,context.bundle,adult_consent=True,research_opt_in=True,enrollment_consent=True)
    identity=UUID(value['identity_id'])
    restart_engine=make_engine(db.runtime_url)
    try:
        restarted=EnrollmentService(EnrollmentRepository(sessions(restart_engine)),GalleryFiles(gallery.root,key=gallery.key_file.read_bytes()))
        assert restarted.inspect(identity,context.bundle.model_id)['status']=='active'
        assert restarted.revoke(identity)['status']=='revoked'
        replaced=restarted.enroll('Fictional Adult',images,context.bundle,adult_consent=True,research_opt_in=True,enrollment_consent=True,identity_id=identity)
        assert replaced['identity_id']==str(identity)
        assert restarted.delete(identity)=={'status':'deleted'}
        assert db.repository.list()==[] and not list(gallery.files.templates.iterdir())
    finally:restart_engine.dispose()

def test_utc_metadata_and_no_vector_path_or_sample_hash_fields(db):
    value=entry();db.repository.reserve(value)
    loaded=db.repository.get(value.identity_id)
    assert loaded.consent_recorded_at.utcoffset()==timedelta(0)
    public=loaded.public()
    assert 'template_locator' not in public and 'embedding' not in public and 'sample_hashes' not in public
    assert loaded.template_locator not in str(public)

def test_retention_removes_only_expired_metadata(db,gallery,context,images):
    service=EnrollmentService(db.repository,gallery.files)
    result=service.enroll('Fictional Adult',images,context.bundle,adult_consent=True,research_opt_in=True,enrollment_consent=True)
    identity=UUID(result['identity_id']);record=db.repository.get(identity)
    db.repository.reserve(entry('Fictional current'))
    service.clock=lambda:record.expires_at
    # Move the second synthetic record's expiry safely beyond that cutoff.
    other=[x for x in db.repository.list() if x.identity_id!=identity][0]
    with db.owner.begin() as c:
        c.execute(text('UPDATE demo.demo_enrolled_identities SET consent_recorded_at=:time,enrolled_at=:time,expires_at=:expiry WHERE identity_id=:id'),
                  {'time':service.clock(),'expiry':service.clock()+timedelta(days=30),'id':other.identity_id})
    assert service.cleanup_expired()=={'deleted_enrollments':1}
    assert len(db.repository.list())==1
