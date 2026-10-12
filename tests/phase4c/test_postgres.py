"""Explicit isolated PostgreSQL catalog tests; no migration or normal DB writes."""
from dataclasses import replace
from datetime import datetime,timezone,timedelta
import os
import pytest
from sqlalchemy import text
from faceattend_enrollment.models import Identity
from faceattend_candidates.catalog import Catalog
from faceattend_candidates.errors import CandidateError
from faceattend_candidates.service import CandidateService
from tests.phase4b.test_postgres import live_database,db,entry

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4C_POSTGRES_TESTS')!='1',reason='Explicit isolated PostgreSQL opt-in required.')

def test_empty_catalog_releases_connection(db):
    assert Catalog(db.repository).snapshot()==[]
    assert db.runtime.pool.checkedout()==0

def test_all_pages_read_once_with_stable_order_and_no_writes(db):
    for n in range(100):db.repository.reserve(entry(f'Fictional {n}'))
    catalog=Catalog(db.repository);rows=catalog.snapshot()
    assert len(rows)==100 and [r.identity_id for r in rows]==sorted(r.identity_id for r in rows)
    assert catalog.revalidate(rows)==rows and db.runtime.pool.checkedout()==0
    with db.owner.connect() as c:
        assert c.execute(text('SELECT count(*) FROM demo.demo_enrolled_identities')).scalar_one()==100

def test_database_above_bound_never_silently_truncated(db):
    values=[entry(f'Fictional {n}') for n in range(101)]
    # Owner inserts only synthetic metadata in the guarded disposable DB.
    with db.owner.begin() as c:
        c.execute(Identity.__table__.insert(),[{field:getattr(row,field) for field in Identity.__table__.columns.keys()} for row in values])
    with pytest.raises(CandidateError) as error:Catalog(db.repository).snapshot()
    assert error.value.code=='CANDIDATE_GALLERY_TOO_LARGE'
    assert db.runtime.pool.checkedout()==0

@pytest.mark.parametrize('change',['revoke','replace','delete'])
def test_final_revalidation_excludes_changed_metadata(db,change):
    row=entry();db.repository.reserve(row);db.repository.activate(row.identity_id,row.template_locator)
    catalog=Catalog(db.repository);snapshot=catalog.snapshot()
    if change=='revoke':db.repository.block(row.identity_id,'revoked')
    elif change=='replace':db.repository.reserve(replace(entry(),identity_id=row.identity_id,display_name=row.display_name),replace=True)
    else:
        db.repository.block(row.identity_id,'deletion_pending');db.repository.remove(row.identity_id)
    assert catalog.revalidate(snapshot)==[] and db.runtime.pool.checkedout()==0

@pytest.mark.parametrize('kind',['pending','revoked','expired','model_mismatch'])
def test_real_metadata_eligibility(db,gallery,kind):
    row=entry();db.repository.reserve(row)
    if kind!='pending':db.repository.activate(row.identity_id,row.template_locator)
    if kind=='revoked':db.repository.block(row.identity_id,'revoked')
    now=datetime.now(timezone.utc)
    if kind=='expired':
        with db.owner.begin() as c:
            c.execute(text('UPDATE demo.demo_enrolled_identities SET consent_recorded_at=:start,enrolled_at=:start,expires_at=:end WHERE identity_id=:id'),
                {'start':now-timedelta(days=2),'end':now-timedelta(days=1),'id':row.identity_id})
    loaded=Catalog(db.repository).snapshot()[0]
    model='different_preprocessing' if kind=='model_mismatch' else row.model_policy_fingerprint
    assert not CandidateService(gallery.service,catalog=Catalog(db.repository)).eligible(loaded,model,now)

def test_read_failure_is_generic_unavailable_without_connection_leak(db):
    def failed_factory():raise RuntimeError('sensitive credential/path deliberately withheld')
    catalog=Catalog(type('Repository',(),{'factory':staticmethod(failed_factory)})())
    for call in (catalog.snapshot,lambda:catalog.revalidate([entry()])):
        with pytest.raises(CandidateError) as error:call()
        assert error.value.code=='CANDIDATE_GALLERY_UNAVAILABLE'
        assert 'sensitive' not in str(error.value)
    assert db.runtime.pool.checkedout()==0
