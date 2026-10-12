"""No expression field/history or write in the guarded isolated PostgreSQL DB."""
from dataclasses import replace
import os
from types import SimpleNamespace
import pytest
from sqlalchemy import inspect,text
from faceattend_enrollment.service import EnrollmentService
from faceattend_candidates.service import CandidateService
from faceattend_analysis.service import FaceAnalysis
from tests.phase4b.test_postgres import live_database,db
from .conftest import FakeCrop,add_reference,invoke

pytestmark=pytest.mark.skipif(os.environ.get('FACEATTEND_RUN_PHASE4D_POSTGRES_TESTS')!='1',reason='Explicit isolated PostgreSQL opt-in required.')

def test_combined_diagnostics_do_not_change_identity_or_aggregate_rows(db,gallery,context,images):
    row=add_reference(gallery,name='Fictional SQL reference')
    db.repository.reserve(replace(row,status='pending'));db.repository.activate(row.identity_id,row.template_locator)
    before=db.repository.get(row.identity_id)
    with db.owner.connect() as c:aggregates=c.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()
    enrollment=EnrollmentService(db.repository,gallery.files)
    analyzer=FaceAnalysis(FakeCrop(),candidate_service=CandidateService(enrollment))
    result=invoke(analyzer,context,images)
    assert result['faces'][0]['identity_diagnostic']['candidate_count']==1
    assert db.repository.get(row.identity_id)==before
    with db.owner.connect() as c:assert c.execute(text('SELECT count(*) FROM public.research_daily_operation_stats')).scalar_one()==aggregates
    assert db.runtime.pool.checkedout()==0

def test_schema_contains_no_prediction_history_columns_or_new_tables(db):
    inspector=inspect(db.owner)
    assert set(inspector.get_table_names(schema='demo'))=={'alembic_version','demo_enrolled_identities'}
    assert {c['name'] for c in inspector.get_columns('demo_enrolled_identities',schema='demo')}=={
        'identity_id','display_name','template_locator','consent_recorded_at','purpose_version','enrolled_at','expires_at','status','model_policy_fingerprint'}
    assert set(inspector.get_table_names(schema='public'))=={'alembic_version','research_daily_operation_stats'}

def test_anonymous_expression_never_needs_repository_session(context,images):
    result=invoke(FaceAnalysis(FakeCrop()),context,images)
    assert result['faces'][0]['identity_diagnostic']['candidate_count']==0
    assert 'identity_id' not in str(result)
