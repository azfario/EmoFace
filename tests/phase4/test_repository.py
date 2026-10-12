from datetime import date, datetime
from unittest.mock import MagicMock
import pytest
from sqlalchemy.dialects import postgresql
from faceattend_db.models import DailyStats, MAX_BIGINT
from faceattend_db.repository import Repository
from faceattend_db.session import DatabaseUnavailable

DAY=date(2026,10,10)

def factory():
    f=MagicMock()
    s=f.return_value.__enter__.return_value
    s.execute.return_value.rowcount=1
    return f,s

@pytest.mark.parametrize('operation,outcome,ms,day',[
    ('face_detect; DROP TABLE users','completed',1,DAY),('students','completed',1,DAY),
    ('face_detect','happy',1,DAY),('face_detect','completed',-1,DAY),
    ('face_detect','completed',True,DAY),('face_detect','completed',1.2,DAY),
    ('face_detect','completed',120001,DAY),('face_detect','completed',MAX_BIGINT,DAY),
    ('face_detect','completed',1,datetime(2026,10,10)),('face_detect','completed',1,'2026-10-10'),
    (None,'completed',1,DAY),('face_detect',{},1,DAY)])
def test_invalid_inputs_never_open_session(operation,outcome,ms,day):
    f,s=factory()
    with pytest.raises(ValueError): Repository(f).increment_daily(operation,outcome,ms,day)
    f.assert_not_called()

def test_postgres_parameterized_atomic_upsert():
    f,s=factory()
    Repository(f).increment_daily('face_detect','completed',20,DAY)
    stmt=s.execute.call_args.args[0]
    query=stmt.compile(dialect=postgresql.dialect())
    assert 'ON CONFLICT (day_utc, operation, outcome) DO UPDATE' in str(query)
    assert 'face_detect' not in str(query) and query.params['request_count']==1
    assert query.params['total_processing_ms']==20
    s.begin.return_value.__exit__.assert_called_once()
    f.return_value.__exit__.assert_called_once()

def test_error_rolls_back_transaction_and_closes_session():
    f,s=factory()
    s.execute.side_effect=RuntimeError('SECRET portrait name')
    with pytest.raises(DatabaseUnavailable) as e: Repository(f).increment_daily('face_detect','completed',10,DAY)
    assert 'SECRET' not in str(e.value)
    assert s.begin.return_value.__exit__.call_args.args[0] is RuntimeError
    f.return_value.__exit__.assert_called_once()

def test_overflow_refusal_not_silent_drop():
    f,s=factory(); s.execute.return_value.scalar_one_or_none.return_value=None
    with pytest.raises(DatabaseUnavailable): Repository(f).increment_daily('face_detect','completed',10,DAY)

def test_retention_strict_before_date_and_only_aggregate_table():
    f,s=factory()
    assert Repository(f).purge_old_aggregates(DAY)==1
    statement=s.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'DELETE FROM research_daily_operation_stats' in str(statement) and 'day_utc <' in str(statement)
    assert list(statement.params.values())==[DAY]

@pytest.mark.parametrize('start,end',[(DAY,date(2025,1,1)),(date(2024,1,1),DAY),(datetime(2026,1,1),DAY)])
def test_summary_ranges_bounded(start,end):
    f,s=factory()
    with pytest.raises(ValueError): Repository(f).get_aggregate_summary(start,end)
    f.assert_not_called()

def test_schema_exact_allowlisted_columns_and_constraints():
    table=DailyStats.__table__
    assert set(table.columns.keys())=={'day_utc','operation','outcome','request_count','total_processing_ms'}
    assert list(table.primary_key.columns.keys())==['day_utc','operation','outcome']
    assert all(not c.nullable for c in table.columns)
    assert len(table.constraints)==5 and len(table.indexes)==0
