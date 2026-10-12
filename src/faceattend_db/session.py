"""Bounded sync access; callers use worker threads, never the event loop."""
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker
from .config import local_url
from .models import DailyStats

class DatabaseUnavailable(Exception):
    def __init__(self):
        super().__init__('Local database is unavailable.')

def make_engine(url):
    return create_engine(local_url(url), pool_size=2, max_overflow=0, pool_timeout=1,
        pool_recycle=60, pool_pre_ping=False, echo=False, hide_parameters=True,
        connect_args={'connect_timeout':2, 'options':'-c statement_timeout=1500 -c lock_timeout=800 -c idle_in_transaction_session_timeout=2000',
                      'keepalives':1, 'keepalives_idle':2, 'keepalives_interval':1, 'keepalives_count':2, 'tcp_user_timeout':3000})

def sessions(engine):
    return sessionmaker(engine, expire_on_commit=False)

def runtime_ready(factory):
    try:
        with factory() as session:
            session.execute(select(DailyStats.day_utc, DailyStats.request_count, DailyStats.total_processing_ms).where(False))
            return bool(session.execute(text("""SELECT
                NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolbypassrls
                AND NOT has_schema_privilege(current_user, 'public', 'CREATE')
                AND NOT has_database_privilege(current_user, current_database(), 'CREATE')
                AND has_table_privilege(current_user, 'research_daily_operation_stats', 'SELECT')
                AND has_table_privilege(current_user, 'research_daily_operation_stats', 'INSERT')
                AND has_table_privilege(current_user, 'research_daily_operation_stats', 'UPDATE')
                AND NOT has_table_privilege(current_user, 'research_daily_operation_stats', 'DELETE')
                AND NOT has_table_privilege(current_user, 'research_daily_operation_stats', 'TRUNCATE')
                FROM pg_roles WHERE rolname=current_user""")).scalar_one())
    except Exception:
        return False
