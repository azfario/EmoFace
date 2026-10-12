from datetime import date, datetime, timezone
from sqlalchemy import select, delete
from sqlalchemy.dialects.postgresql import insert
from .models import DailyStats, OPERATIONS, OUTCOMES, MAX_BIGINT, MAX_PROCESSING_MS
from .session import DatabaseUnavailable

def valid_day(value):
    if type(value) is not date:
        raise ValueError('A UTC date bucket is required.')
    return value

def valid_increment(operation, outcome, processing_ms, day_utc):
    if type(operation) is not str or operation not in OPERATIONS or type(outcome) is not str or outcome not in OUTCOMES:
        raise ValueError('Aggregate category is invalid.')
    if type(processing_ms) is not int or not 0 <= processing_ms <= MAX_PROCESSING_MS:
        raise ValueError('Aggregate processing time is invalid.')
    return valid_day(day_utc) if day_utc is not None else datetime.now(timezone.utc).date()

class Repository:
    def __init__(self, factory):
        self.factory = factory

    def increment_daily(self, operation, outcome, processing_ms, day_utc=None):
        day = valid_increment(operation, outcome, processing_ms, day_utc)
        table = DailyStats.__table__
        stmt = insert(table).values(day_utc=day, operation=operation, outcome=outcome,
                                    request_count=1, total_processing_ms=processing_ms)
        stmt = stmt.on_conflict_do_update(index_elements=[table.c.day_utc, table.c.operation, table.c.outcome],
            set_={'request_count':table.c.request_count+1,
                  'total_processing_ms':table.c.total_processing_ms+stmt.excluded.total_processing_ms},
            where=(table.c.request_count < MAX_BIGINT) & (table.c.total_processing_ms <= MAX_BIGINT-processing_ms)).returning(table.c.day_utc)
        try:
            with self.factory() as session:
                with session.begin():
                    # psycopg may report -1 rowcount; RETURNING distinguishes
                    # a successful atomic increment from overflow refusal.
                    if session.execute(stmt).scalar_one_or_none() is None:
                        raise DatabaseUnavailable()  # Overflow refuses rather than wrapping.
        except Exception:
            raise DatabaseUnavailable() from None

    def get_aggregate_summary(self, start, end):
        valid_day(start); valid_day(end)
        if start > end or (end-start).days > 365:
            raise ValueError('Use an ordered range of at most 366 UTC dates.')
        try:
            with self.factory() as session:
                rows = session.execute(select(DailyStats).where(DailyStats.day_utc.between(start,end))
                                       .order_by(DailyStats.day_utc,DailyStats.operation,DailyStats.outcome)).scalars().all()
                return [{'day_utc':r.day_utc.isoformat(), 'operation':r.operation, 'outcome':r.outcome,
                         'request_count':r.request_count, 'total_processing_ms':r.total_processing_ms} for r in rows]
        except Exception:
            raise DatabaseUnavailable() from None

    def purge_old_aggregates(self, cutoff):
        valid_day(cutoff)
        try:
            with self.factory() as session:
                with session.begin():
                    return session.execute(delete(DailyStats).where(DailyStats.day_utc < cutoff)).rowcount
        except Exception:
            raise DatabaseUnavailable() from None
