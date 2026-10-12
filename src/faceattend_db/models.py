from sqlalchemy import BigInteger, CheckConstraint, Date, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import date

OPERATIONS = ('face_detect', 'embedding_info', 'expression_predict', 'compare_diagnostic')
OUTCOMES = ('completed', 'quality_rejected', 'processing_error')
MAX_BIGINT = 2**63-1
MAX_PROCESSING_MS = 120_000

class Base(DeclarativeBase):
    pass

class DailyStats(Base):
    __tablename__ = 'research_daily_operation_stats'
    day_utc: Mapped[date] = mapped_column(Date, primary_key=True)
    operation: Mapped[str] = mapped_column(String(32), primary_key=True)
    outcome: Mapped[str] = mapped_column(String(24), primary_key=True)
    request_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    total_processing_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    __table_args__ = (
        CheckConstraint("operation IN ('face_detect','embedding_info','expression_predict','compare_diagnostic')", name='ck_stats_operation'),
        CheckConstraint("outcome IN ('completed','quality_rejected','processing_error')", name='ck_stats_outcome'),
        CheckConstraint('request_count >= 0', name='ck_stats_count'),
        CheckConstraint('total_processing_ms >= 0', name='ck_stats_processing'),
    )
