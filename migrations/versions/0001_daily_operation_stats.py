"""Allowlisted day-level operational aggregates only."""
from alembic import op
import sqlalchemy as sa

revision='0001_daily_stats'
down_revision=None
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('research_daily_operation_stats',
        sa.Column('day_utc',sa.Date(),nullable=False),
        sa.Column('operation',sa.String(32),nullable=False),
        sa.Column('outcome',sa.String(24),nullable=False),
        sa.Column('request_count',sa.BigInteger(),nullable=False),
        sa.Column('total_processing_ms',sa.BigInteger(),nullable=False),
        sa.PrimaryKeyConstraint('day_utc','operation','outcome',name='pk_research_daily_operation_stats'),
        sa.CheckConstraint("operation IN ('face_detect','embedding_info','expression_predict','compare_diagnostic')",name='ck_stats_operation'),
        sa.CheckConstraint("outcome IN ('completed','quality_rejected','processing_error')",name='ck_stats_outcome'),
        sa.CheckConstraint('request_count >= 0',name='ck_stats_count'),
        sa.CheckConstraint('total_processing_ms >= 0',name='ck_stats_processing'))

def downgrade():
    # Destructive; only disposable test DBs are downgraded automatically.
    op.drop_table('research_daily_operation_stats')
