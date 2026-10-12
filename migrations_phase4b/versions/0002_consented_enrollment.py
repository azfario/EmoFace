"""Minimal local adult registration metadata, separate from protected aggregates."""
from alembic import op
import sqlalchemy as sa

revision='0002_consented_enrollment'
down_revision='0001_daily_stats'
branch_labels=None
depends_on=None

def upgrade():
    op.create_table('demo_enrolled_identities',
        sa.Column('identity_id',sa.Uuid(),nullable=False),
        sa.Column('display_name',sa.String(64),nullable=False),
        sa.Column('template_locator',sa.String(32),nullable=False),
        sa.Column('consent_recorded_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('purpose_version',sa.String(32),nullable=False),
        sa.Column('enrolled_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('expires_at',sa.DateTime(timezone=True),nullable=False),
        sa.Column('status',sa.String(24),nullable=False),
        sa.Column('model_policy_fingerprint',sa.String(128),nullable=False),
        sa.PrimaryKeyConstraint('identity_id',name='pk_demo_enrolled_identities'),
        sa.UniqueConstraint('template_locator',name='uq_demo_template_locator'),
        sa.CheckConstraint("display_name ~ '^[A-Za-z0-9][A-Za-z0-9 _.:-]{0,63}$' AND display_name=btrim(display_name)",name='ck_demo_name'),
        sa.CheckConstraint("template_locator ~ '^[0-9a-f]{32}$'",name='ck_demo_locator'),
        sa.CheckConstraint("status IN ('pending','active','revoked','expired','deletion_pending')",name='ck_demo_status'),
        sa.CheckConstraint("purpose_version='adult-enrollment-v1'",name='ck_demo_purpose'),
        sa.CheckConstraint("expires_at>consent_recorded_at AND expires_at<=consent_recorded_at+INTERVAL '30 days' AND enrolled_at>=consent_recorded_at AND enrolled_at<expires_at",name='ck_demo_dates'),
        sa.CheckConstraint("model_policy_fingerprint ~ '^[A-Za-z0-9_:-]{1,128}$'",name='ck_demo_fingerprint'),schema='demo')
    op.create_index('uq_demo_display_name','demo_enrolled_identities',[sa.text('lower(display_name)')],unique=True,schema='demo')

def downgrade():
    # Destructive to registration metadata; isolated tests only automatically.
    op.drop_table('demo_enrolled_identities',schema='demo')
