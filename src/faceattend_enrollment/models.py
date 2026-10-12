from datetime import datetime
from uuid import UUID
from sqlalchemy import CheckConstraint,DateTime,Index,String,Uuid,func
from sqlalchemy.orm import DeclarativeBase,Mapped,mapped_column

SCHEMA='demo'
PURPOSE='adult-enrollment-v1'
STATUSES=('pending','active','revoked','expired','deletion_pending')

class EnrollmentBase(DeclarativeBase): pass

class Identity(EnrollmentBase):
    __tablename__='demo_enrolled_identities'
    identity_id:Mapped[UUID]=mapped_column(Uuid,primary_key=True)
    display_name:Mapped[str]=mapped_column(String(64),nullable=False)
    template_locator:Mapped[str]=mapped_column(String(32),nullable=False,unique=True)
    consent_recorded_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    purpose_version:Mapped[str]=mapped_column(String(32),nullable=False)
    enrolled_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    expires_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),nullable=False)
    status:Mapped[str]=mapped_column(String(24),nullable=False)
    model_policy_fingerprint:Mapped[str]=mapped_column(String(128),nullable=False)
    __table_args__=(
        CheckConstraint("display_name ~ '^[A-Za-z0-9][A-Za-z0-9 _.:-]{0,63}$' AND display_name=btrim(display_name)",name='ck_demo_name'),
        CheckConstraint("template_locator ~ '^[0-9a-f]{32}$'",name='ck_demo_locator'),
        CheckConstraint("status IN ('pending','active','revoked','expired','deletion_pending')",name='ck_demo_status'),
        CheckConstraint("purpose_version='adult-enrollment-v1'",name='ck_demo_purpose'),
        CheckConstraint("expires_at>consent_recorded_at AND expires_at<=consent_recorded_at+INTERVAL '30 days' AND enrolled_at>=consent_recorded_at AND enrolled_at<expires_at",name='ck_demo_dates'),
        CheckConstraint("model_policy_fingerprint ~ '^[A-Za-z0-9_:-]{1,128}$'",name='ck_demo_fingerprint'),
        {'schema':SCHEMA})

Index('uq_demo_display_name',func.lower(Identity.display_name),unique=True)
