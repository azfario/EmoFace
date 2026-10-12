from dataclasses import dataclass,field
from datetime import datetime,timezone
from uuid import UUID
import re
from sqlalchemy import select,update,delete,func,text
from sqlalchemy.exc import IntegrityError
from .models import Identity,PURPOSE
from .errors import EnrollmentError

def valid_name(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 _.:-]{0,63}',value) or value!=value.strip():
        raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    return value

def valid_id(value):
    try:
        identity=UUID(str(value))
        if identity.version!=4: raise ValueError
        return identity
    except Exception: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST') from None

def valid_locator(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{32}',value):
        raise EnrollmentError('ENROLLMENT_STORAGE_ERROR')
    return value

@dataclass(frozen=True,repr=False)
class Record:
    identity_id:UUID
    display_name:str
    template_locator:str
    consent_recorded_at:datetime
    purpose_version:str
    enrolled_at:datetime
    expires_at:datetime
    status:str
    model_policy_fingerprint:str

    def public(self):
        status='expired' if self.status=='active' and self.expires_at<=datetime.now(timezone.utc) else self.status
        return {'identity_id':str(self.identity_id),'display_name':self.display_name,'status':status,
                'consent_recorded_at':self.consent_recorded_at.isoformat(),'purpose_version':self.purpose_version,
                'enrolled_at':self.enrolled_at.isoformat(),'expires_at':self.expires_at.isoformat(),
                'verified':False,'threshold':None,'reason':'THRESHOLD_NOT_CALIBRATED'}

def record(row):
    values={c.name:getattr(row,c.name) for c in Identity.__table__.columns}
    for name in ('consent_recorded_at','enrolled_at','expires_at'):
        values[name]=values[name].astimezone(timezone.utc)
    return Record(**values)

class EnrollmentRepository:
    def __init__(self,factory): self.factory=factory

    def get(self,identity_id):
        identity_id=valid_id(identity_id)
        try:
            with self.factory() as session:
                row=session.get(Identity,identity_id)
                if row is None: raise EnrollmentError('ENROLLMENT_NOT_FOUND')
                return record(row)
        except EnrollmentError: raise
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def list(self,page=1):
        if type(page) is not int or not 1<=page<=4: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        try:
            with self.factory() as session:
                rows=session.execute(select(Identity).order_by(Identity.identity_id).offset((page-1)*25).limit(25)).scalars()
                return [record(row) for row in rows]
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def reserve(self,entry,replace=False):
        valid_name(entry.display_name);valid_id(entry.identity_id);valid_locator(entry.template_locator)
        if entry.purpose_version!=PURPOSE or entry.status!='pending': raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        try:
            with self.factory() as session:
                with session.begin():
                    if not session.execute(text('SELECT pg_try_advisory_xact_lock(46040261)')).scalar_one():
                        raise EnrollmentError('ENROLLMENT_BUSY')
                    if replace:
                        old=session.get(Identity,entry.identity_id,with_for_update=True)
                        if old is None: raise EnrollmentError('ENROLLMENT_NOT_FOUND')
                        if old.status not in {'active','revoked','expired'} or old.display_name.casefold()!=entry.display_name.casefold():
                            raise EnrollmentError('ENROLLMENT_INACTIVE')
                        for c in Identity.__table__.columns:
                            if c.name!='identity_id': setattr(old,c.name,getattr(entry,c.name))
                    else:
                        if session.execute(select(func.count()).select_from(Identity)).scalar_one()>=100:
                            raise EnrollmentError('GALLERY_FULL')
                        session.add(Identity(**{c.name:getattr(entry,c.name) for c in Identity.__table__.columns}))
        except EnrollmentError: raise
        except IntegrityError: raise EnrollmentError('DUPLICATE_DISPLAY_NAME') from None
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def activate(self,identity_id,locator):
        valid_id(identity_id);valid_locator(locator)
        try:
            with self.factory() as session:
                with session.begin():
                    result=session.execute(update(Identity).where(Identity.identity_id==identity_id,
                        Identity.template_locator==locator,Identity.status=='pending').values(status='active').returning(Identity.identity_id)).scalar_one_or_none()
                    if result is None: raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING')
        except EnrollmentError: raise
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def block(self,identity_id,status):
        identity_id=valid_id(identity_id)
        if status not in {'revoked','expired','deletion_pending'}: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        try:
            with self.factory() as session:
                with session.begin():
                    row=session.get(Identity,identity_id,with_for_update=True)
                    if row is None: raise EnrollmentError('ENROLLMENT_NOT_FOUND')
                    if row.status=='deletion_pending' and status!='deletion_pending': raise EnrollmentError('ENROLLMENT_INACTIVE')
                    row.status=status
                    result=record(row)
            return result
        except EnrollmentError: raise
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def remove(self,identity_id):
        identity_id=valid_id(identity_id)
        try:
            with self.factory() as session:
                with session.begin():
                    return session.execute(delete(Identity).where(Identity.identity_id==identity_id,
                        Identity.status=='deletion_pending').returning(Identity.identity_id)).scalar_one_or_none() is not None
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def expired(self,now):
        try:
            with self.factory() as session:
                return [record(r) for r in session.execute(select(Identity).where(Identity.expires_at<=now).limit(100)).scalars()]
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    def locator_used(self,locator):
        valid_locator(locator)
        try:
            with self.factory() as session:
                return session.execute(select(Identity.identity_id).where(Identity.template_locator==locator).limit(1)).scalar_one_or_none() is not None
        except Exception: raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

def enrollment_runtime_ready(factory):
    try:
        with factory() as session:
            session.execute(select(Identity.identity_id).where(False))
            return bool(session.execute(text("""SELECT NOT rolsuper AND NOT rolcreatedb AND NOT rolcreaterole AND NOT rolbypassrls
                AND NOT has_database_privilege(current_user,current_database(),'CREATE')
                AND NOT has_schema_privilege(current_user,'demo','CREATE')
                AND NOT has_schema_privilege(current_user,'public','CREATE')
                AND has_table_privilege(current_user,'demo.demo_enrolled_identities','SELECT')
                AND has_table_privilege(current_user,'demo.demo_enrolled_identities','INSERT')
                AND has_table_privilege(current_user,'demo.demo_enrolled_identities','UPDATE')
                AND has_table_privilege(current_user,'demo.demo_enrolled_identities','DELETE')
                AND NOT has_table_privilege(current_user,'demo.demo_enrolled_identities','TRUNCATE')
                FROM pg_roles WHERE rolname=current_user""")).scalar_one())
    except Exception: return False
