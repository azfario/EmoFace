from datetime import datetime,timezone,timedelta
from uuid import uuid4
import re
from face_verification import enroll_user
from face_verification.schemas import Status
from faceattend_api.errors import core_error
from .models import PURPOSE
from .repository import Record,valid_name,valid_id
from .errors import EnrollmentError

class EnrollmentService:
    def __init__(self,repository,files,retention_days=30,clock=lambda:datetime.now(timezone.utc)):
        self.repository=repository;self.files=files;self.retention_days=retention_days;self.clock=clock

    def enroll(self,display_name,images,bundle,*,adult_consent=False,research_opt_in=False,enrollment_consent=False,
               retention_days=None,identity_id=None):
        valid_name(display_name)
        if adult_consent is not True or research_opt_in is not True or enrollment_consent is not True:
            raise EnrollmentError('ENROLLMENT_CONSENT_REQUIRED')
        days=retention_days if retention_days is not None else self.retention_days
        if type(days) is not int or not 1<=days<=min(30,self.retention_days): raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        if not isinstance(bundle.model_id,str) or not re.fullmatch(r'[A-Za-z0-9_:-]{1,128}',bundle.model_id):
            raise EnrollmentError('ENROLLMENT_REFERENCE_UNAVAILABLE')
        if len(images)<3 or len(images)>5: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
        replace=identity_id is not None
        identity_id=valid_id(identity_id) if replace else uuid4()
        locator=uuid4().hex
        with self.files.locked():
            old=self.repository.get(identity_id) if replace else None
            if old and (old.status not in {'active','revoked','expired'} or old.display_name.casefold()!=display_name.casefold()):
                raise EnrollmentError('ENROLLMENT_INACTIVE')
            now=self.clock()
            intent={'identity_id':str(identity_id),'locator':locator,'old_locator':old.template_locator if old else None,
                    'created_at':now.isoformat(),'model_id':bundle.model_id}
            store=self.files.stage(identity_id,locator,intent)
            attempted=False
            try:
                result=enroll_user(self.files.user(locator),images,store,bundle,adult_consent=True,identity_consent=True)
                if result.status!=Status.OK: raise core_error(result.status,
                    photo_number=result.photo_number, quality_reason=result.quality_reason)
                self._check_template(store,locator,bundle.model_id,bundle.embedding_dim)
                # No DB transaction exists while GPU inference is running.
                entry=Record(identity_id,display_name,locator,now,PURPOSE,self.clock(),now+timedelta(days=days),'pending',bundle.model_id)
                attempted=True
                self.repository.reserve(entry,replace=replace)
                self._finish(entry,intent)
                return self.repository.get(identity_id).public()
            except EnrollmentError as exc:
                if exc.code in {'DUPLICATE_DISPLAY_NAME','GALLERY_FULL','ENROLLMENT_INVALID_REQUEST','ENROLLMENT_BUSY','ENROLLMENT_INACTIVE'}:
                    self.files.clean_stage(identity_id)
                elif not attempted:
                    self.files.clean_stage(identity_id)
                raise
            except Exception:
                if not attempted: self.files.clean_stage(identity_id)
                raise

    def _check_template(self,store,locator,model_id,dimension=512):
        try:
            payload=store.load(self.files.user(locator))
            if payload['model_id']!=model_id or len(payload['embedding'])!=dimension:
                raise ValueError
        except Exception: raise EnrollmentError('ENROLLMENT_REFERENCE_UNAVAILABLE') from None

    def _finish(self,entry,intent):
        self.files.promote(entry.identity_id,entry.template_locator)
        self._check_template(self.files.store(),entry.template_locator,entry.model_policy_fingerprint)
        old=intent['old_locator']
        if old and old!=entry.template_locator:
            if self.repository.locator_used(old): raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING')
            self.files.delete(old)
        if entry.status=='pending': self.repository.activate(entry.identity_id,entry.template_locator)
        self.files.clean_stage(entry.identity_id)

    def inspect(self,identity_id,model_id):
        entry=self.repository.get(identity_id)
        if entry.status!='active' or entry.expires_at<=self.clock():
            raise EnrollmentError('ENROLLMENT_INACTIVE')
        if entry.model_policy_fingerprint!=model_id: raise EnrollmentError('ENROLLMENT_REFERENCE_UNAVAILABLE')
        with self.files.locked(): self._check_template(self.files.store(),entry.template_locator,model_id)
        return entry.public()

    def revoke(self,identity_id):
        with self.files.locked():
            entry=self.repository.block(identity_id,'revoked')
            self.files.delete(entry.template_locator)
            return entry.public()

    def delete(self,identity_id):
        identity_id=valid_id(identity_id)
        with self.files.locked(): return self._delete(identity_id)

    def _delete(self,identity_id):
        try: entry=self.repository.block(identity_id,'deletion_pending')
        except EnrollmentError as exc:
            if exc.code=='ENROLLMENT_NOT_FOUND': return {'status':'deleted'}
            raise
        for intent in self.files.intents(identity_id):
            if intent['identity_id']==str(identity_id) and intent['old_locator']:
                if self.repository.locator_used(intent['old_locator']):
                    raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING')
                self.files.delete(intent['old_locator'])
        self.files.delete(entry.template_locator)
        self.files.clean_stage(identity_id)
        self.repository.remove(identity_id)
        return {'status':'deleted'}

    def cleanup_expired(self):
        with self.files.locked():
            rows=self.repository.expired(self.clock())
            for entry in rows:
                if entry.status!='deletion_pending': self.repository.block(entry.identity_id,'expired')
                self._delete(entry.identity_id)
            return {'deleted_enrollments':len(rows)}

    def recover(self):
        with self.files.locked():
            completed=orphans=0
            for intent in self.files.intents():
                identity_id=valid_id(intent['identity_id'])
                try: entry=self.repository.get(identity_id)
                except EnrollmentError as exc:
                    if exc.code!='ENROLLMENT_NOT_FOUND': raise
                    entry=None
                if entry and entry.template_locator==intent['locator']:
                    if entry.status not in {'pending','active'} or entry.expires_at<=self.clock():
                        self._delete(identity_id)
                    else:
                        if entry.model_policy_fingerprint!=intent['model_id']: raise EnrollmentError('ENROLLMENT_REFERENCE_UNAVAILABLE')
                        self._finish(entry,intent);completed+=1
                else:
                    try:
                        created=datetime.fromisoformat(intent['created_at'])
                        old_enough=created.tzinfo is not None and self.clock()-created>=timedelta(hours=24)
                    except Exception: raise EnrollmentError('ENROLLMENT_CLEANUP_PENDING') from None
                    if old_enough and not self.repository.locator_used(intent['locator']):
                        self.files.delete(intent['locator']);self.files.clean_stage(identity_id);orphans+=1
            for page in range(1,5):
                for entry in self.repository.list(page):
                    if entry.status=='deletion_pending': self._delete(entry.identity_id);completed+=1
                    elif entry.status=='revoked': self.files.delete(entry.template_locator)
            return {'completed_operations':completed,'deleted_old_orphan_stages':orphans}
