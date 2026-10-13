from dataclasses import dataclass,field
from pathlib import Path
import os
from sqlalchemy.engine import URL
from faceattend_db.config import local_url
from .security import protected_path
from .errors import EnrollmentError

@dataclass(frozen=True)
class EnrollmentSettings:
    enabled:bool=False
    database_url:URL|None=field(default=None,repr=False)
    gallery_dir:Path|None=field(default=None,repr=False)
    key_file:Path|None=field(default=None,repr=False)
    auth_file:Path|None=field(default=None,repr=False)
    retention_days:int=30
    capacity:int=100
    login_required:bool=True

    def __post_init__(self):
        try:
            if type(self.enabled) is not bool or type(self.login_required) is not bool or type(self.retention_days) is not int or not 1<=self.retention_days<=30 or self.capacity!=100:
                raise ValueError
            if self.enabled:
                object.__setattr__(self,'database_url',local_url(self.database_url))
                for name in ('gallery_dir','key_file','auth_file'):
                    value=getattr(self,name)
                    if value is not None: object.__setattr__(self,name,protected_path(value))
                if self.gallery_dir is None or (self.login_required and self.auth_file is None): raise ValueError
                if (self.auth_file and self.auth_file.is_relative_to(self.gallery_dir)) or (self.key_file and self.key_file.is_relative_to(self.gallery_dir)):
                    raise ValueError
                if self.key_file is not None and self.key_file==self.auth_file: raise ValueError
        except Exception:
            raise EnrollmentError('ENROLLMENT_UNAVAILABLE') from None

    @classmethod
    def from_env(cls):
        flag=os.environ.get('FACEATTEND_ENROLLMENT_ENABLED','false')
        if flag not in {'true','false'}: raise EnrollmentError('ENROLLMENT_UNAVAILABLE')
        if flag=='false': return cls()
        login=os.environ.get('FACEATTEND_OPERATOR_LOGIN_REQUIRED','false')
        if login not in {'true','false'}:raise EnrollmentError('ENROLLMENT_UNAVAILABLE')
        return cls(enabled=True,database_url=os.environ.get('FACEATTEND_ENROLLMENT_DATABASE_URL'),
            gallery_dir=os.environ.get('FACEATTEND_ENROLLMENT_GALLERY_DIR'),
            key_file=os.environ.get('FACEATTEND_ENROLLMENT_KEY_FILE'),auth_file=os.environ.get('FACEATTEND_OPERATOR_AUTH_FILE'),
            retention_days=int(os.environ.get('FACEATTEND_ENROLLMENT_RETENTION_DAYS','30')),login_required=login=='true')
