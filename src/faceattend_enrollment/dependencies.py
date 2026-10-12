from faceattend_db.session import make_engine,sessions
from .config import EnrollmentSettings
from .repository import EnrollmentRepository,enrollment_runtime_ready
from .files import GalleryFiles
from .service import EnrollmentService
from .auth import OperatorAuth
from .security import read_secret
from .errors import EnrollmentError

class EnrollmentDependencies:
    def __init__(self,settings=None,*,service=None,auth=None):
        self.settings=settings or EnrollmentSettings();self.engine=None
        self.service=service;self.auth=auth
        self.error='ENROLLMENT_DISABLED' if not self.settings.enabled and service is None else 'ENROLLMENT_UNAVAILABLE'
        self.injected=service is not None and auth is not None

    @classmethod
    def from_env(cls):
        try: return cls(EnrollmentSettings.from_env())
        except Exception:
            deps=cls();deps.error='ENROLLMENT_UNAVAILABLE';return deps

    def start(self):
        if self.injected: return
        if not self.settings.enabled: return
        try:
            self.auth=OperatorAuth.from_file(self.settings.auth_file)
            self.engine=make_engine(self.settings.database_url)
            factory=sessions(self.engine)
            if not enrollment_runtime_ready(factory): raise EnrollmentError('ENROLLMENT_UNAVAILABLE')
            key=None
            if self.settings.key_file is not None:
                try: key=read_secret(self.settings.key_file,128).strip()
                except EnrollmentError: pass  # Deletion remains possible after key loss.
            files=GalleryFiles(self.settings.gallery_dir,key=key)
            self.service=EnrollmentService(EnrollmentRepository(factory),files,self.settings.retention_days)
        except Exception:
            self.service=None;self.error='ENROLLMENT_UNAVAILABLE'

    def require_auth(self):
        if self.auth is None: raise EnrollmentError(self.error)
        return self.auth

    def require_service(self):
        if self.service is None: raise EnrollmentError(self.error)
        return self.service

    def close(self):
        if self.auth is not None: self.auth.logout()
        if self.engine is not None: self.engine.dispose()
        self.engine=None;self.service=None
