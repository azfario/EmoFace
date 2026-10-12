"""No background queue. Only safe scalar values enter the DB worker."""
import logging
import threading
from .config import DBSettings
from .repository import Repository, valid_increment
from .session import make_engine, sessions, runtime_ready

PATH_OPERATIONS = {'/api/v1/face/detect':'face_detect', '/api/v1/face/embedding-info':'embedding_info',
                   '/api/v1/expression/predict':'expression_predict', '/api/v1/face/compare-diagnostic':'compare_diagnostic'}
QUALITY_CODES = {'NO_FACE','MULTIPLE_FACES','LOW_QUALITY_IMAGE','DUPLICATE_IMAGE','INCONSISTENT_ENROLLMENT','ENROLLMENT_FAILED'}
logger = logging.getLogger('faceattend_db')

class Telemetry:
    def __init__(self, settings=None):
        self.settings = settings or DBSettings()
        self.engine = self.repository = None
        self.status = 'disabled' if not self.settings.enabled else 'unavailable'
        self.gate = threading.Lock()
        self.warned = False

    @classmethod
    def from_env(cls):
        try:
            return cls(DBSettings.from_env())
        except ValueError:
            service = cls()
            service.status = 'unavailable'
            return service

    def _unavailable(self):
        self.status = 'unavailable'
        if not self.warned:
            logger.warning('Local aggregate database unavailable; model processing continues.')
            self.warned = True

    def start(self):
        if not self.settings.enabled:
            return
        try:
            self.engine = make_engine(self.settings.url)
            self.repository = Repository(sessions(self.engine))
            self.health()
        except Exception:
            self._unavailable()

    def health(self):
        if self.repository is None:
            return self.status
        # Health probes and writes have one bounded slot and no queue.
        if not self.gate.acquire(blocking=False):
            return self.status
        try:
            if runtime_ready(self.repository.factory):
                self.status = 'ready'
                self.warned = False
            else:
                self._unavailable()
            return self.status
        except Exception:
            self._unavailable()
            return self.status
        finally:
            self.gate.release()

    def record(self, operation, outcome, processing_ms):
        if self.repository is None or not self.gate.acquire(blocking=False):
            return
        try:
            valid_increment(operation,outcome,processing_ms,None)
            if self.status != 'ready' and not runtime_ready(self.repository.factory):
                self._unavailable()
                return
            self.repository.increment_daily(operation,outcome,processing_ms)
            self.status = 'ready'
            self.warned = False
        except Exception:
            self._unavailable()
        finally:
            self.gate.release()

    def close(self):
        if self.engine is not None:
            self.engine.dispose()
        self.engine = self.repository = None
