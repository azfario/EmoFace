from contextlib import contextmanager
from pathlib import Path
import tempfile
from cryptography.fernet import Fernet
from face_verification.enrollment import enroll_user, delete_enrollment
from face_verification.research_demo import process_frame
from face_verification.template_store import TemplateStore, validate_user_id
from face_verification.schemas import Status, VerificationError
from ..errors import APIError, core_error
from ..schemas import Comparison, Quality

@contextmanager
def request_reference(user_id, images, models, private_parent, *, audit=None,
                      enroller=enroll_user, store_factory=TemplateStore):
    """Reuse core lifecycle with request-local key injection; no environment key mutation."""
    validate_user_id(user_id)
    parent = Path(private_parent)
    if not parent.is_dir():
        raise APIError('STORAGE_ERROR')
    audit = audit if audit is not None else {}
    key = Fernet.generate_key()
    store = None
    directory = None
    try:
        with tempfile.TemporaryDirectory(prefix='phase3_session_', dir=parent) as directory:
            try:
                store = store_factory(directory, key=key, retention_days=models.config.retention_days)
                result = enroller(user_id, images, store, models, adult_consent=True, identity_consent=True)
                if result.status != Status.OK:
                    raise core_error(result.status)
                yield store
            finally:
                if store is not None:
                    delete_enrollment(user_id, store)
                    audit['template_removed'] = not store.exists(user_id)
    finally:
        audit['directory_removed'] = directory is not None and not Path(directory).exists()
        del store
        del key
        audit['key_reference_dropped'] = True  # Not secure memory erasure.

def compare(user_id, images, probe, models, private_parent):
    try:
        with request_reference(user_id, images, models, private_parent) as store:
            result = process_frame(probe, models, adult_consent=True, identity_consent=True,
                                   claimed_user_id=user_id, store=store)
            if result.status != 'DIAGNOSTIC_COMPARISON_COMPLETED':
                raise core_error(result.status)
            # Deliberately reconstruct a safe response, never relay a caller/core decision.
            return Comparison(quality=Quality(**result.quality), cosine_similarity=result.cosine_similarity)
    except VerificationError as exc:
        raise core_error(exc.status) from None
    except OSError:
        raise APIError('STORAGE_ERROR') from None

