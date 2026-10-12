"""Authenticated encryption; pseudonyms, sample hashes and vectors stay encrypted."""
import hashlib
import json
import os
from pathlib import Path
import re
from datetime import datetime, timezone, timedelta
import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from .quality import normalize
from .schemas import Status, VerificationError

def validate_user_id(user_id):
    if not isinstance(user_id, str) or not re.fullmatch(r'volunteer_[a-z0-9_]{3,40}', user_id):
        raise VerificationError(Status.INVALID_ID, 'Use a pseudonym such as volunteer_001, not a name or institutional ID.')
    return user_id

class TemplateStore:
    """Single local operator; OS permissions remain required. No identity listing API."""
    def __init__(self, directory, *, key=None, key_env='FACEATTEND_TEMPLATE_KEY', retention_days=30):
        if not 1 <= retention_days <= 365:
            raise ValueError('Retention must be 1–365 days.')
        self.retention_days = retention_days
        value = key or os.environ.get(key_env)
        if not value:
            raise VerificationError(Status.KEY_UNAVAILABLE, f'Set {key_env} from your user-controlled secret source.')
        try:
            self._cipher = Fernet(value.encode() if isinstance(value, str) else value)
        except (ValueError, TypeError) as exc:
            raise VerificationError(Status.KEY_UNAVAILABLE, 'Encryption key must be a valid Fernet key.') from exc
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Best effort on POSIX; on Windows configure a private NTFS ACL manually.
        if os.name != 'nt':
            self.directory.chmod(0o700)
    def _path(self, user_id):
        validate_user_id(user_id)
        return self.directory / (hashlib.sha256(user_id.encode()).hexdigest() + '.fernet')
    def exists(self, user_id):
        return self._path(user_id).exists()
    def save(self, user_id, vector, *, model_id, sample_hashes, consent):
        path = self._path(user_id)
        vector = normalize(vector)
        payload = {'schema_version': 1, 'user_id': user_id, 'model_id': model_id, 'embedding': vector.tolist(),
            'sample_hashes': list(sample_hashes), 'consent': consent}
        encrypted = self._cipher.encrypt(json.dumps(payload, allow_nan=False).encode())
        try:
            # O_EXCL prevents concurrent duplicate enrollment; no plaintext temporary file.
            with path.open('xb') as stream:
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != 'nt':
                path.chmod(0o600)
        except FileExistsError as exc:
            raise VerificationError(Status.DUPLICATE_ID, 'That pseudonym is already enrolled; delete before re-enrollment.') from exc
        except OSError as exc:
            raise VerificationError(Status.STORAGE_ERROR, 'Encrypted template write failed.') from exc
    def load(self, user_id):
        path = self._path(user_id)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise VerificationError(Status.UNKNOWN_ID, 'No template exists for the claimed pseudonym.') from exc
        except OSError as exc:
            raise VerificationError(Status.STORAGE_ERROR, 'Template read failed.') from exc
        try:
            value = json.loads(self._cipher.decrypt(content))
            if value['schema_version'] != 1 or value['user_id'] != user_id or not isinstance(value['model_id'], str):
                raise ValueError('Metadata mismatch.')
            vector = normalize(value['embedding'])
            if not np.isclose(np.linalg.norm(value['embedding']), 1, atol=1e-5):
                raise ValueError('Template is not normalized.')
            if value['consent']['adult'] is not True or value['consent']['identity_verification'] is not True:
                raise ValueError('Missing consent.')
            timestamp = datetime.fromisoformat(value['consent']['recorded_at'])
            if timestamp.tzinfo is None or timestamp > datetime.now(timezone.utc)+timedelta(minutes=5):
                raise ValueError('Invalid enrollment timestamp.')
            if datetime.now(timezone.utc)-timestamp > timedelta(days=self.retention_days):
                raise VerificationError(Status.TEMPLATE_EXPIRED, 'Enrollment retention period expired. Remove the template and obtain fresh consent before re-enrollment.')
            if not isinstance(value['sample_hashes'], list):
                raise ValueError('Missing provenance.')
            value['embedding'] = vector
            return value
        except (InvalidToken, ValueError, KeyError, TypeError, OverflowError) as exc:
            raise VerificationError(Status.CORRUPT_TEMPLATE, 'Template is corrupt, incompatible, or encrypted with a different key.') from exc
    def delete(self, user_id):
        try:
            self._path(user_id).unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise VerificationError(Status.STORAGE_ERROR, 'Template deletion failed.') from exc
