"""Typed public outcomes; no embeddings or photographs in result objects."""
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

QUALITY_REASONS = frozenset({'resolution', 'face_size', 'clipping', 'landmarks', 'pose', 'blur', 'lighting'})

class Status(StrEnum):
    OK = 'OK'
    VERIFIED = 'VERIFIED'
    NOT_VERIFIED = 'NOT_VERIFIED'
    NO_FACE = 'NO_FACE'
    MULTIPLE_FACES = 'MULTIPLE_FACES'
    LOW_QUALITY_IMAGE = 'LOW_QUALITY_IMAGE'
    INVALID_IMAGE = 'INVALID_IMAGE'
    GPU_UNAVAILABLE = 'GPU_UNAVAILABLE'
    MODEL_ERROR = 'MODEL_ERROR'
    MODEL_MISMATCH = 'MODEL_MISMATCH'
    UNKNOWN_ID = 'UNKNOWN_ID'
    INVALID_ID = 'INVALID_ID'
    DUPLICATE_ID = 'DUPLICATE_ID'
    CORRUPT_TEMPLATE = 'CORRUPT_TEMPLATE'
    TEMPLATE_EXPIRED = 'TEMPLATE_EXPIRED'
    KEY_UNAVAILABLE = 'KEY_UNAVAILABLE'
    STORAGE_ERROR = 'STORAGE_ERROR'
    CONSENT_REQUIRED = 'CONSENT_REQUIRED'
    INCONSISTENT_ENROLLMENT = 'INCONSISTENT_ENROLLMENT'
    DUPLICATE_IMAGE = 'DUPLICATE_IMAGE'
    INSUFFICIENT_DATA = 'INSUFFICIENT_DATA'
    THRESHOLD_NOT_CALIBRATED = 'THRESHOLD_NOT_CALIBRATED'

class VerificationError(RuntimeError):
    def __init__(self, status: Status, message: str, *, quality_reason=None, photo_number=None):
        super().__init__(message)
        self.status = status
        self.quality_reason = quality_reason if quality_reason in QUALITY_REASONS else None
        self.photo_number = photo_number

@dataclass
class Result:
    status: Status
    message: str = ''
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass
class VerificationResult(Result):
    claimed_user_id: str = ''
    verified: bool = False
    score: float | None = None
    threshold: float | None = None
    model_id: str = ''

@dataclass
class EnrollmentResult(Result):
    user_id: str = ''
    sample_count: int = 0
    photo_number: int | None = None
    quality_reason: str | None = None

@dataclass
class DeletionResult(Result):
    user_id: str = ''
    deleted: bool = False

@dataclass
class FaceResult(Result):
    # Internal only: never serialize the crop through to_dict.
    bbox: Any = field(default=None, repr=False)
    landmarks: Any = field(default=None, repr=False)
    aligned: Any = field(default=None, repr=False)
    quality: dict = field(default_factory=dict)
    quality_reason: str | None = None
    def to_dict(self):
        return {'status': self.status.value, 'message': self.message, 'quality': self.quality}
