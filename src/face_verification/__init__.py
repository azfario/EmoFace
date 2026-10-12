"""Opt-in adult 1:1 research verification. No attendance or liveness claims."""
from .config import Config, load_config
from .detector import ModelBundle, initialize_models, detect_single_face
from .embedder import generate_embedding
from .enrollment import enroll_user, delete_enrollment
from .verification import verify_claimed_identity
from .template_store import TemplateStore
from .schemas import Status, VerificationError, VerificationResult, EnrollmentResult, DeletionResult

__all__ = ['Config', 'load_config', 'ModelBundle', 'initialize_models', 'detect_single_face',
    'generate_embedding', 'enroll_user', 'verify_claimed_identity', 'delete_enrollment',
    'TemplateStore', 'Status', 'VerificationError', 'VerificationResult', 'EnrollmentResult', 'DeletionResult']
