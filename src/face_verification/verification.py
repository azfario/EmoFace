from .calibration import load_threshold
from .detector import detect_single_face
from .embedder import generate_embedding
from .quality import cosine_similarity, image_digest
from .schemas import Status, VerificationResult, VerificationError
from .template_store import validate_user_id

def verify_claimed_identity(user_id, image, store, model_bundle, config=None, *, adult_consent=False,
                            identity_consent=False, detector=detect_single_face, embedder=generate_embedding):
    result = VerificationResult(Status.MODEL_ERROR, claimed_user_id=user_id, model_id=model_bundle.model_id)
    config = config or model_bundle.config
    try:
        validate_user_id(user_id)
        if adult_consent is not True or identity_consent is not True:
            raise VerificationError(Status.CONSENT_REQUIRED, 'The submitter must explicitly consent to this adult 1:1 verification attempt.')
        # Only one selected record is loaded. No iteration or 1:N search.
        template = store.load(user_id)
        if template['model_id'] != model_bundle.model_id:
            raise VerificationError(Status.MODEL_MISMATCH, 'Enrollment uses a different model/preprocessing policy; re-enroll.')
        if len(template['embedding']) != model_bundle.embedding_dim:
            raise VerificationError(Status.CORRUPT_TEMPLATE, 'Template dimension is incompatible.')
        threshold = load_threshold(config.path(config.threshold_file), model_bundle.model_id, config)
        if threshold is None:
            raise VerificationError(Status.THRESHOLD_NOT_CALIBRATED, 'No adequate real development calibration is available. Verification acceptance is disabled.')
        face = detector(image, model_bundle)
        if face.status != Status.OK:
            raise VerificationError(face.status, face.message)
        if image_digest(image) in template['sample_hashes']:
            raise VerificationError(Status.DUPLICATE_IMAGE, 'Use a fresh probe, not an enrollment image.')
        query = embedder(face, model_bundle)
        score = cosine_similarity(query, template['embedding'])
        result.score, result.threshold = score, threshold
        result.verified = score >= threshold
        result.status = Status.VERIFIED if result.verified else Status.NOT_VERIFIED
    except VerificationError as exc:
        result.status, result.message = exc.status, str(exc)
    except (ValueError, TypeError):
        result.status, result.message = Status.MODEL_ERROR, 'Invalid embedding or configuration.'
    return result
