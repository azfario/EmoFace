from datetime import datetime, timezone
import numpy as np
from .detector import detect_single_face
from .embedder import generate_embedding
from .quality import normalize, image_digest
from .schemas import Status, EnrollmentResult, DeletionResult, VerificationError
from .template_store import validate_user_id

def aggregate_embeddings(vectors, consistency_cosine=0.6):
    vectors = np.stack([normalize(v) for v in vectors])
    if len(vectors) < 3:
        raise VerificationError(Status.INSUFFICIENT_DATA, 'At least three independent enrollment images are required.')
    similarities = vectors @ vectors.T
    if float(similarities[np.triu_indices(len(vectors), 1)].min()) < consistency_cosine:
        raise VerificationError(Status.INCONSISTENT_ENROLLMENT, 'Enrollment samples disagree; recollect controlled single-person images.')
    return normalize(vectors.mean(axis=0))

def enroll_user(user_id, images, store, model_bundle, *, adult_consent=False, identity_consent=False,
                detector=detect_single_face, embedder=generate_embedding):
    try:
        validate_user_id(user_id)
        if adult_consent is not True or identity_consent is not True:
            raise VerificationError(Status.CONSENT_REQUIRED, 'Explicit adult participation and identity-verification consent are required.')
        if store.exists(user_id):
            raise VerificationError(Status.DUPLICATE_ID, 'Delete the existing enrollment before replacing it.')
        images = list(images)
        config = model_bundle.config
        if not config.min_enrollment_images <= len(images) <= config.max_enrollment_images:
            raise VerificationError(Status.INSUFFICIENT_DATA, 'Provide 3–5 distinct enrollment images.')
        vectors, hashes = [], []
        for photo_number, image in enumerate(images, 1):
            face = detector(image, model_bundle)
            if face.status != Status.OK:
                raise VerificationError(face.status, face.message, quality_reason=face.quality_reason, photo_number=photo_number)
            digest = image_digest(image)
            if digest in hashes:
                raise VerificationError(Status.DUPLICATE_IMAGE, 'Repeated enrollment frames are not independent samples.')
            hashes.append(digest)
            vectors.append(embedder(face, model_bundle))
        template = aggregate_embeddings(vectors, config.consistency_cosine)
        store.save(user_id, template, model_id=model_bundle.model_id, sample_hashes=hashes,
            consent={'adult': True, 'identity_verification': True, 'recorded_at': datetime.now(timezone.utc).isoformat()})
        return EnrollmentResult(Status.OK, 'Encrypted enrollment created.', user_id, len(images))
    except VerificationError as exc:
        return EnrollmentResult(exc.status, str(exc), user_id, photo_number=exc.photo_number, quality_reason=exc.quality_reason)
    except (ValueError, TypeError):
        return EnrollmentResult(Status.MODEL_ERROR, 'Enrollment embeddings were invalid or inconsistent.', user_id)

def delete_enrollment(user_id, store):
    try:
        deleted = store.delete(user_id)
        return DeletionResult(Status.OK if deleted else Status.UNKNOWN_ID, 'Template removed.' if deleted else 'Template was absent.', user_id, deleted)
    except VerificationError as exc:
        return DeletionResult(exc.status, str(exc), user_id)
