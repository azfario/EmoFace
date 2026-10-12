import numpy as np
from face_verification.detector import detect_single_face
from face_verification.embedder import generate_embedding
from face_verification.schemas import Status
from ..errors import APIError, core_error
from ..schemas import FaceDetection, EmbeddingInfo, Quality

def detect(image, models):
    result = detect_single_face(image, models)
    if result.status != Status.OK:
        raise core_error(result.status)
    return result

def detect_info(image, models):
    result = detect(image, models)
    return FaceDetection(quality=Quality(**result.quality))

def embedding_info(image, models):
    result = detect(image, models)
    vector = generate_embedding(result, models)
    try:
        norm = float(np.linalg.norm(vector))
        if len(vector) != 512 or not np.isfinite(norm) or not np.isclose(norm, 1, atol=1e-5):
            raise APIError('MODEL_NOT_READY')
        return EmbeddingInfo(quality=Quality(**result.quality), l2_norm=norm)
    finally:
        del vector

