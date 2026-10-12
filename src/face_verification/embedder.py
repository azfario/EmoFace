import numpy as np
from .gpu import require_cuda
from .quality import normalize
from .schemas import Status, VerificationError

def generate_embedding(face, model_bundle):
    if face.status != Status.OK or face.aligned is None:
        raise VerificationError(face.status, face.message)
    try:
        require_cuda(model_bundle.recognizer.session)
        vector = normalize(np.asarray(model_bundle.recognizer.get_feat(face.aligned)).flatten())
        if vector.size != model_bundle.embedding_dim:
            raise ValueError('Unexpected embedding dimension.')
        return vector
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(Status.MODEL_ERROR, 'Embedding inference failed.') from exc
