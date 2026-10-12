from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class SafeModel(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Failure(SafeModel):
    status: str
    message: str

class Health(SafeModel):
    status: Literal['ready', 'not_ready']
    model_readiness: dict[str, Literal['ready', 'unavailable', 'disabled']]
    gpu_required: Literal[True] = True
    gpu_ready: bool

class ModelInfo(SafeModel):
    face_detector: str = 'SCRFD-10GF'
    face_embedding: str = 'ArcFace ResNet50@WebFace600K'
    expression_classifier: str = 'MobileNetV3-Small M1'
    classifier_labels: list[str]
    model_readiness: dict[str, str]
    gpu_device: str | None

class Quality(SafeModel):
    blur_variance: float
    brightness: float
    face_width: float
    face_height: float

class FaceDetection(SafeModel):
    status: Literal['OK'] = 'OK'
    face_count: Literal[1] = 1
    quality: Quality
    device: Literal['CUDA'] = 'CUDA'

class EmbeddingInfo(FaceDetection):
    status: Literal['EMBEDDING_GENERATED'] = 'EMBEDDING_GENERATED'
    dimension: Literal[512] = 512
    l2_norm: float

class OperationalVerification(SafeModel):
    verified: Literal[False] = False
    threshold: None = None
    reason: Literal['THRESHOLD_NOT_CALIBRATED'] = 'THRESHOLD_NOT_CALIBRATED'

class Comparison(SafeModel):
    status: Literal['DIAGNOSTIC_COMPARISON_COMPLETED'] = 'DIAGNOSTIC_COMPARISON_COMPLETED'
    face_count: Literal[1] = 1
    quality: Quality
    similarity_metric: Literal['cosine'] = 'cosine'
    cosine_similarity: float = Field(ge=-1, le=1)
    operational_verification: OperationalVerification = Field(default_factory=OperationalVerification)
    disclaimer: str = 'Research diagnostic only. Not identity verification or attendance approval.'

class Expression(SafeModel):
    status: Literal['VISIBLE_EXPRESSION_PREDICTED'] = 'VISIBLE_EXPRESSION_PREDICTED'
    label: Literal['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']
    confidence: float = Field(ge=0, le=1)
    confidence_calibrated: Literal[False] = False
    device: Literal['CUDA'] = 'CUDA'
    interpretation: str = 'Visible expression category only; not internal mood, identity or attendance.'

