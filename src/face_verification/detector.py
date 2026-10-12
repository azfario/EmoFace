"""Official SCRFD and ArcFace wrappers with injectable dependencies."""
from dataclasses import dataclass
import hashlib
import json
from importlib.metadata import version
import numpy as np
from .gpu import cuda_session, require_cuda
from .quality import validate_image, assess_face
from .schemas import FaceResult, Status, VerificationError

@dataclass
class ModelBundle:
    detector: object
    recognizer: object
    align: object
    config: object
    model_id: str
    embedding_dim: int = 512

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def initialize_models(config, *, profile=False):
    folder = config.path(config.model_root) / 'models' / config.pack
    paths = [folder / config.detector_file, folder / config.recognition_file]
    if not all(p.is_file() for p in paths):
        raise VerificationError(Status.MODEL_ERROR, 'Official model files are missing. Acquisition is an explicit licensed action; initialize_models never downloads.')
    try:
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.utils.face_align import norm_crop
        input_shapes = [(1,3,config.detection_size,config.detection_size), (1,3,112,112)]
        sessions = [cuda_session(p, config, config.path(f'.cache/phase2/profiles/{p.stem}') if profile else None,
            input_shape=shape) for p,shape in zip(paths,input_shapes)]
        detector = SCRFD(model_file=str(paths[0]), session=sessions[0])
        detector.prepare(ctx_id=config.device_id, input_size=(config.detection_size, config.detection_size), det_thresh=config.detection_confidence)
        recognizer = ArcFaceONNX(model_file=str(paths[1]), session=sessions[1])
        recognizer.prepare(ctx_id=config.device_id)
        for session in sessions:
            require_cuda(session)
        if len(sessions[0].get_outputs()) != 9 or tuple(recognizer.input_size) != (112,112):
            raise VerificationError(Status.MODEL_ERROR, 'Unexpected model contract: require SCRFD five-point outputs and ArcFace 112x112.')
        fingerprint = {'detector': file_hash(paths[0]), 'recognizer': file_hash(paths[1]), 'det_size': config.detection_size,
            'det_confidence': config.detection_confidence, 'alignment': 'insightface.norm_crop',
            'quality': {k: v for k, v in vars(config).items() if k.startswith(('min_', 'max_'))},
            'consistency_cosine': config.consistency_cosine, 'insightface': version('insightface'),
            'onnxruntime_gpu': version('onnxruntime-gpu'), 'fixed_session_inputs': input_shapes, 'version': 2}
        model_id = f'{config.pack}:' + hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
        return ModelBundle(detector, recognizer, norm_crop, config, model_id, int(recognizer.output_shape[-1]))
    except VerificationError:
        raise
    except Exception as exc:
        raise VerificationError(Status.MODEL_ERROR, 'Model wrapper initialization failed; inspect dependency versions and ONNX contracts.') from exc

def detect_single_face(image, model_bundle):
    try:
        validate_image(image, model_bundle.config)
        require_cuda(model_bundle.detector.session)
        boxes, points = model_bundle.detector.detect(image, input_size=(model_bundle.config.detection_size,)*2, max_num=0)
        # Count every detected plausible face before quality filtering.
        if len(boxes) == 0:
            return FaceResult(Status.NO_FACE, 'No face met the detection confidence gate.')
        if len(boxes) != 1:
            return FaceResult(Status.MULTIPLE_FACES, 'Use an image containing exactly one face.')
        if points is None:
            raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Five facial keypoints are required.')
        quality = assess_face(image, np.asarray(boxes[0,:4]), np.asarray(points[0]), model_bundle.config)
        aligned = model_bundle.align(image, landmark=points[0], image_size=112)
        return FaceResult(Status.OK, bbox=boxes[0,:4], landmarks=points[0], aligned=aligned, quality=quality)
    except VerificationError as exc:
        return FaceResult(exc.status, str(exc))
    except Exception:
        return FaceResult(Status.MODEL_ERROR, 'Detection or official alignment failed.')
