"""One shared model bundle per process, explicit lifespan loading and truthful readiness."""
import threading
import numpy as np
from face_verification import initialize_models, load_config
from face_verification.gpu import inspect_profile, require_cuda
from face_verification.schemas import VerificationError, Status
from faceattend import CLASS_NAMES
from .errors import APIError
from .services.expression import ExpressionService

class ModelDependencies:
    def __init__(self, *, face_models=None, expression_service=None, load_on_start=True):
        self.face_models = face_models
        self.expression_service = expression_service
        self.load_on_start = load_on_start
        self.gate = threading.Lock()
        self.device_label = None
        self.startup_gpu_evidence = {}
        self.face_error = 'MODEL_NOT_READY'
        self.expression_error = 'MODEL_NOT_READY'
        self.started = False

    def start(self, settings):
        if self.started:
            return
        self.started = True
        if not self.load_on_start:
            return  # Unit-test injection; no CUDA import/availability claim.
        import torch
        try:
            if not torch.cuda.is_available():
                raise APIError('CUDA_UNAVAILABLE')
            self.device_label = torch.cuda.get_device_name(0)
            self.face_models = initialize_models(load_config(root=settings.root), profile=True)
            # Synthetic startup compute proves execution, not real-face accuracy.
            self.face_models.detector.detect(np.full((640,640,3), 127, dtype=np.uint8),
                                             input_size=(640,640), max_num=0)
            self.face_models.recognizer.get_feat(np.full((112,112,3), 127, dtype=np.uint8))
            self.startup_gpu_evidence['face'] = {
                name: inspect_profile(model.session.end_profiling())
                for name, model in [('detector', self.face_models.detector), ('recognizer', self.face_models.recognizer)]}
        except (APIError, VerificationError) as exc:
            self.face_models = None
            self.face_error = 'CUDA_UNAVAILABLE' if isinstance(exc, APIError) or exc.status == Status.GPU_UNAVAILABLE else 'MODEL_NOT_READY'
        except Exception:
            self.face_models = None
            self.face_error = 'MODEL_NOT_READY'
        if settings.expression_enabled:
            try:
                self.expression_service = ExpressionService(settings.root / 'models/faceattend_best.pt')
                self.expression_service.predict(np.full((224,224,3), 127, dtype=np.uint8))
                self.startup_gpu_evidence['expression'] = {'parameter_device': 'cuda:0', 'forward_output_device': 'cuda:0', 'scope': 'synthetic startup forward'}
            except APIError as exc:
                self.expression_service = None
                self.expression_error = exc.code
            except Exception:
                self.expression_service = None
                self.expression_error = 'MODEL_NOT_READY'

    def readiness(self, settings):
        if self.load_on_start:
            import torch
            if not torch.cuda.is_available():
                return {'face': 'unavailable', 'expression': 'unavailable' if settings.expression_enabled else 'disabled'}
        return {'face': 'ready' if self.face_models is not None else 'unavailable',
                'expression': 'disabled' if not settings.expression_enabled else 'ready' if self.expression_service is not None else 'unavailable'}

    def face(self):
        if self.face_models is None:
            raise APIError(self.face_error)
        try:
            require_cuda(self.face_models.detector.session)
            require_cuda(self.face_models.recognizer.session)
        except VerificationError:
            self.face_models = None
            self.face_error = 'CUDA_UNAVAILABLE'
            raise APIError('CUDA_UNAVAILABLE') from None
        return self.face_models

    def close(self):
        self.face_models = None
        self.expression_service = None

