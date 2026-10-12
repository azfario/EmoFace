from fastapi import APIRouter, Request
from faceattend import CLASS_NAMES
from ..schemas import ModelInfo

router = APIRouter(prefix='/api/v1/models')

@router.get('/info', response_model=ModelInfo, tags=['readiness'])
def info(request: Request):
    models = request.app.state.models
    return ModelInfo(classifier_labels=CLASS_NAMES, model_readiness=models.readiness(request.app.state.settings),
                     gpu_device=models.device_label)
