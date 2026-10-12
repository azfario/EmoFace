from fastapi import APIRouter, Request
from ..schemas import Expression
from ..services.face import detect
from ..errors import APIError
from .common import execute, upload_schema, FAILURES

router = APIRouter(prefix='/api/v1/expression', tags=['visible expression research'])

@router.post('/predict', response_model=Expression, responses=FAILURES, openapi_extra=upload_schema())
async def predict(request: Request):
    models = request.app.state.models
    if not request.app.state.settings.expression_enabled or models.expression_service is None:
        raise APIError(models.expression_error)
    def operation(bundle, images, parts):
        image = images[0][1]
        detect(image, bundle)  # Reuse full-image single-face QC; no largest-face selection.
        encoded = next(p.data for p in parts if p.name == 'image')
        return models.expression_service.predict(image, encoded=encoded)
    return await execute(request, operation)
