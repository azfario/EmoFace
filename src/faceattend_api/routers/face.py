from fastapi import APIRouter, Request
from ..schemas import FaceDetection, EmbeddingInfo, Comparison
from ..services import face, temporary_compare
from ..errors import APIError
from .common import execute, upload_schema, FAILURES

router = APIRouter(prefix='/api/v1/face', tags=['face research'])

@router.post('/detect', response_model=FaceDetection, responses=FAILURES, openapi_extra=upload_schema())
async def detect(request: Request):
    return await execute(request, lambda models, images, parts: face.detect_info(images[0][1], models))

@router.post('/embedding-info', response_model=EmbeddingInfo, responses=FAILURES, openapi_extra=upload_schema())
async def embedding(request: Request):
    return await execute(request, lambda models, images, parts: face.embedding_info(images[0][1], models))

@router.post('/compare-diagnostic', response_model=Comparison, responses=FAILURES, openapi_extra=upload_schema(True))
async def comparison(request: Request):
    def operation(models, images, parts):
        try:
            selected = next(p for p in parts if p.name == 'claimed_id').data.decode('ascii')
        except UnicodeError:
            raise APIError('INVALID_ID') from None
        references = [image for name, image in images if name == 'enrollment_images']
        probe = next(image for name, image in images if name == 'probe')
        return temporary_compare.compare(selected, references, probe, models, request.app.state.settings.private_parent)
    return await execute(request, operation, comparison=True)
