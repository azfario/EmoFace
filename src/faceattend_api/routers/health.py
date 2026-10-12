from fastapi import APIRouter, Request, Response
from ..schemas import Health

router = APIRouter()

@router.get('/health', response_model=Health, tags=['readiness'])
def health(request: Request, response: Response):
    states = request.app.state.models.readiness(request.app.state.settings)
    ready = all(value in {'ready', 'disabled'} for value in states.values())
    if not ready:
        response.status_code = 503
    return Health(status='ready' if ready else 'not_ready', model_readiness=states, gpu_ready=ready)
