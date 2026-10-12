from typing import Literal
from fastapi import APIRouter, Request, Response
from pydantic import BaseModel

class DBHealth(BaseModel):
    status: Literal['disabled','ready','unavailable']

router = APIRouter()

@router.get('/health/db', response_model=DBHealth, include_in_schema=False)
def db_health(request: Request, response: Response):
    status = request.app.state.telemetry.health()
    if status == 'unavailable':
        response.status_code = 503
    return DBHealth(status=status)
