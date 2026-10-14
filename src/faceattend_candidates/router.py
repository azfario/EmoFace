from fastapi import APIRouter,Request
from faceattend_api.errors import APIError
from face_verification.demo_quality import demo_face_bundle
from faceattend_api.services.image_input import read_parts,decode
from faceattend_enrollment.router import authorization,native_worker
from .errors import CandidateError
from .service import CandidateService

router=APIRouter(prefix='/api/v1/demo',include_in_schema=False)

@router.post('/candidates')
async def candidates(request:Request,top_k:int=3):
    deps=authorization(request,True)
    settings=request.app.state.candidates
    if not settings.enabled:raise CandidateError('CANDIDATES_DISABLED')
    if request.headers.get('x-faceattend-candidate-opt-in')!='1':
        raise CandidateError('CANDIDATE_SEARCH_OPT_IN_REQUIRED')
    if not 1<=top_k<=settings.max_top_k:raise CandidateError('INVALID_CANDIDATE_REQUEST')
    enrollment=deps.require_service()
    models=request.app.state.models
    if not models.gate.acquire(blocking=False):raise APIError('SERVICE_BUSY')
    parts=[]
    try:
        parts=await read_parts(request,request.app.state.settings)
        image_part=next(p for p in parts if p.is_file)
        def work():
            try:
                bundle=demo_face_bundle(models.face())
                image=decode(image_part,request.app.state.settings)
                return CandidateService(enrollment,settings).search(image,bundle,top_k=top_k,
                    adult_consent=True,research_opt_in=True,search_opt_in=True)
            finally:
                for part in parts:part.data.clear()
        result=await native_worker(work,request.app.state.settings.inference_timeout_seconds)
        # Do not reveal names if the operator logged out/session expired during work.
        authorization(request)
        return result
    finally:
        for part in parts:part.data.clear()
        models.gate.release()
