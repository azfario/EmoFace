from fastapi import APIRouter,Request
from faceattend_api.errors import APIError
from face_verification.demo_quality import demo_face_bundle
from faceattend_api.services.image_input import read_parts,decode
from faceattend_enrollment.router import authorization,native_worker
from faceattend_candidates.service import CandidateService
from faceattend_candidates.errors import CandidateError
from .expression import CropExpression
from .service import FaceAnalysis
from .errors import AnalysisError

router=APIRouter(prefix='/api/v1/demo',include_in_schema=False)

@router.post('/analyze-face')
async def analyze_face(request:Request,include_candidates:bool=False,top_k:int=3,include_face_box:bool=False,allow_blurry_expression:bool=False,live_camera:bool=False):
    from faceattend_ui.live import analysis_authorization
    dependencies=analysis_authorization(request,live_camera)
    if not request.app.state.face_analysis.enabled:raise AnalysisError('FACE_ANALYSIS_DISABLED')
    if request.headers.get('x-faceattend-expression-opt-in')!='1':raise AnalysisError('EXPRESSION_OPT_IN_REQUIRED')
    if allow_blurry_expression and request.headers.get('x-faceattend-blur-opt-in')!='1':raise AnalysisError('EXPRESSION_OPT_IN_REQUIRED')
    if include_face_box and request.headers.get('x-faceattend-box-opt-in')!='1':raise AnalysisError('EXPRESSION_OPT_IN_REQUIRED')
    settings=request.app.state.candidates
    if not 1<=top_k<=5:raise CandidateError('INVALID_CANDIDATE_REQUEST')
    enrollment=None
    if include_candidates:
        if not settings.enabled:raise CandidateError('CANDIDATES_DISABLED')
        if request.headers.get('x-faceattend-candidate-opt-in')!='1':raise CandidateError('CANDIDATE_SEARCH_OPT_IN_REQUIRED')
        enrollment=dependencies.require_service()
    models=request.app.state.models
    if not request.app.state.settings.expression_enabled or models.expression_service is None:raise APIError(models.expression_error)
    if not models.gate.acquire(blocking=False):raise APIError('SERVICE_BUSY')
    parts=[]
    try:
        parts=await read_parts(request,request.app.state.settings)
        part=next(p for p in parts if p.is_file)
        def work():
            try:
                bundle=demo_face_bundle(models.face())
                expression=CropExpression(models.expression_service)
                candidates=CandidateService(enrollment,settings) if enrollment is not None else None
                return FaceAnalysis(expression,candidate_service=candidates).analyze(decode(part,request.app.state.settings),bundle,
                    adult_consent=True,research_opt_in=True,expression_opt_in=True,top_k=top_k,include_face_box=include_face_box,allow_blurry_expression=allow_blurry_expression)
            finally:
                for item in parts:item.data.clear()
        result=await native_worker(work,request.app.state.settings.inference_timeout_seconds)
        authorization(request)
        return result
    finally:
        for item in parts:item.data.clear()
        models.gate.release()
