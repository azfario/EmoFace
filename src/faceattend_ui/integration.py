from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
import os
import anyio
from fastapi import APIRouter,Request
from starlette.staticfiles import StaticFiles
from face_verification.detector import detect_single_face
from face_verification.demo_quality import demo_face_bundle
from face_verification.schemas import Status
from faceattend_api.errors import APIError,core_error
from faceattend_api.services.image_input import read_parts,decode
from faceattend_enrollment.router import authorization,native_worker
from faceattend_enrollment.repository import enrollment_runtime_ready
from faceattend_candidates.service import CandidateService
from faceattend_candidates.errors import CandidateError

@dataclass(frozen=True)
class UISettings:
    enabled:bool=False
    fixture_mode:bool=False
    @classmethod
    def from_env(cls):return cls(enabled=os.environ.get('FACEATTEND_FRONTEND_ENABLED','false')=='true')

CSP="default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob:; media-src 'self' blob:; font-src 'self'; worker-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"

class UIHeaders:
    def __init__(self,app):self.app=app
    async def __call__(self,scope,receive,send):
        async def secure_send(message):
            if scope.get('path','').startswith('/app') and message['type']=='http.response.start':
                message['headers']=list(message.get('headers',[]))+[(b'content-security-policy',CSP.encode()),
                    (b'permissions-policy',b'camera=(self), microphone=(), geolocation=(), display-capture=()'),
                    (b'referrer-policy',b'no-referrer'),(b'x-frame-options',b'DENY')]
            await send(message)
        await self.app(scope,receive,secure_send)

def mount_ui(app,root):
    app.add_middleware(UIHeaders)
    directory=Path(root)/'frontend/dist'
    if app.state.ui.enabled and directory.is_dir():
        if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob('*')):raise ValueError('Unsafe frontend build.')
        app.mount('/app',StaticFiles(directory=directory,html=True),name='local-ui')

def face_box(face,image):
    import math
    x1,y1,x2,y2=face.bbox
    return {'x':math.floor(float(x1)),'y':math.floor(float(y1)),
        'width':math.ceil(float(x2))-math.floor(float(x1)),'height':math.ceil(float(y2))-math.floor(float(y1)),
        'image_width':int(image.shape[1]),'image_height':int(image.shape[0])}

router=APIRouter(prefix='/api/v1/demo',include_in_schema=False)

@router.get('/ui-status')
async def ui_status(request:Request):
    if not request.app.state.ui.enabled:raise APIError('MODEL_NOT_READY')
    dependencies=request.app.state.enrollment;telemetry=request.app.state.telemetry
    def read():
        database='disabled' if not dependencies.settings.enabled and dependencies.service is None else 'unavailable'
        if dependencies.service is not None:
            if request.app.state.ui.fixture_mode:database='fictional'
            elif enrollment_runtime_ready(dependencies.service.repository.factory):database='ready'
        totals=None
        if telemetry.settings.enabled and telemetry.repository is not None and telemetry.gate.acquire(blocking=False):
            try:
                day=datetime.now(timezone.utc).date();rows=telemetry.repository.get_aggregate_summary(day,day)
                totals={'day_utc':day.isoformat(),'requests':sum(r['request_count'] for r in rows)}
            except Exception:pass
            finally:telemetry.gate.release()
        return {'fixture_mode':request.app.state.ui.fixture_mode,'registration_database':database,
            'operator_auth_ready':dependencies.auth is not None,'login_required':dependencies.auth.login_required if dependencies.auth is not None else dependencies.settings.login_required,
            'candidates_enabled':request.app.state.candidates.enabled,
            'analysis_enabled':request.app.state.face_analysis.enabled,'telemetry_enabled':telemetry.settings.enabled,
            'anonymous_totals':totals}
    return await anyio.to_thread.run_sync(read)

@router.post('/preview-face')
async def preview_face(request:Request,include_candidates:bool=False,include_face_box:bool=False,top_k:int=3,live_camera:bool=False):
    from .live import analysis_authorization
    dependencies=analysis_authorization(request,live_camera)
    if not request.app.state.ui.enabled:raise APIError('MODEL_NOT_READY')
    if not 1<=top_k<=5:raise CandidateError('INVALID_CANDIDATE_REQUEST')
    if include_face_box and request.headers.get('x-faceattend-box-opt-in')!='1':raise CandidateError('CANDIDATE_SEARCH_OPT_IN_REQUIRED')
    enrollment=None
    if include_candidates:
        if not request.app.state.candidates.enabled:raise CandidateError('CANDIDATES_DISABLED')
        if request.headers.get('x-faceattend-candidate-opt-in')!='1':raise CandidateError('CANDIDATE_SEARCH_OPT_IN_REQUIRED')
        enrollment=dependencies.require_service()
    models=request.app.state.models
    if not models.gate.acquire(blocking=False):raise APIError('SERVICE_BUSY')
    parts=[]
    try:
        parts=await read_parts(request,request.app.state.settings)
        part=next(p for p in parts if p.is_file)
        def work():
            try:
                bundle=demo_face_bundle(models.face());image=decode(part,request.app.state.settings);face=detect_single_face(image,bundle)
                faces=[]
                if face.status!=Status.NO_FACE:
                    if face.status!=Status.OK:raise core_error(face.status)
                    identity=CandidateService.output([],True,'CANDIDATE_SEARCH_NOT_REQUESTED')
                    if enrollment is not None:identity=CandidateService(enrollment,request.app.state.candidates)._rank_face(face,bundle,top_k)
                    item={'face_index':0,'identity_diagnostic':identity}
                    if include_face_box:item['face_box']=face_box(face,image)
                    faces.append(item)
                return {'status':'PER_FACE_RESEARCH_DIAGNOSTIC_COMPLETED','face_detected':bool(faces),'face_count':len(faces),
                    'faces':faces,'verified':False,'threshold':None,'not_a_mood_or_diagnosis':True,'persistent_inference_history':False}
            finally:
                for item in parts:item.data.clear()
        result=await native_worker(work,request.app.state.settings.inference_timeout_seconds)
        authorization(request);return result
    finally:
        for item in parts:item.data.clear()
        models.gate.release()
