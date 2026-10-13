import asyncio,json,time
import anyio
from fastapi import APIRouter,Request
from starlette.responses import JSONResponse
from faceattend_api.errors import APIError
from faceattend_api.services.image_input import read_parts,decode
from .auth import COOKIE
from .errors import EnrollmentError
from .repository import valid_id,valid_name

router=APIRouter(prefix='/api/v1/demo',include_in_schema=False)

def authorization(request,mutation=False):
    deps=request.app.state.enrollment
    deps.require_auth().authorize(request,mutation=mutation)
    return deps

async def private_json(request):
    if request.headers.get('content-type','').split(';')[0]!='application/json':
        raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    value=bytearray()
    try:
        with anyio.fail_after(5):
            async for chunk in request.stream():
                if len(value)+len(chunk)>4096: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
                value.extend(chunk)
        parsed=json.loads(value)
        if not isinstance(parsed,dict): raise ValueError
        return parsed
    except EnrollmentError: raise
    except Exception: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST') from None
    finally: value.clear()

@router.post('/session')
async def login(request:Request):
    if request.headers.get('x-faceattend-login')!='1': raise EnrollmentError('CSRF_REQUIRED')
    auth=request.app.state.enrollment.require_auth()
    data=await private_json(request)
    if set(data)!=({'password'} if auth.login_required else set()): raise EnrollmentError('ENROLLMENT_INVALID_REQUEST')
    try:
        if auth.login_required:token,csrf=await anyio.to_thread.run_sync(auth.login,data['password'])
        else:token,csrf=auth.open_session(request)
    finally: data.clear()
    response=JSONResponse({'status':'operator_session_active','csrf_token':csrf,'expires_in_seconds':900})
    response.set_cookie(COOKIE,token,max_age=900,httponly=True,samesite='strict',secure=False,path='/api/v1/demo')
    return response

@router.delete('/session')
def logout(request:Request):
    authorization(request,True).require_auth().logout(request)
    response=JSONResponse({'status':'operator_session_ended'})
    response.delete_cookie(COOKIE,path='/api/v1/demo',httponly=True,samesite='strict')
    return response

@router.get('/enrollments')
async def listing(request:Request,page:int=1):
    service=authorization(request).require_service()
    rows=await anyio.to_thread.run_sync(service.repository.list,page)
    return {'enrollments':[row.public() for row in rows],'page':page,'page_size':25}

@router.get('/enrollments/{identity_id}')
async def status(request:Request,identity_id:str):
    service=authorization(request).require_service()
    entry=await anyio.to_thread.run_sync(service.repository.get,valid_id(identity_id))
    return entry.public()

@router.post('/enrollments/{identity_id}/inspect-reference')
async def inspect_reference(request:Request,identity_id:str):
    service=authorization(request,True).require_service()
    bundle=request.app.state.models.face()
    return await anyio.to_thread.run_sync(service.inspect,valid_id(identity_id),bundle.model_id)

async def native_worker(work,timeout=60):
    task=asyncio.create_task(anyio.to_thread.run_sync(work,abandon_on_cancel=False))
    started=time.monotonic()
    try:
        with anyio.fail_after(timeout):
            try: result=await asyncio.shield(task)
            except asyncio.CancelledError:
                with anyio.CancelScope(shield=True):
                    while not task.done():
                        try: await asyncio.shield(task)
                        except asyncio.CancelledError: continue
                        except Exception: break
                    if not task.cancelled(): task.exception()
                raise
        if time.monotonic()-started>timeout: raise APIError('INFERENCE_TIMEOUT')
        return result
    except TimeoutError: raise APIError('INFERENCE_TIMEOUT') from None

async def enrollment_upload(request,identity_id=None):
    service=authorization(request,True).require_service()
    if identity_id is not None: identity_id=valid_id(identity_id)
    models=request.app.state.models;settings=request.app.state.settings
    if not models.gate.acquire(blocking=False): raise EnrollmentError('ENROLLMENT_BUSY')
    parts=[]
    try:
        parts=await read_parts(request,settings,enrollment=True)
        fields={p.name:bytes(p.data).decode('ascii') for p in parts if not p.is_file}
        if fields['enrollment_consent']!='true': raise EnrollmentError('ENROLLMENT_CONSENT_REQUIRED')
        name=valid_name(fields['display_name'])
        try: days=int(fields['retention_days'])
        except ValueError: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST') from None
        def work():
            try:
                bundle=models.face()
                images=[decode(part,settings) for part in parts if part.is_file]
                return service.enroll(name,images,bundle,adult_consent=True,research_opt_in=True,
                    enrollment_consent=True,retention_days=days,identity_id=identity_id)
            finally:
                for part in parts: part.data.clear()
        result=await native_worker(work,settings.inference_timeout_seconds)
        return result
    except UnicodeError: raise EnrollmentError('ENROLLMENT_INVALID_REQUEST') from None
    finally:
        for part in parts: part.data.clear()
        models.gate.release()

@router.post('/enrollments')
async def enroll(request:Request): return await enrollment_upload(request)

@router.post('/enrollments/{identity_id}/reenroll')
async def reenroll(request:Request,identity_id:str):
    return await enrollment_upload(request,identity_id)

@router.post('/enrollments/{identity_id}/revoke')
async def revoke(request:Request,identity_id:str):
    service=authorization(request,True).require_service()
    return await native_worker(lambda:service.revoke(valid_id(identity_id)))

@router.delete('/enrollments/{identity_id}')
async def delete(request:Request,identity_id:str):
    service=authorization(request,True).require_service()
    return await native_worker(lambda:service.delete(valid_id(identity_id)))
