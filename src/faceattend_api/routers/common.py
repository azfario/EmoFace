import warnings
import time
import asyncio
import anyio
from face_verification.schemas import VerificationError
from ..errors import APIError, core_error
from ..schemas import Failure
from ..services.image_input import read_parts, decode
from faceattend_db.instrumentation import PATH_OPERATIONS, QUALITY_CODES
from faceattend_db.models import MAX_PROCESSING_MS

FAILURES = {code: {'model': Failure, 'description': text} for code, text in
            [(400, 'Invalid image'), (403, 'Consent or local-origin refusal'), (408, 'Upload timeout'),
             (413, 'Payload/pixel limit'), (415, 'Unsupported encoding'), (422, 'Quality or request rejection'),
             (503, 'Model/CUDA unavailable or busy'), (504, 'Inference timeout')]}

def upload_schema(comparison=False):
    properties = {'adult_consent': {'type': 'string', 'enum': ['true'], 'description': 'Operator confirms consent from every adult pictured.'},
                  'research_opt_in': {'type': 'string', 'enum': ['true'], 'description': 'Explicit opt-in to this route\'s local research processing.'}}
    if comparison:
        properties.update(claimed_id={'type': 'string', 'pattern': '^volunteer_[a-z0-9_]{3,40}$', 'example': 'volunteer_demo'},
                          enrollment_images={'type': 'array', 'minItems': 3, 'maxItems': 5, 'items': {'type': 'string', 'format': 'binary'}},
                          probe={'type': 'string', 'format': 'binary'})
    else:
        properties['image'] = {'type': 'string', 'format': 'binary'}
    return {'requestBody': {'required': True, 'content': {'multipart/form-data': {'schema': {
        'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}}}}}

async def execute(request, operation, *, comparison=False):
    settings, models = request.app.state.settings, request.app.state.models
    parts = await read_parts(request, settings, comparison=comparison)
    # Never retain requests/results in a telemetry queue. Workers share only
    # a broad outcome and a coarse integer duration after consent validation.
    metric = [None, 0]

    def work():
        ml_started = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                bundle = models.face()
                images = [(p.name, decode(p, settings)) for p in parts if p.is_file]
                ml_started = time.monotonic()
                result = operation(bundle, images, parts)
                metric[0] = 'completed'
                return result
        except VerificationError as exc:
            error = core_error(exc.status)
            metric[0] = 'quality_rejected' if error.code in QUALITY_CODES else 'processing_error'
            raise error from None
        except APIError as exc:
            if exc.code not in {'INVALID_ID', 'INVALID_REQUEST', 'UNKNOWN_ID'}:
                metric[0] = 'quality_rejected' if exc.code in QUALITY_CODES else 'processing_error'
            raise
        except Exception:
            metric[0] = 'processing_error'
            raise APIError('INTERNAL_ERROR') from None
        finally:
            for part in parts:
                part.data.clear()
            if ml_started is None:
                metric[0] = None  # Decode/model admission failures never count.
            else:
                metric[1] = min(MAX_PROCESSING_MS, int((time.monotonic()-ml_started)*1000)//10*10)

    try:
        # Shield a running native call on cancellation: no early gate release or
        # deleted reference beneath an active GPU worker. Deadline is cooperative.
        started = time.monotonic()
        worker = asyncio.create_task(anyio.to_thread.run_sync(work, abandon_on_cancel=False))
        with anyio.fail_after(settings.inference_timeout_seconds):
            try:
                result = await asyncio.shield(worker)
            except asyncio.CancelledError:
                # Also handle direct Task.cancel(), not only AnyIO cancellation.
                # Keep the admission gate held until the native worker cleans up.
                with anyio.CancelScope(shield=True):
                    while not worker.done():
                        try:
                            await asyncio.shield(worker)
                        except asyncio.CancelledError:
                            continue
                        except Exception:
                            break
                    if not worker.cancelled():
                        worker.exception()  # Retrieve any worker error without logging sensitive context.
                raise
        if time.monotonic()-started > settings.inference_timeout_seconds:
            metric[0] = 'processing_error' if metric[0] else None
            raise APIError('INFERENCE_TIMEOUT')
        return result
    except TimeoutError:
        metric[0] = 'processing_error' if metric[0] else None
        raise APIError('INFERENCE_TIMEOUT') from None
    except APIError as exc:
        if exc.code in {'CUDA_UNAVAILABLE', 'MODEL_NOT_READY'}:
            if request.url.path.startswith('/api/v1/expression/'):
                models.expression_service = None
                models.expression_error = exc.code
            else:
                models.face_models = None
                models.face_error = exc.code
        raise
    finally:
        operation_name = PATH_OPERATIONS.get(request.url.path)
        if metric[0] and operation_name and not asyncio.current_task().cancelling():
            # Native inference/cleanup is finished. SQL runs in a bounded sync
            # worker; its failure cannot change the existing response.
            try:
                await anyio.to_thread.run_sync(request.app.state.telemetry.record,
                                               operation_name, metric[0], metric[1])
            except Exception:
                # Even an unexpected telemetry-adapter failure is fail-open.
                # Never log the exception/request/result.
                pass
