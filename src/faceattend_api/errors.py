from starlette.responses import JSONResponse

ERRORS = {
    'INVALID_IMAGE': (400, 'Supply one valid JPEG or PNG image.'),
    'UNSUPPORTED_MEDIA_TYPE': (415, 'Only JPEG/PNG in multipart/form-data is supported.'),
    'PAYLOAD_TOO_LARGE': (413, 'Encoded image, request or decoded pixel limit exceeded.'),
    'INVALID_REQUEST': (422, 'Request fields or reference count are invalid.'),
    'INVALID_ID': (422, 'Use a pseudonymous volunteer ID.'),
    'CONSENT_REQUIRED': (403, 'Explicit adult consent and research opt-in are required.'),
    'NO_FACE': (422, 'No usable face was detected.'),
    'MULTIPLE_FACES': (422, 'Exactly one face is required.'),
    'LOW_QUALITY_IMAGE': (422, 'The image failed the existing face-quality policy.'),
    'BLACK_FRAME_REJECTED': (400, 'The supplied image is completely black.'),
    'DUPLICATE_IMAGE': (422, 'Use distinct reference images and a fresh probe.'),
    'INCONSISTENT_ENROLLMENT': (422, 'Reference images failed the consistency policy.'),
    'ENROLLMENT_FAILED': (422, 'Temporary enrollment failed.'),
    'UNKNOWN_ID': (422, 'Selected temporary reference is unavailable.'),
    'CUDA_UNAVAILABLE': (503, 'Required CUDA computation is unavailable; CPU fallback is disabled.'),
    'MODEL_NOT_READY': (503, 'The requested model is not ready.'),
    'SERVICE_BUSY': (503, 'One inference request is already active. Retry later.'),
    'REQUEST_TIMEOUT': (408, 'The upload deadline expired.'),
    'INFERENCE_TIMEOUT': (504, 'The inference deadline expired; cleanup completes before releasing the worker.'),
    'LOCAL_ONLY': (403, 'This service accepts same-origin loopback requests only.'),
    'STORAGE_ERROR': (503, 'Private temporary storage or cleanup is unavailable.'),
    'INTERNAL_ERROR': (500, 'The local request could not be completed.'),
}

class APIError(Exception):
    def __init__(self, code):
        self.code = code if code in ERRORS else 'INTERNAL_ERROR'

def error_response(code):
    code = code if code in ERRORS else 'INTERNAL_ERROR'
    http, message = ERRORS[code]
    return JSONResponse({'status': code, 'message': message}, status_code=http)

def core_error(code):
    value = str(code)
    aliases = {'GPU_UNAVAILABLE': 'CUDA_UNAVAILABLE', 'MODEL_ERROR': 'MODEL_NOT_READY',
               'MODEL_MISMATCH': 'MODEL_NOT_READY', 'INSUFFICIENT_DATA': 'ENROLLMENT_FAILED',
               'KEY_UNAVAILABLE': 'STORAGE_ERROR', 'CORRUPT_TEMPLATE': 'STORAGE_ERROR',
               'TEMPLATE_EXPIRED': 'STORAGE_ERROR', 'DUPLICATE_ID': 'ENROLLMENT_FAILED'}
    return APIError(aliases.get(value, value))

