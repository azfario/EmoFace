from starlette.responses import JSONResponse

ERRORS = {
    'ENROLLMENT_DISABLED':(503,'Persistent enrollment is disabled.'),
    'ENROLLMENT_UNAVAILABLE':(503,'Persistent enrollment is unavailable.'),
    'OPERATOR_AUTH_REQUIRED':(401,'An active local operator session is required.'),
    'CSRF_REQUIRED':(403,'Same-origin mutation authorization is required.'),
    'RATE_LIMITED':(429,'Local operator request limit reached; retry later.'),
    'ENROLLMENT_INVALID_REQUEST':(422,'Enrollment fields, consent, name or expiry are invalid.'),
    'ENROLLMENT_CONSENT_REQUIRED':(403,'Explicit adult persistent-enrollment consent is required.'),
    'DUPLICATE_DISPLAY_NAME':(409,'That chosen display name is reserved; manage its existing enrollment.'),
    'ENROLLMENT_NOT_FOUND':(404,'That enrollment is unavailable.'),
    'ENROLLMENT_BUSY':(503,'One local enrollment or inference operation is already active.'),
    'GALLERY_FULL':(409,'The bounded local enrollment capacity is reached.'),
    'ENROLLMENT_KEY_UNAVAILABLE':(503,'The independently managed enrollment key is unavailable.'),
    'ENROLLMENT_REFERENCE_UNAVAILABLE':(503,'The encrypted reference is unavailable or incompatible.'),
    'ENROLLMENT_INACTIVE':(422,'Consent or enrollment is inactive or expired.'),
    'ENROLLMENT_CLEANUP_PENDING':(503,'Enrollment is blocked pending explicit local recovery or cleanup.'),
    'ENROLLMENT_STORAGE_ERROR':(503,'Protected enrollment storage is unavailable.'),
}

class EnrollmentError(Exception):
    def __init__(self,code):
        self.code=code if code in ERRORS else 'ENROLLMENT_UNAVAILABLE'
        super().__init__(ERRORS[self.code][1])

def response(error):
    status,message=ERRORS[error.code]
    return JSONResponse({'status':error.code,'message':message},status_code=status)
