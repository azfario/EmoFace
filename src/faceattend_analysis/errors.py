from starlette.responses import JSONResponse
ERRORS={
    'FACE_ANALYSIS_DISABLED':(503,'Per-face visible-expression diagnostics are disabled.'),
    'EXPRESSION_OPT_IN_REQUIRED':(403,'Explicit voluntary visible-expression research opt-in is required.'),
    'INVALID_EXPRESSION_CROP':(422,'A valid, unclipped single-face crop is required.'),
    'EXPRESSION_POLICY_MISMATCH':(503,'The saved expression preprocessing policy is incompatible.'),
}
class AnalysisError(Exception):
    def __init__(self,code):
        self.code=code if code in ERRORS else 'EXPRESSION_POLICY_MISMATCH'
        super().__init__(self.code)
def response(error):
    status,message=ERRORS[error.code]
    return JSONResponse({'status':error.code,'message':message,'verified':False,'threshold':None,'not_a_mood_or_diagnosis':True},status_code=status)
