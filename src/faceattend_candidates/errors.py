from starlette.responses import JSONResponse

ERRORS={
    'CANDIDATES_DISABLED':(503,'Candidate diagnostics are disabled.'),
    'CANDIDATE_SEARCH_OPT_IN_REQUIRED':(403,'Explicit candidate-search research opt-in is required.'),
    'INVALID_CANDIDATE_REQUEST':(422,'Use a bounded candidate count and one consented adult image.'),
    'CANDIDATE_GALLERY_UNAVAILABLE':(503,'The protected reference gallery cannot be read safely.'),
    'CANDIDATE_GALLERY_TOO_LARGE':(503,'The reference gallery exceeds the research bound.'),
}
class CandidateError(Exception):
    def __init__(self,code):
        self.code=code if code in ERRORS else 'CANDIDATE_GALLERY_UNAVAILABLE'
        super().__init__(self.code)
def response(error):
    http,message=ERRORS[error.code]
    return JSONResponse({'status':error.code,'message':message,'verified':False,'threshold':None},status_code=http)
