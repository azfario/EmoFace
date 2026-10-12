"""Explicit camera-analysis pacing, separate from enrollment mutation limits."""
from faceattend_api.errors import APIError
from faceattend_analysis.errors import AnalysisError
from faceattend_enrollment.router import authorization

def analysis_authorization(request,live_camera):
    if not live_camera:return authorization(request,True)
    dependencies=authorization(request)
    if not request.app.state.ui.enabled:raise APIError('MODEL_NOT_READY')
    if request.headers.get('x-faceattend-live-opt-in')!='1':raise AnalysisError('EXPRESSION_OPT_IN_REQUIRED')
    dependencies.require_auth().authorize(request,mutation=True,live=True)
    return dependencies
