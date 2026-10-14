from contextlib import asynccontextmanager
import logging
import warnings
from pathlib import Path
import anyio
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from starlette.staticfiles import StaticFiles
from starlette.exceptions import HTTPException
from face_verification.schemas import VerificationError
from .config import Settings
from .dependencies import ModelDependencies
from .errors import APIError, error_response, core_error
from .middleware import LocalAdmission
from .routers import health, models, face, expression
from faceattend_db.instrumentation import Telemetry
from faceattend_db.health import router as db_health_router
from faceattend_enrollment.dependencies import EnrollmentDependencies
from faceattend_enrollment.router import router as enrollment_router
from faceattend_enrollment.errors import EnrollmentError, response as enrollment_error_response
from faceattend_candidates.config import CandidateSettings
from faceattend_candidates.router import router as candidate_router
from faceattend_candidates.errors import CandidateError,response as candidate_error_response
from faceattend_analysis.config import AnalysisSettings
from faceattend_analysis.router import router as analysis_router
from faceattend_analysis.errors import AnalysisError,response as analysis_error_response
from faceattend_ui.integration import UISettings,mount_ui,router as ui_router

class PrivateFastAPI(FastAPI):
    async def __call__(self, scope, receive, send):
        started = False
        async def safe_send(message):
            nonlocal started
            if message['type'] == 'http.response.start':
                started = True
                message['headers'] = list(message.get('headers', [])) + [(b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff')]
            await send(message)
        try:
            await super().__call__(scope, receive, safe_send)
        except Exception:
            # Starlette re-raises handled 500s for server logging. Do not expose
            # model/private paths through an outer server traceback.
            if scope['type'] == 'http' and not started:
                await error_response('INTERNAL_ERROR')(scope, receive, safe_send)
            elif scope['type'] != 'http':
                raise

def create_app(settings=None, model_dependencies=None, *, telemetry=None, enrollment_dependencies=None,candidate_settings=None,analysis_settings=None,ui_settings=None):
    settings = settings or Settings()
    dependencies = model_dependencies or ModelDependencies()
    telemetry = telemetry if telemetry is not None else Telemetry.from_env()
    enrollment = enrollment_dependencies if enrollment_dependencies is not None else EnrollmentDependencies.from_env()
    @asynccontextmanager
    async def lifespan(app):
        def start():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', FutureWarning)
                dependencies.start(settings)
        await anyio.to_thread.run_sync(start)
        await anyio.to_thread.run_sync(telemetry.start)
        await anyio.to_thread.run_sync(enrollment.start)
        try:
            yield
        finally:
            await anyio.to_thread.run_sync(enrollment.close)
            await anyio.to_thread.run_sync(telemetry.close)
            dependencies.close()

    app = PrivateFastAPI(title='FaceAttend local research API', version='3.0.0', lifespan=lifespan,
                  description='Loopback-only opt-in research. No identity acceptance, attendance or liveness.',
                  docs_url=None, redoc_url=None)
    app.state.settings, app.state.models = settings, dependencies
    app.state.telemetry = telemetry
    app.state.enrollment = enrollment
    app.state.candidates = candidate_settings if candidate_settings is not None else CandidateSettings.from_env()
    app.state.face_analysis = analysis_settings if analysis_settings is not None else AnalysisSettings.from_env()
    app.state.ui = ui_settings if ui_settings is not None else UISettings.from_env()
    # Parser diagnostics must never emit header/filename/payload fragments.
    logging.getLogger('python_multipart.multipart').disabled = True
    app.add_middleware(LocalAdmission, settings=settings, models=dependencies)

    @app.exception_handler(APIError)
    async def api_error(request: Request, exc: APIError):
        return error_response(exc.code, photo_number=exc.photo_number, quality_reason=exc.quality_reason)

    @app.exception_handler(EnrollmentError)
    async def enrollment_error(request: Request, exc: EnrollmentError):
        return enrollment_error_response(exc)

    @app.exception_handler(CandidateError)
    async def candidate_error(request: Request, exc: CandidateError):
        return candidate_error_response(exc)

    @app.exception_handler(AnalysisError)
    async def analysis_error(request: Request, exc: AnalysisError):
        return analysis_error_response(exc)

    @app.exception_handler(VerificationError)
    async def verification_error(request: Request, exc: VerificationError):
        return error_response(core_error(exc.status).code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response('INVALID_REQUEST')  # Never echo input or validation details.

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        from starlette.responses import JSONResponse
        return JSONResponse({'status': 'HTTP_ERROR', 'message': 'Request route or method is unavailable.'}, status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        return error_response('INTERNAL_ERROR')

    for router in (health.router, models.router, face.router, expression.router):
        app.include_router(router)
    app.include_router(db_health_router)
    app.include_router(enrollment_router)
    app.include_router(candidate_router)
    app.include_router(analysis_router)
    app.include_router(ui_router)
    mount_ui(app,settings.root)
    app.mount('/docs-assets', StaticFiles(directory=Path(__file__).parent/'static'), name='docs-assets')

    @app.get('/docs', include_in_schema=False)
    def docs():
        return get_swagger_ui_html(openapi_url='/openapi.json', title='FaceAttend local Swagger',
            swagger_js_url='/docs-assets/swagger-ui-bundle.js', swagger_css_url='/docs-assets/swagger-ui.css',
            swagger_favicon_url='data:,', swagger_ui_parameters={'validatorUrl':None, 'persistAuthorization':False})
    return app

# Construct routes only. GPU loading happens on lifespan startup, once.
app = create_app()
