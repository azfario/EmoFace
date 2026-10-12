"""Loopback/same-origin admission and one active upload+inference request, no queue."""
import ipaddress
from urllib.parse import urlsplit
from .errors import error_response

INFERENCE_PATHS = {'/api/v1/face/detect', '/api/v1/face/embedding-info',
                   '/api/v1/face/compare-diagnostic', '/api/v1/expression/predict'}

def is_loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == 'localhost'

class LocalAdmission:
    def __init__(self, app, *, settings, models):
        self.app, self.settings, self.models = app, settings, models

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1008})
                return
            await self.app(scope, receive, send)
            return
        headers = {}
        for key, value in scope.get('headers', []):
            if key in headers and key in {b'host', b'origin', b'content-length'}:
                await error_response('LOCAL_ONLY')(scope, receive, send)
                return
            headers[key] = value
        try:
            authority = headers.get(b'host', b'').decode('ascii')
            url = urlsplit('http://' + authority)
            port = url.port  # Reject malformed port syntax before admission.
            valid = url.hostname in {'127.0.0.1', 'localhost', '::1'} and not url.username and not url.password and not url.path and not url.query and not url.fragment
            origin = headers.get(b'origin')
            if origin is not None:
                valid = valid and origin.decode('ascii') == scope.get('scheme', 'http') + '://' + authority
            client = scope.get('client')
            server = scope.get('server')
            valid = valid and bool(client and is_loopback(client[0])) and bool(server and is_loopback(server[0]))
        except (ValueError, UnicodeError):
            valid = False
        if not valid:
            await error_response('LOCAL_ONLY')(scope, receive, send)
            return
        length = headers.get(b'content-length')
        if length:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                await error_response('INVALID_REQUEST')(scope, receive, send)
                return
            if size > self.settings.max_request_bytes:
                await error_response('PAYLOAD_TOO_LARGE')(scope, receive, send)
                return
        gated = scope.get('path') in INFERENCE_PATHS and scope['method'] == 'POST'
        if gated and not self.models.gate.acquire(blocking=False):
            await error_response('SERVICE_BUSY')(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            if gated:
                self.models.gate.release()
