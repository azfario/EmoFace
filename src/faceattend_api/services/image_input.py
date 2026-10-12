"""Streaming bounded RAM-only multipart parsing; no UploadFile disk spooling."""
from dataclasses import dataclass, field
import io
import warnings
import anyio
import cv2
import numpy as np
from PIL import Image
from python_multipart.multipart import MultipartParser, parse_options_header
from ..errors import APIError

@dataclass
class Part:
    name: str = ''
    mime: str = ''
    is_file: bool = False
    data: bytearray = field(default_factory=bytearray)

async def read_parts(request, settings, *, comparison=False, enrollment=False):
    if comparison and enrollment:
        raise APIError('INVALID_REQUEST')
    media, options = parse_options_header(request.headers.get('content-type', ''))
    boundary = options.get(b'boundary')
    if media != b'multipart/form-data' or not boundary or len(boundary) > 200:
        raise APIError('UNSUPPORTED_MEDIA_TYPE')
    parts, headers = [], {}
    current = None
    header_name, header_value = bytearray(), bytearray()
    header_bytes = 0
    complete = False
    allowed = {'probe', 'enrollment_images', 'claimed_id', 'adult_consent', 'research_opt_in'} if comparison else {'image', 'adult_consent', 'research_opt_in'}
    if enrollment:
        allowed = {'enrollment_images','display_name','adult_consent','research_opt_in','enrollment_consent','retention_days'}

    def begin():
        nonlocal current, header_bytes
        if len(parts) >= (10 if enrollment else 9 if comparison else 3):
            raise APIError('INVALID_REQUEST')
        current = Part()
        headers.clear()
        header_bytes = 0

    def header_field(data, start, end):
        nonlocal header_bytes
        header_bytes += end-start
        if header_bytes > 8192:
            raise APIError('INVALID_REQUEST')
        header_name.extend(data[start:end])

    def header_val(data, start, end):
        nonlocal header_bytes
        header_bytes += end-start
        if header_bytes > 8192:
            raise APIError('INVALID_REQUEST')
        header_value.extend(data[start:end])

    def header_end():
        name = bytes(header_name).lower()
        if name in headers:
            raise APIError('INVALID_REQUEST')
        headers[name] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def headers_finished():
        disposition, opts = parse_options_header(headers.get(b'content-disposition', b''))
        if disposition != b'form-data':
            raise APIError('INVALID_REQUEST')
        try:
            current.name = opts.get(b'name', b'').decode('ascii')
            current.mime = headers.get(b'content-type', b'').decode('ascii').lower()
        except UnicodeError:
            raise APIError('INVALID_REQUEST') from None
        current.is_file = b'filename' in opts
        if current.name not in allowed or current.is_file != (current.name in {'image', 'probe', 'enrollment_images'}):
            raise APIError('INVALID_REQUEST')
        if current.is_file and current.mime not in {'image/jpeg', 'image/png'}:
            raise APIError('UNSUPPORTED_MEDIA_TYPE')

    def part_data(data, start, end):
        limit = settings.max_image_bytes if current.is_file else 128
        if len(current.data)+end-start > limit:
            raise APIError('PAYLOAD_TOO_LARGE')
        current.data.extend(data[start:end])

    def part_end():
        if current.name != 'enrollment_images' and any(p.name == current.name for p in parts):
            raise APIError('INVALID_REQUEST')
        parts.append(current)

    def finished():
        nonlocal complete
        complete = True

    parser = MultipartParser(boundary, {'on_part_begin': begin, 'on_header_field': header_field,
        'on_header_value': header_val, 'on_header_end': header_end, 'on_headers_finished': headers_finished,
        'on_part_data': part_data, 'on_part_end': part_end, 'on_end': finished})
    total = 0
    try:
        with anyio.fail_after(settings.upload_timeout_seconds):
            async for chunk in request.stream():
                total += len(chunk)
                if total > settings.max_request_bytes:
                    raise APIError('PAYLOAD_TOO_LARGE')
                parser.write(chunk)
            parser.finalize()
    except APIError:
        raise
    except TimeoutError:
        raise APIError('REQUEST_TIMEOUT') from None
    except Exception:
        raise APIError('INVALID_REQUEST') from None
    if not complete:
        raise APIError('INVALID_REQUEST')
    required = allowed
    if {p.name for p in parts} != required:
        raise APIError('INVALID_REQUEST')
    fields = {p.name: p.data for p in parts if not p.is_file}
    if fields['adult_consent'] != b'true' or fields['research_opt_in'] != b'true':
        raise APIError('CONSENT_REQUIRED')
    if comparison and not 3 <= sum(p.name == 'enrollment_images' for p in parts) <= 5:
        raise APIError('INVALID_REQUEST')
    if enrollment and not 3 <= sum(p.name == 'enrollment_images' for p in parts) <= 5:
        raise APIError('INVALID_REQUEST')
    return parts

def decode(part, settings):
    data = part.data
    jpeg = data[:3] == b'\xff\xd8\xff'
    png = data[:8] == b'\x89PNG\r\n\x1a\n'
    if not ((jpeg and part.mime == 'image/jpeg') or (png and part.mime == 'image/png')):
        raise APIError('UNSUPPORTED_MEDIA_TYPE')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {'JPEG', 'PNG'} or getattr(image, 'n_frames', 1) != 1:
                    raise APIError('INVALID_IMAGE')
                if image.width <= 0 or image.height <= 0 or image.width*image.height > settings.max_pixels:
                    raise APIError('PAYLOAD_TOO_LARGE')
                image.verify()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise APIError('INVALID_IMAGE')
        if image.shape[0]*image.shape[1] > settings.max_pixels:
            raise APIError('PAYLOAD_TOO_LARGE')
        if not image.any():
            raise APIError('BLACK_FRAME_REJECTED')
        return image
    except APIError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise APIError('PAYLOAD_TOO_LARGE') from None
    except Exception:
        raise APIError('INVALID_IMAGE') from None
