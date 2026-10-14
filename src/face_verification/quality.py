"""Measurable image gates, not a claim that spoofing/occlusion is solved."""
import hashlib
from dataclasses import dataclass
import cv2
import numpy as np
from .schemas import Status, VerificationError

@dataclass(frozen=True)
class FaceQualityPolicy:
    min_eye_spacing_ratio: float = 0.15
    max_box_overflow_ratio: float = 0.0
    landmark_padding_ratio: float = 0.0


def quality_error(message, reason):
    return VerificationError(Status.LOW_QUALITY_IMAGE, message, quality_reason=reason)

def normalize(vector):
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError('Embedding must be a finite nonempty vector.')
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError('Zero or invalid embedding.')
    return vector / norm

def cosine_similarity(a, b):
    a, b = normalize(a), normalize(b)
    if a.shape != b.shape:
        raise ValueError('Embedding dimension mismatch.')
    return float(np.clip(np.dot(a, b), -1, 1))

def validate_image(image, config):
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or image.size == 0:
        raise VerificationError(Status.INVALID_IMAGE, 'Provide a decoded uint8 BGR image.')
    if min(image.shape[:2]) < config.min_image_side:
        raise quality_error('Image resolution is too low.', 'resolution')

def image_digest(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()

def assess_face(image, bbox, landmarks, config, policy=None):
    policy = policy or FaceQualityPolicy()
    bbox, landmarks = np.asarray(bbox), np.asarray(landmarks)
    if bbox.shape != (4,) or landmarks.shape != (5, 2) or not np.all(np.isfinite(bbox)) or not np.all(np.isfinite(landmarks)):
        raise quality_error('Missing or invalid five-point landmarks.', 'landmarks')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    width, height = x2-x1, y2-y1
    if min(width, height) < config.min_face_side:
        raise quality_error('Face is too small.', 'face_size')
    allowance = policy.max_box_overflow_ratio
    if x1 < -allowance*width or y1 < -allowance*height or x2 > w+allowance*width or y2 > h+allowance*height:
        raise quality_error('Face is clipped beyond the allowed boundary.', 'clipping')
    pad_x, pad_y = policy.landmark_padding_ratio*width, policy.landmark_padding_ratio*height
    left, right, top, bottom = x1-pad_x, x2+pad_x, y1-pad_y, y2+pad_y
    if policy.landmark_padding_ratio or policy.max_box_overflow_ratio:
        left, right, top, bottom = max(0,left), min(w-1,right), max(0,top), min(h-1,bottom)
    if (np.any(landmarks[:, 0] < left) or np.any(landmarks[:, 0] > right)
            or np.any(landmarks[:, 1] < top) or np.any(landmarks[:, 1] > bottom)):
        raise quality_error('Landmarks lie outside the usable face area.', 'landmarks')
    if np.linalg.norm(landmarks[0] - landmarks[1]) < policy.min_eye_spacing_ratio * width:
        raise quality_error('Eye spacing suggests poor pose or landmarks.', 'pose')
    crop = image[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))]
    if not crop.size:
        raise quality_error('Face crop is empty.', 'clipping')
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness, blur = float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < config.min_blur_variance:
        raise quality_error(f'Face is blurred: Laplacian variance {blur:.2f} is below {config.min_blur_variance:.2f}.', 'blur')
    if not config.min_brightness <= brightness <= config.max_brightness:
        raise quality_error(f'Face brightness {brightness:.2f} is outside {config.min_brightness:.2f}–{config.max_brightness:.2f}.', 'lighting')
    return {'blur_variance': blur, 'brightness': brightness, 'face_width': float(x2-x1), 'face_height': float(y2-y1)}
