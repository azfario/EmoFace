"""Measurable image gates, not a claim that spoofing/occlusion is solved."""
import hashlib
import cv2
import numpy as np
from .schemas import Status, VerificationError

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
        raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Image resolution is too low.')

def image_digest(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()

def assess_face(image, bbox, landmarks, config):
    bbox, landmarks = np.asarray(bbox), np.asarray(landmarks)
    if bbox.shape != (4,) or landmarks.shape != (5, 2) or not np.all(np.isfinite(bbox)) or not np.all(np.isfinite(landmarks)):
        raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Missing or invalid five-point landmarks.')
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or min(x2-x1, y2-y1) < config.min_face_side:
        raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Face is too small or clipped.')
    if np.any(landmarks[:, 0] < x1) or np.any(landmarks[:, 0] > x2) or np.any(landmarks[:, 1] < y1) or np.any(landmarks[:, 1] > y2):
        raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Landmarks lie outside the face box.')
    if np.linalg.norm(landmarks[0] - landmarks[1]) < 0.15 * (x2-x1):
        raise VerificationError(Status.LOW_QUALITY_IMAGE, 'Eye spacing suggests poor pose or landmarks.')
    crop = image[int(y1):int(y2), int(x1):int(x2)]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    brightness, blur = float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < config.min_blur_variance:
        raise VerificationError(Status.LOW_QUALITY_IMAGE, f'Face is blurred: Laplacian variance {blur:.2f} is below {config.min_blur_variance:.2f}.')
    if not config.min_brightness <= brightness <= config.max_brightness:
        raise VerificationError(Status.LOW_QUALITY_IMAGE, f'Face brightness {brightness:.2f} is outside {config.min_brightness:.2f}–{config.max_brightness:.2f}.')
    return {'blur_variance': blur, 'brightness': brightness, 'face_width': float(x2-x1), 'face_height': float(y2-y1)}
