"""Request-local everyday-photo profile for the demo, sharing loaded GPU models."""
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
from .quality import FaceQualityPolicy

PROFILE_VERSION = 'demo-everyday-v1'
DEMO_LIMITS = {
    'min_image_side': 128, 'min_face_side': 48, 'min_blur_variance': 20.0,
    'min_brightness': 20.0, 'max_brightness': 240.0,
}
DEMO_GEOMETRY = FaceQualityPolicy(min_eye_spacing_ratio=0.08,
    max_box_overflow_ratio=0.10, landmark_padding_ratio=0.10)


def demo_face_bundle(bundle):
    if getattr(bundle, 'demo_quality_profile', None) == PROFILE_VERSION:
        return bundle
    fingerprint = {'base_model_id': bundle.model_id, 'version': PROFILE_VERSION,
                   'limits': DEMO_LIMITS, 'geometry': asdict(DEMO_GEOMETRY)}
    model_id = PROFILE_VERSION + ':' + hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    return SimpleNamespace(**{**vars(bundle), 'config': replace(bundle.config, **DEMO_LIMITS),
        'quality_policy': DEMO_GEOMETRY, 'model_id': model_id,
        'demo_quality_profile': PROFILE_VERSION})
