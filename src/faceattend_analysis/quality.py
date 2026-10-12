"""Explicit expression-only blur relaxation; identity/enrollment policy is unchanged."""
from dataclasses import replace
from types import SimpleNamespace
from face_verification.detector import detect_single_face
from face_verification.schemas import Status

BLUR_WARNING='Blurry frame: the expression prediction may be unreliable. Identity matching was skipped.'

def expression_detection(image,bundle,allow_blurry_expression):
    if not allow_blurry_expression:return detect_single_face(image,bundle),False
    # Request-local copy: never mutate the original config or model fingerprint.
    relaxed=SimpleNamespace(**{**vars(bundle),'config':replace(bundle.config,min_blur_variance=0)})
    face=detect_single_face(image,relaxed)
    blurry=face.status==Status.OK and face.quality['blur_variance']<bundle.config.min_blur_variance
    return face,blurry
