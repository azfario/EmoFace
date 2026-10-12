"""Project-relative settings with conservative, explicit research defaults."""
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    root: Path
    model_root: str = 'models/external/insightface'
    pack: str = 'buffalo_l'
    detector_file: str = 'det_10g.onnx'
    recognition_file: str = 'w600k_r50.onnx'
    store_dir: str = 'data/private/templates'
    key_env: str = 'FACEATTEND_TEMPLATE_KEY'
    retention_days: int = 30
    threshold_file: str = 'reports/phase2/calibrated_threshold.json'
    device_id: int = 0
    detection_size: int = 640
    detection_confidence: float = 0.6
    min_image_side: int = 160
    min_face_side: int = 80
    min_blur_variance: float = 60.0
    min_brightness: float = 35.0
    max_brightness: float = 220.0
    min_enrollment_images: int = 3
    max_enrollment_images: int = 5
    consistency_cosine: float = 0.6
    target_fmr: float = 0.01
    confidence: float = 0.95
    min_development_subjects: int = 10
    min_genuine_pairs: int = 30
    def path(self, value: str) -> Path:
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Configured paths must stay inside the project.')
        return path
    def __post_init__(self):
        if not 1 <= self.retention_days <= 365:
            raise ValueError('Research template retention must be 1–365 days.')
        for name in (self.pack, self.detector_file, self.recognition_file):
            if not name or '/' in name or '\\' in name or name in ('.', '..'):
                raise ValueError('Model names must be simple local filenames.')
        if not 0 < self.target_fmr < 1 or not 0 < self.confidence < 1:
            raise ValueError('Invalid calibration policy.')
        if not 0 < self.detection_confidence < 1 or not -1 < self.consistency_cosine < 1:
            raise ValueError('Invalid quality policy.')
        if self.min_enrollment_images < 3 or self.max_enrollment_images > 5 or self.min_enrollment_images > self.max_enrollment_images:
            raise ValueError('Enrollment requires 3–5 images.')
        if self.detection_size < 128 or self.detection_size % 32:
            raise ValueError('Detector input must be a positive multiple of 32.')
        if self.device_id < 0 or self.min_face_side < 1 or self.min_image_side < self.min_face_side or self.min_blur_variance < 0:
            raise ValueError('Invalid GPU or image quality configuration.')

def load_config(path=None, root=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    path = Path(path) if path else root / 'configs/phase2_verification.yaml'
    values = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return Config(root=root, **values)
