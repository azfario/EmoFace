from dataclasses import dataclass, field
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def private_base():
    base = os.environ.get('LOCALAPPDATA')
    return Path(base) / 'FaceAttend/phase2/private' if base else Path.home() / '.local/share/FaceAttend/phase2/private'

@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    private_parent: Path = field(default_factory=private_base)
    expression_enabled: bool = True
    max_image_bytes: int = 8 * 1024 * 1024
    max_request_bytes: int = 40 * 1024 * 1024
    max_pixels: int = 10_000_000
    upload_timeout_seconds: float = 30.0
    inference_timeout_seconds: float = 60.0
    host: str = '127.0.0.1'
    port: int = 8000

    def __post_init__(self):
        if self.host != '127.0.0.1' or not 1024 <= self.port <= 65535:
            raise ValueError('Only loopback binding and an unprivileged port are supported.')
        if not 0 < self.max_image_bytes <= 8*1024*1024 or not 0 < self.max_request_bytes <= 40*1024*1024:
            raise ValueError('Encoded limits must be positive and no greater than the research defaults.')
        if not 0 < self.max_pixels <= 10_000_000 or not 0 < self.upload_timeout_seconds <= 120 or not 0 < self.inference_timeout_seconds <= 120:
            raise ValueError('Invalid pixel/timeout bounds.')
        if self.private_parent.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Temporary references must be outside the project.')
        resolved = self.private_parent.resolve()
        if resolved.anchor.startswith('\\\\') or any(part.lower().startswith('onedrive') for part in resolved.parts):
            raise ValueError('Private references must stay on a local, non-OneDrive filesystem.')

