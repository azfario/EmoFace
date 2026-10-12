from dataclasses import dataclass
import os

@dataclass(frozen=True)
class AnalysisSettings:
    enabled:bool=False
    def __post_init__(self):
        if type(self.enabled) is not bool:raise ValueError('Explicit analysis enable flag required.')
    @classmethod
    def from_env(cls):return cls(enabled=os.environ.get('FACEATTEND_FACE_ANALYSIS_ENABLED','false')=='true')
