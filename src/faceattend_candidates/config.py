from dataclasses import dataclass
import os

@dataclass(frozen=True)
class CandidateSettings:
    enabled:bool=False
    gallery_limit:int=100
    max_top_k:int=5
    max_template_bytes:int=65536
    def __post_init__(self):
        if type(self.enabled) is not bool or self.gallery_limit!=100 or self.max_top_k!=5 or self.max_template_bytes!=65536:
            raise ValueError('Candidate research bounds cannot be expanded.')
    @classmethod
    def from_env(cls):
        # Invalid operator configuration fails closed without affecting health.
        return cls(enabled=os.environ.get('FACEATTEND_CANDIDATES_ENABLED','false')=='true')
