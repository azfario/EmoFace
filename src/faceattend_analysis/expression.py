"""Reuse loaded M1 model/transform, enforce CUDA and separate crop-domain output."""
import torch
from faceattend import CLASS_NAMES,MEAN,STD
from faceattend.engine import assert_gpu
from faceattend_api.errors import APIError
from .crop import CROP_VERSION
from .errors import AnalysisError

WARNING=('Visible-expression classification prediction only; not an internal emotional state, mood or diagnosis. '
    'Softmax scores are uncalibrated. This face-crop domain has no measured accuracy; original RAF-DB performance does not transfer.')

class CropExpression:
    def __init__(self,existing):
        self.existing=existing
        try:
            metadata=existing.metadata
            if metadata.get('architecture')!='mobilenet_v3_small' or metadata.get('class_names')!=CLASS_NAMES or metadata.get('input_size')!=[224,224] or metadata.get('normalization')!={'mean':MEAN,'std':STD}:raise ValueError
        except Exception:raise AnalysisError('EXPRESSION_POLICY_MISMATCH') from None

    @torch.inference_mode()
    def predict(self,rgb):
        existing=self.existing
        if not torch.cuda.is_available():raise APIError('CUDA_UNAVAILABLE')
        try:assert_gpu(existing.model)
        except Exception:raise APIError('CUDA_UNAVAILABLE') from None
        tensor=None;logits=None;scores=None
        try:
            if rgb.mode!='RGB':raise AnalysisError('INVALID_EXPRESSION_CROP')
            tensor=existing.transform(rgb).unsqueeze(0)
            if tensor.shape!=(1,3,224,224) or not torch.isfinite(tensor).all():raise AnalysisError('EXPRESSION_POLICY_MISMATCH')
            tensor=tensor.to(existing.device)
            assert_gpu(existing.model,tensor)
            logits=existing.model(tensor)
            if logits.shape!=(1,7) or logits.device.type!='cuda' or not torch.isfinite(logits).all():raise APIError('MODEL_NOT_READY')
            scores=torch.softmax(logits,dim=1)[0]
            ordered=torch.sort(scores,descending=True).values
            index=int(scores.argmax().item())
            entropy=float((-(scores*torch.log(scores.clamp_min(torch.finfo(scores.dtype).tiny))).sum()).item())
            return {'status':'VISIBLE_EXPRESSION_PREDICTED','label':CLASS_NAMES[index],
                'softmax_score':float(scores[index].item()),'score_calibrated':False,
                'softmax_entropy_nats':entropy,'top_score_gap':float((ordered[0]-ordered[1]).item()),
                'not_a_mood_or_diagnosis':True,'device':'CUDA','crop_version':CROP_VERSION,
                'domain_accuracy_validated':False,'warning':WARNING}
        except (APIError,AnalysisError):raise
        except Exception:raise APIError('MODEL_NOT_READY') from None
        finally:
            # Best effort reference release, not a secure CUDA/RAM erasure claim.
            del tensor,logits,scores
