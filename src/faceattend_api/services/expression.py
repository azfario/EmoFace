import cv2
import io
import torch
from PIL import Image
from faceattend import CLASS_NAMES
from faceattend.engine import assert_gpu
from faceattend.inference import load_model
from ..errors import APIError
from ..schemas import Expression

def rgb_input(image, encoded=None):
    if encoded is not None:
        # Phase 1 used Pillow RGB decoding without EXIF transposition. Face QC
        # separately uses OpenCV's BGR decoder, preserving its existing behavior.
        with Image.open(io.BytesIO(encoded)) as original:
            return original.convert('RGB')
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

class ExpressionService:
    def __init__(self, checkpoint):
        if not torch.cuda.is_available():
            raise APIError('CUDA_UNAVAILABLE')
        self.device = torch.device('cuda:0')
        self.model, self.transform, self.metadata = load_model(checkpoint, self.device)
        assert_gpu(self.model)

    @torch.inference_mode()
    def predict(self, image, *, encoded=None):
        # Same full-image RGB path as Phase 1; QC is applied by the API separately.
        if not torch.cuda.is_available():
            raise APIError('CUDA_UNAVAILABLE')
        rgb = rgb_input(image, encoded)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        assert_gpu(self.model, tensor)
        logits = self.model(tensor)
        if logits.shape != (1, 7) or logits.device.type != 'cuda' or not torch.isfinite(logits).all():
            raise APIError('MODEL_NOT_READY')
        scores = torch.softmax(logits, dim=1)[0]
        index = int(scores.argmax().item())
        return Expression(label=CLASS_NAMES[index], confidence=float(scores[index].item()))

