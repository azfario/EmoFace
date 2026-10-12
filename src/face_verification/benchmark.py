"""Synchronous wall-clock timings with explicit inclusion boundaries."""
import time
import numpy as np
from .detector import detect_single_face
from .embedder import generate_embedding
from .verification import verify_claimed_identity
from .schemas import Status, VerificationError

def measure(operation, *, warmups=20, repetitions=100):
    if warmups < 1 or repetitions < 30:
        raise ValueError('Require warmup and at least 30 measured repetitions.')
    for _ in range(warmups):
        operation()
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        operation()
        timings.append((time.perf_counter_ns()-start)/1e6)
    return {'mean_ms':float(np.mean(timings)), 'median_ms':float(np.median(timings)),
        'p95_ms':float(np.percentile(timings,95)), 'warmups':warmups, 'repetitions':repetitions}

def benchmark_verification(image, user_id, store, model_bundle, *, adult_consent=False, identity_consent=False, warmups=20, repetitions=100):
    if adult_consent is not True or identity_consent is not True:
        raise VerificationError(Status.CONSENT_REQUIRED, 'Benchmarking a face requires adult verification consent.')
    first = verify_claimed_identity(user_id,image,store,model_bundle,adult_consent=True,identity_consent=True)
    if first.status not in (Status.VERIFIED,Status.NOT_VERIFIED):
        raise VerificationError(first.status,'Full verification latency cannot be measured without a valid fresh probe and calibrated threshold.')
    face = detect_single_face(image,model_bundle)
    if face.status != Status.OK:
        raise VerificationError(face.status,face.message)
    def complete():
        result = verify_claimed_identity(user_id,image,store,model_bundle,adult_consent=True,identity_consent=True)
        if result.status not in (Status.VERIFIED,Status.NOT_VERIFIED):
            raise VerificationError(result.status,'Verification stopped during timing.')
    config = model_bundle.config
    report = {'status':'MEASURED','batch_size':1,'decoded_image_resolution':list(image.shape[:2]),
        'detector_input':[config.detection_size]*2,'recognizer_input':[112,112], 'model_id':model_bundle.model_id,
        'clock':'perf_counter_ns; session.run returns CPU output synchronously', 'image_decode_included':False}
    report['scrfd_only'] = measure(lambda:model_bundle.detector.detect(image,input_size=(config.detection_size,)*2,max_num=0),warmups=warmups,repetitions=repetitions)
    report['scrfd_only']['scope'] = 'Official preprocessing, host-to-GPU transfer, detector inference, CPU output transfer, decoding and NMS; excludes face quality/alignment.'
    report['embedding_only'] = measure(lambda:generate_embedding(face,model_bundle),warmups=warmups,repetitions=repetitions)
    report['embedding_only']['scope'] = 'Already aligned crop; official recognition preprocessing, GPU inference, CPU output transfer and L2 normalization; excludes detection/alignment.'
    report['full_verification'] = measure(complete,warmups=warmups,repetitions=repetitions)
    report['full_verification']['scope'] = 'Decoded image, claimed-template read/decryption, threshold loading, image/face gates, SCRFD, official alignment, ArcFace, cosine decision; excludes image decoding and camera capture.'
    import torch, onnxruntime as ort
    report['hardware'] = {'gpu':torch.cuda.get_device_name(config.device_id), 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(), 'onnxruntime':ort.__version__}
    return report
