"""Opt-in diagnostic processing; never grants identity verification/attendance."""
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from pathlib import Path
import tempfile
import time
import numpy as np
from .detector import detect_single_face
from .embedder import generate_embedding
from .enrollment import enroll_user,delete_enrollment
from .gpu import require_cuda
from .quality import validate_image,cosine_similarity,image_digest
from .schemas import Status,VerificationError
from .template_store import TemplateStore,validate_user_id

@dataclass
class DiagnosticResult:
    status: str
    message: str = ''
    detected_faces: int | None = None
    quality: dict | None = None
    embedding_dimension: int | None = None
    embedding_norm: float | None = None
    cosine_similarity: float | None = None
    timings_ms: dict | None = None
    operational_verification: str = 'UNAVAILABLE'
    operational_reason: str = 'THRESHOLD_NOT_CALIBRATED'
    verified: bool = False
    threshold: None = None
    def to_dict(self):
        return asdict(self)

def process_frame(image,models,*,adult_consent=False,identity_consent=False,claimed_user_id=None,store=None,
                  detector=detect_single_face,embedder=generate_embedding):
    """Compare only the selected reference, without a calibration/decision path.

    This interface intentionally differs from operational verification: it can
    return diagnostic cosine with operational UNAVAILABLE, always verified=False.
    Existing verify_claimed_identity's early calibration refusal stays intact.
    """
    start = time.perf_counter_ns()
    timings = {}
    result = DiagnosticResult(Status.MODEL_ERROR.value)
    try:
        if adult_consent is not True or identity_consent is not True:
            raise VerificationError(Status.CONSENT_REQUIRED,'Explicit adult identity-research consent is required.')
        mark = time.perf_counter_ns()
        validate_image(image,models.config)
        timings['input_validation'] = (time.perf_counter_ns()-mark)/1e6
        require_cuda(models.detector.session)
        require_cuda(models.recognizer.session)
        template = None
        if claimed_user_id is not None:
            validate_user_id(claimed_user_id)
            if store is None:
                raise VerificationError(Status.UNKNOWN_ID,'No enrollment store was supplied for the claimed ID.')
            mark = time.perf_counter_ns()
            template = store.load(claimed_user_id)  # Never list/search enrolled people.
            timings['reference_read_decrypt'] = (time.perf_counter_ns()-mark)/1e6
            if template['model_id'] != models.model_id:
                raise VerificationError(Status.MODEL_MISMATCH,'Reference model/policy differs; re-enroll.')
            if len(template['embedding']) != models.embedding_dim:
                raise VerificationError(Status.CORRUPT_TEMPLATE,'Reference dimension differs from the model.')
            if image_digest(image) in template['sample_hashes']:
                raise VerificationError(Status.DUPLICATE_IMAGE,'Use a fresh diagnostic probe, not an enrollment frame.')
        mark = time.perf_counter_ns()
        face = detector(image,models)
        timings['detection_quality_alignment'] = (time.perf_counter_ns()-mark)/1e6
        if face.status!=Status.OK:
            result.detected_faces = 0 if face.status==Status.NO_FACE else 1 if face.status==Status.LOW_QUALITY_IMAGE else None
            raise VerificationError(face.status,face.message)
        result.detected_faces=1
        result.quality=face.quality
        mark = time.perf_counter_ns()
        vector = embedder(face,models)
        timings['embedding'] = (time.perf_counter_ns()-mark)/1e6
        timings['decoded_frame_to_embedding'] = (time.perf_counter_ns()-start)/1e6
        result.embedding_dimension=int(len(vector))
        result.embedding_norm=float(np.linalg.norm(vector))
        result.status='EMBEDDING_GENERATED'
        if template is not None:
            mark = time.perf_counter_ns()
            result.cosine_similarity=cosine_similarity(vector,template['embedding'])
            timings['cosine_only']=(time.perf_counter_ns()-mark)/1e6
            result.status='DIAGNOSTIC_COMPARISON_COMPLETED'
        del vector
    except VerificationError as exc:
        result.status,result.message=exc.status.value,str(exc)
    except Exception:
        result.status,result.message=Status.MODEL_ERROR.value,'Research diagnostic processing failed.'
    timings['overall_decoded_frame_diagnostic']=(time.perf_counter_ns()-start)/1e6
    result.timings_ms=timings
    return result

@contextmanager
def temporary_enrollment(user_id,images,models,*,private_parent,key_env='FACEATTEND_TEMPLATE_KEY',
                         adult_consent=False,identity_consent=False,audit=None,
                         enroller=enroll_user,store_factory=TemplateStore):
    """Use an operator-provided env key; delete selected template/temp directory.

    The caller owns the secret's lifecycle. This context never prints or writes
    the key and never modifies/removes a pre-existing operator environment key.
    """
    if adult_consent is not True or identity_consent is not True:
        raise VerificationError(Status.CONSENT_REQUIRED,'Adult enrollment consent is required.')
    validate_user_id(user_id)
    parent = Path(private_parent)
    parent.mkdir(parents=True,exist_ok=True)
    audit = audit if audit is not None else {}
    audit.update(cleanup_attempted=False,templates_removed=False,directory_removed=False)
    store = None
    directory=None
    try:
        with tempfile.TemporaryDirectory(dir=parent,prefix='phase2a_session_') as directory:
            try:
                store=store_factory(directory,key_env=key_env,retention_days=models.config.retention_days)
                enrolled=enroller(user_id,images,store,models,adult_consent=True,identity_consent=True)
                audit['enrollment_status']=enrolled.status.value
                if enrolled.status!=Status.OK:
                    raise VerificationError(enrolled.status,enrolled.message)
                yield store
            finally:
                audit['cleanup_attempted']=True
                if store is not None:
                    removed=delete_enrollment(user_id,store)
                    audit['deletion_status']=removed.status.value
                    audit['templates_removed']=not store.exists(user_id)
    finally:
        audit['directory_removed']=directory is not None and not Path(directory).exists()

def summarize_timings(samples):
    if not samples:
        return None
    return {'sample_count':len(samples),'median_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95))}

def benchmark_diagnostic(image,models,*,claimed_user_id=None,store=None,warmups=20,repetitions=100):
    if warmups<1 or repetitions<30:
        raise ValueError('Require warmups and at least 30 measured repetitions.')
    def run():
        result=process_frame(image,models,adult_consent=True,identity_consent=True,claimed_user_id=claimed_user_id,store=store)
        if result.status not in ('EMBEDDING_GENERATED','DIAGNOSTIC_COMPARISON_COMPLETED'):
            raise VerificationError(Status(result.status),'Cannot benchmark a rejected/error fast path as successful inference.')
        return result
    for _ in range(warmups):
        run()
    buckets={}
    for _ in range(repetitions):
        result=run()
        for name,value in result.timings_ms.items():
            buckets.setdefault(name,[]).append(value)
    # Actual official SCRFD-only path includes its preprocessing, transfer, decode/NMS.
    detection=[]
    for _ in range(warmups):
        require_cuda(models.detector.session)
        models.detector.detect(image,input_size=(models.config.detection_size,)*2,max_num=0)
    for _ in range(repetitions):
        mark=time.perf_counter_ns()
        models.detector.detect(image,input_size=(models.config.detection_size,)*2,max_num=0)
        detection.append((time.perf_counter_ns()-mark)/1e6)
    return {'status':'NEW_PHASE2A_MEASUREMENTS','warmups':warmups,'repetitions':repetitions,'batch_size':1,
        'image_shape':list(image.shape),'clock':'perf_counter_ns; synchronous session.run returns CPU outputs',
        'timings':{name:summarize_timings(values) for name,values in buckets.items()},
        'scrfd_preprocessing_inference_postprocessing':summarize_timings(detection),
        'isolated_image_preprocessing':None,
        'preprocessing_reason':'Official SCRFD wrapper combines resize/blob conversion/inference/decoding; no duplicate preprocessing implementation was introduced.',
        'image_decode_included':False,'camera_capture_included':False,'gpu_profiling_enabled_during_timing':False,
        'full_operational_verification':None,
        'scope':'Decoded image validation, strict provider checks, optional selected-template decryption, SCRFD, existing QC, official alignment, ArcFace and optional cosine; no calibrated decision.'}
