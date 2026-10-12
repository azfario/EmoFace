"""User-operated local CUDA research demo; no recording or identity acceptance."""
import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4
import cv2
from cryptography.fernet import Fernet

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models,VerificationError,Status
from face_verification.gpu import inspect_profile
from face_verification.webcam import capture_one_frame
from face_verification.research_demo import process_frame,temporary_enrollment,benchmark_diagnostic,summarize_timings

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--webcam',action='store_true')
    source.add_argument('--image',type=Path)
    parser.add_argument('--confirm-adult-consent',action='store_true')
    parser.add_argument('--camera-index',type=int,default=0)
    parser.add_argument('--timeout',type=int,default=45)
    parser.add_argument('--claimed-id')
    parser.add_argument('--enrollment-images',type=Path,nargs='+')
    parser.add_argument('--private-dir',type=Path)
    parser.add_argument('--ephemeral-key',action='store_true',help='Operator explicitly chooses a fresh process-only session key; no key is printed/persisted.')
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--warmups',type=int,default=20)
    parser.add_argument('--repetitions',type=int,default=100)
    args=parser.parse_args()
    if not args.confirm_adult_consent:
        parser.error('Explicit adult identity-research consent is required before image/camera processing.')
    if args.enrollment_images and (not args.claimed_id or not args.private_dir):
        parser.error('Temporary enrollment requires an explicitly selected --claimed-id and an external --private-dir.')
    if args.enrollment_images and not 3<=len(args.enrollment_images)<=5:
        parser.error('Temporary enrollment requires 3–5 distinct images.')
    if args.private_dir and args.private_dir.resolve().is_relative_to(ROOT):
        parser.error('Use a private directory outside this OneDrive/project checkout.')
    if not 1<=args.timeout<=120 or args.camera_index<0 or args.warmups<1 or args.repetitions<30:
        parser.error('Invalid camera index, 1–120 second timeout, warmup or >=30 repetition setting.')
    token=uuid4().hex[:12]
    report={'run_id':token,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'mode':'USER_INITIATED_WEBCAM' if args.webcam else 'CONSENTED_LOCAL_IMAGE',
        'raw_frame_saved':False,'embeddings_saved_to_reports':False,'keys_persisted':False,
        'verified':False,'threshold':None,'operational_verification':'UNAVAILABLE','operational_reason':'THRESHOLD_NOT_CALIBRATED',
        'camera_test':'NOT_REQUESTED','cleanup':{},'benchmark':None}
    old_key=None
    key_installed=False
    key_name=None
    models=None
    camera_metrics={}
    try:
        print('Loading existing strict CUDA models; no weights downloaded.',flush=True)
        models=initialize_models(load_config(root=ROOT),profile=True)
        report['model_id']=models.model_id
        with ExitStack() as stack:
            store=None
            if args.enrollment_images:
                if args.ephemeral_key:
                    key_name=models.config.key_env
                    old_key=os.environ.get(models.config.key_env)
                    os.environ[models.config.key_env]=Fernet.generate_key().decode()
                    key_installed=True
                enrollment_images=[cv2.imread(str(p)) for p in args.enrollment_images]
                store=stack.enter_context(temporary_enrollment(args.claimed_id,enrollment_images,models,
                    private_parent=args.private_dir,key_env=models.config.key_env,
                    adult_consent=True,identity_consent=True,audit=report['cleanup']))
                del enrollment_images
            interaction_start=time.perf_counter_ns()
            if args.webcam:
                report['camera_test']='STARTED'
                print('SPACE: capture once | ESC/Q: cancel | close window: stop',flush=True)
                image=capture_one_frame(adult_consent=True,identity_consent=True,user_initiated=True,
                    camera_index=args.camera_index,timeout_seconds=args.timeout,metrics=camera_metrics)
                report['capture_wall_ms']=(time.perf_counter_ns()-interaction_start)/1e6
                report['capture_wall_scope']='Camera open, preview and human button wait; excludes model initialization/enrollment.'
                report['camera_test']=camera_metrics.get('stop_reason','UNKNOWN')
                report['camera_metrics']={k:v for k,v in camera_metrics.items() if k!='read_durations_ms'}
                report['camera_frame_read_timings']=summarize_timings(camera_metrics.get('read_durations_ms',[]))
                if image is None:
                    report['status']='CAPTURE_CANCELLED'
            else:
                decode=time.perf_counter_ns()
                image=cv2.imread(str(args.image))
                report['image_decode_ms']=(time.perf_counter_ns()-decode)/1e6
            if image is not None or not args.webcam:
                result=process_frame(image,models,adult_consent=True,identity_consent=True,
                    claimed_user_id=args.claimed_id,store=store)
                report.update(result.to_dict())
                report['captured_image_shape']=list(image.shape) if image is not None else None
                if args.webcam:
                    report['capture_to_diagnostic_wall_ms']=(time.perf_counter_ns()-interaction_start)/1e6
                    report['capture_to_embedding_wall_ms']=report['capture_wall_ms']+result.timings_ms['decoded_frame_to_embedding'] if result.embedding_dimension is not None else None
                if args.benchmark and result.status in ('EMBEDDING_GENERATED','DIAGNOSTIC_COMPARISON_COMPLETED'):
                    # Finish profiling first; timing uses independent unprofiled
                    # sessions with identical weights/input/quality policy.
                    report['gpu_profiles']={role:inspect_profile(model.session.end_profiling()) for role,model in [('detector',models.detector),('recognizer',models.recognizer)]}
                    models=None
                    timing_models=initialize_models(load_config(root=ROOT))
                    report['benchmark']=benchmark_diagnostic(image,timing_models,claimed_user_id=args.claimed_id,store=store,
                        warmups=args.warmups,repetitions=args.repetitions)
                del image
            if models is not None:
                profiles={}
                for role,model in [('detector',models.detector),('recognizer',models.recognizer)]:
                    path=model.session.end_profiling()
                    try:
                        profiles[role]=inspect_profile(path)
                    except VerificationError:
                        events=json.loads(Path(path).read_text())
                        if any(e.get('cat')=='Node' and e.get('args',{}).get('provider') for e in events):
                            raise  # Computational non-CUDA events must fail clearly.
                        profiles[role]={'status':'NO_SUCCESSFUL_COMPUTE_PROFILE','reason':'Model may not have executed because capture, reference or QC was rejected.'}
                report['gpu_profiles']=profiles
    except VerificationError as exc:
        report.update(status=exc.status.value,message=str(exc))
    except Exception:
        report.update(status='LOCAL_DEMO_ERROR',message='Local camera, enrollment or model processing failed. Check permissions, enrollment images and OpenCV GUI support.')
    finally:
        if key_installed:
            if old_key is None:
                os.environ.pop(key_name,None)
            else:
                os.environ[key_name]=old_key
        if camera_metrics:
            report['camera_metrics']={k:v for k,v in camera_metrics.items() if k!='read_durations_ms'}
        report['session_key_removed']=key_installed
    import torch,onnxruntime as ort
    report['hardware']={'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'onnxruntime':ort.__version__}
    path=ROOT/'reports/phase2'/f'phase2a_demo_{token}.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report.get('status') in ('EMBEDDING_GENERATED','DIAGNOSTIC_COMPARISON_COMPLETED','CAPTURE_CANCELLED') else 2

if __name__=='__main__':
    sys.exit(main())
