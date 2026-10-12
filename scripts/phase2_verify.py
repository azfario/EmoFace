"""Fresh-process 1:1 verification, or a safe no-data import smoke."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config, initialize_models, TemplateStore, verify_claimed_identity, VerificationError, Status

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-no-data',action='store_true')
    parser.add_argument('--user-id')
    parser.add_argument('--image',type=Path)
    parser.add_argument('--confirm-adult-verification-consent',action='store_true')
    args = parser.parse_args()
    config = load_config(root=ROOT)
    if args.smoke_no_data:
        try:
            bundle = initialize_models(config,profile=True)
            import numpy as np
            from face_verification.gpu import inspect_profile
            bundle.detector.detect(np.zeros((640,640,3),np.uint8),input_size=(640,640),max_num=0)
            bundle.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
            print(json.dumps({'status':'REAL_MODELS_SYNTHETIC_INPUT_SMOKE','profiles': [inspect_profile(m.session.end_profiling()) for m in (bundle.detector,bundle.recognizer)]}))
        except VerificationError as exc:
            print(json.dumps({'status':exc.status.value,'message':str(exc),'real_face_inference':'NOT_RUN'}))
            if exc.status != Status.MODEL_ERROR:
                return 2
        return 0
    if not args.user_id or not args.image or not args.confirm_adult_verification_consent:
        parser.error('Provide a claimed pseudonym, fresh image and explicit adult verification consent.')
    try:
        bundle = initialize_models(config)
        store = TemplateStore(config.path(config.store_dir),key_env=config.key_env,retention_days=config.retention_days)
        import cv2
        image = cv2.imread(str(args.image))
        result = verify_claimed_identity(args.user_id,image,store,bundle,adult_consent=True,identity_consent=True)
        print(json.dumps(result.to_dict()))
        return 0 if result.status in (Status.VERIFIED,Status.NOT_VERIFIED) else 2
    except VerificationError as exc:
        print(json.dumps({'status':exc.status.value,'message':str(exc)}))
        return 2

if __name__=='__main__':
    sys.exit(main())
