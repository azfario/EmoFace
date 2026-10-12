"""Consented local diagnostic: no raw portraits/vectors in public artifacts."""
import argparse
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models,detect_single_face,generate_embedding,Status
from face_verification.gpu import inspect_profile
from face_verification.benchmark import measure

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-consented-adult-data',action='store_true')
    parser.add_argument('--prepare-photo-regions',action='store_true')
    parser.add_argument('--skip-latency',action='store_true')
    parser.add_argument('--private-dir',type=Path)
    args = parser.parse_args()
    if not args.confirm_consented_adult_data:
        parser.error('Explicit adult identity-verification consent is required.')
    private = args.private_dir or Path(os.environ.get('FACEATTEND_PRIVATE_INPUTS',str(Path(os.environ['LOCALAPPDATA'])/'FaceAttend/phase2/private')))
    captures = private/'captures'
    # User-supplied screenshots: keep only the main photograph, excluding gallery
    # UI and all thumbnail faces. No generated variants or enrollment duplication.
    entries = [
        {'id':'volunteer_001','source':'volunteer_001_source.jpeg','photo_region':[0,326,591,953]},
        {'id':'volunteer_002','source':'volunteer_002_source.jpeg','photo_region':[0,168,591,1096]},
        {'id':'volunteer_003','source':'volunteer_003_source.jpeg','photo_region':None},
    ]
    if args.prepare_photo_regions:
        for entry in entries:
            image = cv2.imread(str(captures/entry['source']))
            if image is None:
                raise ValueError('A consented source image is missing or unreadable.')
            if entry['photo_region']:
                x1,y1,x2,y2 = entry['photo_region']
                if x2 > image.shape[1] or y2 > image.shape[0]:
                    raise ValueError('Screenshot dimensions changed; review the photo region manually.')
                image = image[y1:y2,x1:x2].copy()
            if not cv2.imwrite(str(captures/(entry['id']+'_input.jpeg')),image,[cv2.IMWRITE_JPEG_QUALITY,98]):
                raise OSError('Private photo-region write failed.')
        manifest = {'schema_version':1,'adult_identity_verification_consent_confirmed_by_user':True,
            'research_terms_confirmed_by_user':True,'capture_count':3,'independent_probes':0,
            'purpose':'Initial detection/alignment/GPU smoke only; insufficient enrollment or calibration data',
            'entries':entries}
        (private/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
    config = load_config(root=ROOT)
    models = initialize_models(config,profile=True)
    results, accepted = [], []
    for entry in entries:
        image = cv2.imread(str(captures/(entry['id']+'_input.jpeg')))
        face = detect_single_face(image,models)
        value = {'input_index':len(results)+1,'input_resolution':list(image.shape[:2]) if image is not None else None,
            'photo_region_used':entry['photo_region'] is not None,'status':face.status.value,'message':face.message,
            'embedding_generated':False}
        if face.status == Status.OK:
            embedding = generate_embedding(face,models)
            value.update(quality=face.quality,embedding_generated=True,embedding_dimension=int(embedding.size),
                embedding_norm=float(np.linalg.norm(embedding)),alignment_shape=list(face.aligned.shape))
            accepted.append((image,face))
            del embedding
        results.append(value)
    # Both actual models must execute, even if image quality rejects every input.
    models.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    profiles = [inspect_profile(model.session.end_profiling()) for model in (models.detector,models.recognizer)]
    report = {'status':'REAL_IMAGE_DIAGNOSTIC_COMPLETED','input_count':len(results),'model_id':models.model_id,
        'results':results,'profiles':profiles,'raw_images_saved_to_project':False,'embeddings_persisted':False,
        'real_enrollment':'INSUFFICIENT_DATA_ONE_IMAGE_PER_SUBJECT','calibration':'INSUFFICIENT_DATA',
        'verification_acceptance':False,'threshold':None}
    (ROOT/'reports/phase2/private_image_smoke.json').write_text(json.dumps(report,indent=2))
    # Recreate unprofiled models so profiling overhead is absent from timings.
    if accepted and not args.skip_latency:
        del models
        models = initialize_models(config)
        image = accepted[0][0]
        face = detect_single_face(image,models)
        latency = {'status':'MODEL_COMPONENTS_MEASURED','batch_size':1,'warmups':20,'repetitions':100,
            'input_resolution':list(image.shape[:2]),'detector_input_resolution':[640,640],'recognizer_input_resolution':[112,112],
            'scrfd_only':measure(lambda:models.detector.detect(image,input_size=(640,640),max_num=0)),
            'embedding_only':measure(lambda:generate_embedding(face,models)),
            'detection_alignment_embedding':measure(lambda:generate_embedding(detect_single_face(image,models),models)),
            'full_verification':None,'full_verification_reason':'No 3–5 image enrollment, independent probes or calibrated threshold.',
            'image_decode_included':False,'scrfd_scope':'Official preprocessing, CUDA inference, transfer back, bbox/keypoint decode and NMS.',
            'embedding_scope':'Already aligned 112x112 crop; official preprocessing, CUDA inference, transfer back, normalization.',
            'pipeline_scope':'Image/face gates, detector, official alignment and normalized embedding; excludes store/threshold/similarity decision.',
            'webcam':'NOT_RUN'}
        import torch,onnxruntime as ort
        latency['hardware'] = {'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'torch':torch.__version__,'onnxruntime':ort.__version__}
        (ROOT/'reports/phase2/latency_report.json').write_text(json.dumps(latency,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
