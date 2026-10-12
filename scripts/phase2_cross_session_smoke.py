"""Private single-subject cross-session enrollment/probe diagnostics."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import cv2
import numpy as np
from cryptography.fernet import Fernet
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models,detect_single_face,generate_embedding,enroll_user,verify_claimed_identity,delete_enrollment,TemplateStore,Status
from face_verification.gpu import inspect_profile
from face_verification.quality import image_digest,cosine_similarity

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--private-display-label')
    parser.add_argument('--confirm-consented-adult-data',action='store_true')
    parser.add_argument('--confirm-each-photo-separate-session',action='store_true')
    parser.add_argument('--include-prior-labeled-controls',action='store_true')
    args = parser.parse_args()
    if not args.confirm_consented_adult_data or not args.confirm_each_photo_separate_session:
        parser.error('Explicit adult research authorization and user-confirmed separate sessions are required.')
    base = args.private_dir.resolve()
    private = base/'batch_03'
    captures = private/'captures'
    captures.mkdir(parents=True,exist_ok=True)
    user_id = 'volunteer_006'
    enrollment = [1,2,7]
    probes = [3,4,5,6,8,9,10]
    mapping = []
    for number in range(1,11):
        source = args.source_dir/f'{number}-Photo-{number}.jpg'
        target = captures/f'{user_id}_capture_{number:02d}.jpg'
        if not source.is_file():
            raise ValueError('A supplied attachment is missing.')
        if target.exists() and target.read_bytes()!=source.read_bytes():
            raise ValueError('A different capture occupies the selected private path.')
        if not target.exists():
            shutil.copyfile(source,target)
        mapping.append({'photo':number,'path':str(target),'role':'enrollment' if number in enrollment else 'probe',
            'session':f'user_confirmed_capture_session_{number:02d}'})
    manifest = {'schema_version':1,'subject_id':user_id,'label_supplied_by_user':args.private_display_label,
        'adult_research_context_confirmed':True,'distinct_sessions_confirmed_by_user':True,
        'session_dates':'Not supplied; anonymous session labels only', 'enrollment':enrollment,'probes':probes,'captures':mapping}
    (private/'manifest.json').write_text(json.dumps(manifest,indent=2))
    config = load_config(root=ROOT)
    models = initialize_models(config,profile=True)
    images,vectors,observations,hashes = {},{},[],set()
    for record in mapping:
        image = cv2.imread(record['path'])
        if image is None:
            raise ValueError('A supplied image did not decode.')
        digest = image_digest(image)
        if digest in hashes:
            raise ValueError('Duplicate decoded photo content across enrollment/probe roles.')
        hashes.add(digest)
        images[record['photo']] = image
        face = detect_single_face(image,models)
        row = {'photo':record['photo'],'role':record['role'],'session_index':record['photo'],
            'status':face.status.value,'message':face.message,'quality':face.quality,
            'decoded_image_shape':list(image.shape),'embedding_generated':False}
        if face.status==Status.OK:
            vector = generate_embedding(face,models)
            vectors[record['photo']] = vector
            row.update(embedding_generated=True,embedding_dimension=len(vector),embedding_norm=float(np.linalg.norm(vector)))
        observations.append(row)
    previous_key = os.environ.get(config.key_env)
    os.environ[config.key_env] = Fernet.generate_key().decode()
    attempts,comparisons,control_results,removal = [],[],[],[]
    try:
        with tempfile.TemporaryDirectory(dir=private,prefix='encrypted_cross_session_smoke_') as directory:
            store = TemplateStore(directory,key_env=config.key_env,retention_days=config.retention_days)
            no_threshold = replace(config,threshold_file='reports/phase2/batch03_no_calibration.json')
            if no_threshold.path(no_threshold.threshold_file).exists():
                raise ValueError('Unexpected batch threshold artifact; stop.')
            result = enroll_user(user_id,[images[n] for n in enrollment],store,models,adult_consent=True,identity_consent=True)
            enrollment_result = {'subject_id':user_id,'photos':enrollment,'status':result.status.value,
                'message':result.message,'sample_count':result.sample_count}
            if result.status==Status.OK:
                template = store.load(user_id)
                enrollment_result.update(encrypted_roundtrip=True,template_dimension=len(template['embedding']),
                    template_norm=float(np.linalg.norm(template['embedding'])),
                    duplicate_status=enroll_user(user_id,[images[n] for n in enrollment],store,models,adult_consent=True,identity_consent=True).status.value)
            # Previous two labeled participants are explicit negative controls;
            # no identity search or comparison with uninformed people is performed.
            if args.include_prior_labeled_controls:
                for control,numbers in [('volunteer_004',[1,2,4]),('volunteer_005',[2,3,4])]:
                    control_images = [cv2.imread(str(base/'batch_02/captures'/f'{control}_capture_{n:02d}.jpg')) for n in numbers]
                    result = enroll_user(control,control_images,store,models,adult_consent=True,identity_consent=True)
                    control_results.append({'subject_id':control,'status':result.status.value})
            for number in probes:
                attempt = verify_claimed_identity(user_id,images[number],store,models,no_threshold,adult_consent=True,identity_consent=True)
                attempts.append({'photo':number,'status':attempt.status.value,'verified':attempt.verified,
                    'score':attempt.score,'threshold':attempt.threshold})
                if number not in vectors:
                    continue
                references = [user_id]+[c['subject_id'] for c in control_results if c['status']=='OK']
                for reference in references:
                    if not store.exists(reference):
                        continue
                    value = store.load(reference)
                    comparisons.append({'probe_photo':number,'reference_id':reference,
                        'label':'genuine' if reference==user_id else 'impostor',
                        'score':cosine_similarity(value['embedding'],vectors[number]),
                        'capture_sessions_disjoint_from_reference':True if reference==user_id else None,
                        'session_provenance':'Confirmed by user; exact dates unavailable' if reference==user_id else 'Different user-labeled subjects; cross-subject session chronology unconfirmed',
                        'decision':None,'used_for_calibration':False})
            for reference in [user_id]+[c['subject_id'] for c in control_results]:
                removal.append({'subject_id':reference,'status':delete_enrollment(reference,store).status.value})
            templates_remaining = len(list(Path(directory).glob('*.fernet')))
            assert templates_remaining==0
    finally:
        if previous_key is None:
            os.environ.pop(config.key_env,None)
        else:
            os.environ[config.key_env]=previous_key
    models.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    profiles = {role:inspect_profile(model.session.end_profiling()) for role,model in [('detector',models.detector),('recognizer',models.recognizer)]}
    report = {'status':'CROSS_SESSION_SINGLE_SUBJECT_DIAGNOSTIC_COMPLETED','batch':'03','subject_id':user_id,
        'input_count':10,'sessions':10,'separate_sessions_confirmed_by_user':True,'session_dates_available':False,
        'selection_fixed_before_scores':True,'quality_policy_unchanged':True,'model_id':models.model_id,
        'observations':observations,'quality_passed':sum(r['status']=='OK' for r in observations),
        'enrollment_smoke':enrollment_result,'prior_control_enrollments':control_results,
        'verification_attempts':attempts,'descriptive_probe_comparisons':comparisons,
        'genuine_diagnostic_comparisons':sum(c['label']=='genuine' for c in comparisons),
        'impostor_diagnostic_comparisons':sum(c['label']=='impostor' for c in comparisons),
        'template_deletion':removal,'templates_remaining':templates_remaining,
        'calibration_status':'INSUFFICIENT_DATA','threshold':None,'fmr':None,'fnmr':None,
        'reason':'One added subject plus two prior controls cannot support independent subject-disjoint development/test calibration or population error-rate estimates.',
        'gpu_profiles':profiles,'private_images_outside_project':True,'keys_persisted':False,
        'embeddings_persisted_after_smoke':False,'photos_or_vectors_uploaded_to_github':False,
        'notes':['All originals tested as decoded, without manually cropping out other faces or correcting arbitrary rotation.',
            'No largest-face selection or quality/consistency gate relaxation.',
            'Rejected images have no embedding or similarity score. All scores are diagnostics, not match decisions.']}
    (ROOT/'reports/phase2/batch03_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
