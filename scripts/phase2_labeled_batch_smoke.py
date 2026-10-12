"""Rerun a user-labeled, same-session private cohort without calibration claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import os
import cv2
import numpy as np
from cryptography.fernet import Fernet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from face_verification import load_config,initialize_models,detect_single_face,generate_embedding,enroll_user,verify_claimed_identity,delete_enrollment,TemplateStore,Status
from face_verification.gpu import inspect_profile
from face_verification.quality import image_digest,cosine_similarity

def summary(scores):
    return {'count':len(scores),'min':float(min(scores)) if scores else None,
        'median':float(np.median(scores)) if scores else None,'max':float(max(scores)) if scores else None}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--private-dir',type=Path,required=True)
    parser.add_argument('--confirm-consented-adult-data',action='store_true')
    args = parser.parse_args()
    if not args.confirm_consented_adult_data:
        parser.error('Confirm the authorized adult research context before processing these images.')
    private = args.private_dir.resolve()/'batch_02'
    captures = private/'captures'
    captures.mkdir(parents=True,exist_ok=True)
    groups = [
        {'id':'volunteer_004','photos':[1,2,3,4,5],'enrollment':[1,2,4],'probes':[3,5]},
        {'id':'volunteer_005','photos':[6,7,8,9,10],'enrollment':[7,8,9],'probes':[6,10]},
    ]
    # Selection is fixed before score inspection: front/quarter poses enroll;
    # profiles exercise the probe quality gate. No capture is reused across roles.
    mapping = []
    for person in groups:
        for index,number in enumerate(person['photos'],1):
            source = args.source_dir/f'{number}-Photo-{number}.jpg'
            if not source.is_file():
                raise ValueError('A supplied attachment is missing; no replacement is inferred.')
            target = captures/f"{person['id']}_capture_{index:02d}.jpg"
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError('A different private capture already exists at this batch path.')
            if not target.exists():
                shutil.copyfile(source,target)
            mapping.append({'photo':number,'id':person['id'],'path':str(target),
                'role':'enrollment' if number in person['enrollment'] else 'probe','session':'single_session_user_confirmed'})
    # Preserve an existing private label mapping without embedding participant
    # names in source code or public report fields.
    private_groups = groups
    if (private/'manifest.json').exists():
        previous = json.loads((private/'manifest.json').read_text())
        labels = {g['id']:g.get('display_name') for g in previous.get('groups',[]) if g.get('display_name')}
        private_groups = [{**g,**({'display_name':labels[g['id']]} if g['id'] in labels else {})} for g in groups]
    (private/'manifest.json').write_text(json.dumps({'schema_version':1,'adult_research_consent_context':'confirmed in current Phase 2 conversation',
        'identity_labels_supplied_by_user':True,'independent_sessions':False,'groups':private_groups,'captures':mapping},indent=2))
    config = load_config(root=ROOT)
    models = initialize_models(config,profile=True)
    observations,images,vectors,hashes = [],{},{},set()
    for record in sorted(mapping,key=lambda r:r['photo']):
        image = cv2.imread(record['path'])
        if image is None:
            raise ValueError('A supplied capture failed decoding.')
        digest = image_digest(image)
        if digest in hashes:
            raise ValueError('Duplicate decoded capture in the supplied batch.')
        hashes.add(digest)
        images[record['photo']] = image
        face = detect_single_face(image,models)
        observation = {'photo':record['photo'],'subject_id':record['id'],'role':record['role'],
            'status':face.status.value,'message':face.message,'image_shape':list(image.shape),
            'quality':face.quality,'embedding_generated':False}
        if face.status==Status.OK:
            vector = generate_embedding(face,models)
            vectors[record['photo']] = vector
            observation.update(embedding_generated=True,embedding_dimension=len(vector),embedding_norm=float(np.linalg.norm(vector)))
        observations.append(observation)
    # Encrypted real-data enrollment smoke is transient. Key stays only in process
    # memory/environment; all test templates are explicitly deleted before exit.
    original_key = os.environ.get(config.key_env)
    os.environ[config.key_env] = Fernet.generate_key().decode()
    enrollments,attempts,removals,probe_descriptive = [],[],[],[]
    try:
        with tempfile.TemporaryDirectory(dir=private,prefix='encrypted_smoke_') as store_directory:
            store = TemplateStore(store_directory,key_env=config.key_env,retention_days=config.retention_days)
            # Isolated config prevents any previously issued cutoff from authorizing
            # this small same-session cohort. Models/quality policy remain unchanged.
            no_threshold = replace(config,threshold_file='reports/phase2/batch02_no_calibration.json')
            if no_threshold.path(no_threshold.threshold_file).exists():
                raise ValueError('Unexpected cohort cutoff artifact; stop rather than use it.')
            for person in groups:
                result = enroll_user(person['id'],[images[n] for n in person['enrollment']],store,models,
                    adult_consent=True,identity_consent=True)
                entry = {'subject_id':person['id'],'enrollment_photos':person['enrollment'],
                    'status':result.status.value,'message':result.message,'sample_count':result.sample_count}
                if result.status==Status.OK:
                    saved = store.load(person['id'])
                    entry.update(template_dimension=len(saved['embedding']),template_norm=float(np.linalg.norm(saved['embedding'])),
                        encrypted_roundtrip=True,duplicate_id_status=enroll_user(person['id'],[images[n] for n in person['enrollment']],store,models,adult_consent=True,identity_consent=True).status.value)
                enrollments.append(entry)
                for number in person['probes']:
                    result = verify_claimed_identity(person['id'],images[number],store,models,no_threshold,
                        adult_consent=True,identity_consent=True)
                    attempts.append({'claimed_subject_id':person['id'],'photo':number,'status':result.status.value,
                        'verified':result.verified,'score':result.score,'threshold':result.threshold})
            # Known labeled comparisons only, not identity search. Enrollment and
            # query roles are distinct, but all captures are from the same session.
            for submitter in groups:
                for number in submitter['probes']:
                    if number not in vectors:
                        continue
                    for reference in groups:
                        if store.exists(reference['id']):
                            template = store.load(reference['id'])
                            probe_descriptive.append({'reference_id':reference['id'],'probe_photo':number,
                                'label':'genuine' if reference['id']==submitter['id'] else 'impostor',
                                'score':cosine_similarity(template['embedding'],vectors[number]),
                                'decision':None,'same_session':True,'used_for_calibration':False})
            for person in groups:
                removals.append({'subject_id':person['id'],'status':delete_enrollment(person['id'],store).status.value})
            remaining_templates = len(list(Path(store_directory).glob('*.fernet')))
            assert remaining_templates==0
    finally:
        if original_key is None:
            os.environ.pop(config.key_env,None)
        else:
            os.environ[config.key_env] = original_key
    # Descriptive, explicitly labeled pair scores on quality-passing captures only.
    # They never enter calibration or authorize a match; same-session dependence
    # precludes held-out FMR/FNMR estimation.
    genuine,impostor = [],[]
    a,b = groups
    for person in groups:
        numbers = [n for n in person['photos'] if n in vectors]
        for i,n in enumerate(numbers):
            for m in numbers[i+1:]:
                genuine.append(cosine_similarity(vectors[n],vectors[m]))
    for n in a['photos']:
        for m in b['photos']:
            if n in vectors and m in vectors:
                impostor.append(cosine_similarity(vectors[n],vectors[m]))
    # Ensure both actual sessions have execution evidence even if QC rejects all.
    models.recognizer.get_feat(np.zeros((112,112,3),np.uint8))
    profiles = {role:inspect_profile(model.session.end_profiling()) for role,model in [('detector',models.detector),('recognizer',models.recognizer)]}
    report = {'status':'SAME_SESSION_LABELED_DIAGNOSTIC_COMPLETED','batch':'02','input_count':10,'subjects':2,
        'capture_sessions':'ONE_SESSION_CONFIRMED_BY_USER','selection_fixed_before_scores':True,
        'model_id':models.model_id,'quality_policy_unchanged':True,'observations':observations,
        'quality_passed':sum(r['status']=='OK' for r in observations),'enrollment_smoke':enrollments,
        'verification_attempts':attempts,'template_deletion':removals,'templates_remaining':remaining_templates,
        'same_session_probe_comparisons':probe_descriptive,
        'descriptive_scores':{'genuine':summary(genuine),'impostor':summary(impostor),
            'note':'Dependent same-session all-pair description of QC-passing images; includes enrollment captures. Not development/final-test biometric accuracy.'},
        'calibration_status':'INSUFFICIENT_DATA','threshold':None,'fmr':None,'fnmr':None,
        'reason':'Only two identities, one capture session, no independent probes or subject-disjoint development/test sets.',
        'gpu_profiles':profiles,'private_images_outside_project':True,'embeddings_persisted_after_smoke':False,
        'keys_persisted':False,'photos_or_vectors_uploaded_to_github':False}
    (ROOT/'reports/phase2/batch02_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
