"""Private consent manifest -> independent-session scores; never saves raw images."""
import json
from pathlib import Path
import cv2
from .detector import detect_single_face
from .embedder import generate_embedding
from .enrollment import aggregate_embeddings
from .quality import image_digest, cosine_similarity
from .calibration import Pair, validate_pairs
from .schemas import Status, VerificationError
from .template_store import validate_user_id

def load_protocol(path, *, consent_confirmed=False):
    if consent_confirmed is not True:
        raise VerificationError(Status.CONSENT_REQUIRED, 'Confirm consented adult data availability before opening a private manifest.')
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('authorized_noncommercial_research') is not True:
        raise ValueError('Manifest must record the authorized research purpose.')
    participants = manifest.get('participants', [])
    seen, paths = set(), set()
    for person in participants:
        user = validate_user_id(person['user_id'])
        if user in seen or person['split'] not in ('development', 'test'):
            raise ValueError('Each pseudonym must appear once in exactly one partition.')
        seen.add(user)
        if person.get('adult_consent') is not True or person.get('identity_consent') is not True:
            raise VerificationError(Status.CONSENT_REQUIRED, 'Every participant must consent as an adult.')
        if not 3 <= len(person['enrollment']) <= 5 or not person['probes']:
            raise ValueError('Each identity needs 3–5 enrollment images and independent probes.')
        enrollment_sessions = {record['session'] for record in person['enrollment']}
        if len(enrollment_sessions) != 1 or not all(enrollment_sessions):
            raise ValueError('Enrollment records must name one nonempty capture-session ID.')
        for role in ('enrollment', 'probes'):
            for record in person[role]:
                if not record['session'] or (role == 'probes' and record['session'] in enrollment_sessions):
                    raise ValueError('Probe sessions must differ from enrollment sessions.')
                image_path = (path.parent / record['path']).resolve()
                if image_path in paths:
                    raise ValueError('An image path is repeated across participants or roles.')
                paths.add(image_path)
                if not image_path.is_file():
                    raise ValueError('A manifest image is missing; correct the private manifest.')
                record['_resolved_path'] = image_path
    return manifest

def score_protocol(manifest, model_bundle):
    """Caller must use load_protocol's explicit consent gate first."""
    samples = {'development': [], 'test': []}
    seen_hashes = set()
    for person in manifest['participants']:
        if person.get('adult_consent') is not True or person.get('identity_consent') is not True:
            raise VerificationError(Status.CONSENT_REQUIRED, 'Missing adult verification consent.')
        vectors, hashes, probes = [], [], []
        for role in ('enrollment', 'probes'):
            for record in person[role]:
                image = cv2.imread(str(record['_resolved_path']),cv2.IMREAD_COLOR)
                face = detect_single_face(image,model_bundle)
                if face.status != Status.OK:
                    raise VerificationError(face.status, 'A private protocol capture failed the image/face gate; recollect it before evaluation.')
                digest = image_digest(image)
                if digest in seen_hashes:
                    raise ValueError('Duplicate decoded image content across protocol roles or subjects.')
                seen_hashes.add(digest)
                embedding = generate_embedding(face,model_bundle)
                if role == 'enrollment':
                    vectors.append(embedding)
                    hashes.append(digest)
                else:
                    probes.append((embedding,digest,record['session']))
        samples[person['split']].append({'subject':person['user_id'],
            'template':aggregate_embeddings(vectors,model_bundle.config.consistency_cosine),
            'hashes':tuple(hashes),'session':person['enrollment'][0]['session'],'probes':probes})
    output = {}
    for split, records in samples.items():
        pairs = []
        for reference in records:
            for submitter in records:
                for embedding,digest,session in submitter['probes']:
                    # Prefix subject to avoid unrelated people using the same session label.
                    pairs.append(Pair(cosine_similarity(reference['template'],embedding),
                        reference['subject']==submitter['subject'], reference['subject'],submitter['subject'],
                        reference['hashes'],digest,reference['subject']+':'+reference['session'],submitter['subject']+':'+session))
        validate_pairs(pairs)
        output[split] = pairs
    return output
