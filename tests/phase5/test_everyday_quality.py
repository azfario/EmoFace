"""Everyday-profile boundaries and end-to-end demo compatibility, using synthetic images."""
import io
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image
import pytest
from face_verification.config import Config
from face_verification.demo_quality import demo_face_bundle
from face_verification.detector import detect_single_face
from face_verification.quality import assess_face, validate_image
from face_verification.schemas import Status, VerificationError
from tests.phase4b.conftest import fields
from .conftest import CONSENT, add_reference, upload

BOX=np.array([30.,30.,170.,170.])
POINTS=np.array([[65.,70.],[135.,70.],[100.,100.],[75.,135.],[125.,135.]])


def noise(seed=0, mean=128, sigma=10, side=200):
    grey=np.clip(np.random.default_rng(seed).normal(mean,sigma,(side,side)),1,254).astype(np.uint8)
    return np.repeat(grey[:,:,None],3,axis=2)


def assess(image,box=BOX,points=POINTS,*,strict=False):
    original=SimpleNamespace(config=Config(root=Path('.')),model_id='fixture-base')
    bundle=original if strict else demo_face_bundle(original)
    validate_image(image,bundle.config)
    return assess_face(image,box,points,bundle.config,getattr(bundle,'quality_policy',None))


def encoded(image):
    stream=io.BytesIO();Image.fromarray(image).save(stream,format='PNG')
    return ('fictional-reference',stream.getvalue(),'image/png')


@pytest.mark.parametrize('case',['blur','dim','bright','small_face','pose','clipping','landmarks','resolution'])
def test_ordinary_photos_pass_demo_but_fail_strict(case):
    image,box,points=noise(),BOX.copy(),POINTS.copy()
    if case=='blur':image=noise(sigma=1.3)
    elif case=='dim':image=noise(mean=25,sigma=2)
    elif case=='bright':image=noise(mean=235,sigma=2)
    elif case=='small_face':
        box=np.array([50.,50.,110.,110.]);points=np.array([[65.,65.],[95.,65.],[80.,80.],[65.,95.],[95.,95.]])
    elif case=='pose':points[:2]=[[93.,70.],[107.,70.]]
    elif case=='clipping':box[0]=-8
    elif case=='landmarks':points[0,0]=22
    elif case=='resolution':
        image=noise(side=128);box=np.array([20.,20.,110.,110.]);points=POINTS*.6
    with pytest.raises(VerificationError):assess(image,box,points,strict=True)
    quality=assess(image,box,points)
    if case=='blur':assert 20<=quality['blur_variance']<60


@pytest.mark.parametrize('case,reason',[('blur','blur'),('dim','lighting'),('bright','lighting'),
    ('small_face','face_size'),('pose','pose'),('clipping','clipping'),('landmarks','landmarks'),('resolution','resolution')])
def test_unusable_photos_still_fail_with_fixed_reason(case,reason):
    image,box,points=noise(),BOX.copy(),POINTS.copy()
    if case=='blur':image=np.full((200,200,3),128,np.uint8)
    elif case=='dim':image=noise(mean=10,sigma=2)
    elif case=='bright':image=noise(mean=248,sigma=2)
    elif case=='small_face':box=np.array([60.,60.,100.,100.])
    elif case=='pose':points[:2]=[[95.,70.],[100.,70.]]
    elif case=='clipping':box[0]=-40
    elif case=='landmarks':points[0,0]=-1
    elif case=='resolution':image=noise(side=127)
    with pytest.raises(VerificationError) as error:assess(image,box,points)
    assert error.value.quality_reason==reason


def test_profile_is_distinct_idempotent_and_does_not_mutate_models(context):
    original=context.bundle;demo=demo_face_bundle(original)
    assert demo.model_id!=original.model_id and demo.model_id==demo_face_bundle(original).model_id
    assert demo_face_bundle(demo) is demo
    assert demo.detector is original.detector and demo.recognizer is original.recognizer and demo.align is original.align
    assert original.config.min_blur_variance==60 and demo.config.min_blur_variance==20
    assert not hasattr(original,'quality_policy')


def test_clipped_boxes_are_clamped_and_unusable_alignment_is_rejected(context):
    context.bundle.detector.mode='clipped';image=noise();demo=demo_face_bundle(context.bundle)
    face=detect_single_face(image,demo)
    assert face.status==Status.OK and face.bbox[0]==0
    assert detect_single_face(image,context.bundle).status==Status.LOW_QUALITY_IMAGE
    demo.align=lambda *a,**k:np.zeros((112,112,3),np.uint8)
    rejected=detect_single_face(image,demo)
    assert rejected.status==Status.LOW_QUALITY_IMAGE and rejected.quality_reason=='landmarks'


def test_error_identifies_second_photo_and_removes_failed_stage(ui):
    photos=[encoded(noise(1)),encoded(np.full((200,200,3),128,np.uint8)),encoded(noise(3))]
    response=ui.client.post('/api/v1/demo/enrollments',headers=ui.headers,data=fields(),
        files=[('enrollment_images',photo) for photo in photos])
    assert response.status_code==422
    assert response.json()=={'status':'LOW_QUALITY_IMAGE','message':'The image failed the existing face-quality policy.',
                            'photo_number':2,'quality_reason':'blur'}
    assert not ui.gallery.repository.rows and not list(ui.gallery.files.staging.iterdir())


def test_tolerant_enrollment_reference_inspection_preview_matching_and_reenrollment(ui):
    ui.context.bundle.detector.mode='clipped'
    photos=[encoded(noise(seed,sigma=1.3)) for seed in (1,2,3)]
    def save(path):return ui.client.post(path,headers=ui.headers,data=fields(),files=[('enrollment_images',p) for p in photos])
    response=save('/api/v1/demo/enrollments');assert response.status_code==200,response.text
    identity=response.json()['identity_id'];profile=demo_face_bundle(ui.context.bundle)
    assert next(iter(ui.gallery.repository.rows.values())).model_policy_fingerprint==profile.model_id
    assert ui.client.post('/api/v1/demo/enrollments/'+identity+'/inspect-reference',headers=ui.headers).status_code==200
    response=ui.client.post('/api/v1/demo/preview-face?include_candidates=true&include_face_box=true',
        headers=ui.headers,data=CONSENT,files={'image':photos[0]})
    assert response.status_code==200,response.text
    face=response.json()['faces'][0]
    assert face['face_box']['x']==0 and face['identity_diagnostic']['candidate_count']==1
    candidates=ui.client.post('/api/v1/demo/candidates',headers=ui.headers,data=CONSENT,files={'image':photos[0]})
    assert candidates.status_code==200 and candidates.json()['candidate_count']==1
    ui.headers['x-faceattend-blur-opt-in']='1'
    combined=ui.client.post('/api/v1/demo/analyze-face?allow_blurry_expression=true&include_candidates=true',
        headers=ui.headers,data=CONSENT,files={'image':photos[0]})
    assert combined.status_code==200 and combined.json()['faces'][0]['identity_diagnostic']['candidate_count']==1
    assert 'input_quality' not in combined.json()['faces'][0]
    # The standalone strict endpoint still rejects the same clipping/blur.
    strict=ui.client.post('/api/v1/face/detect',data=CONSENT,files={'image':photos[0]})
    assert strict.status_code==422
    assert ui.context.bundle.config.min_blur_variance==60
    reenrolled=save('/api/v1/demo/enrollments/'+identity+'/reenroll')
    assert reenrolled.status_code==200 and reenrolled.json()['identity_id']==identity


def test_old_policy_references_are_preserved_but_excluded(ui):
    row=add_reference(ui.gallery,model_id=ui.context.bundle.model_id)
    response=ui.client.post('/api/v1/demo/preview-face?include_candidates=true',headers=ui.headers,data=CONSENT,files={'image':upload()})
    assert response.status_code==200 and response.json()['faces'][0]['identity_diagnostic']['candidate_count']==0
    assert ui.gallery.repository.rows[row.identity_id]==row and ui.gallery.files.final(row.template_locator).exists()
    inspected=ui.client.post('/api/v1/demo/enrollments/'+str(row.identity_id)+'/inspect-reference',headers=ui.headers)
    assert inspected.status_code==503 and inspected.json()['status']=='ENROLLMENT_REFERENCE_UNAVAILABLE'


def test_combined_expression_accepts_landmarks_inside_demo_padding(ui):
    ui.context.bundle.detector.mode='padded_landmark'
    response=ui.client.post('/api/v1/demo/analyze-face?include_face_box=true',headers=ui.headers,
        data=CONSENT,files={'image':upload()})
    assert response.status_code==200,response.text
    assert response.json()['faces'][0]['visible_expression_prediction']['label']=='neutral'
    assert detect_single_face(noise(),ui.context.bundle).status==Status.LOW_QUALITY_IMAGE


def test_inconsistent_tolerant_references_are_atomic(ui,monkeypatch):
    calls=[]
    def feature(image):
        calls.append(1);return np.ones((1,512),np.float32)*(1 if len(calls)<3 else -1)
    monkeypatch.setattr(ui.context.bundle.recognizer,'get_feat',feature)
    response=ui.client.post('/api/v1/demo/enrollments',headers=ui.headers,data=fields(),
        files=[('enrollment_images',encoded(noise(seed,sigma=1.3))) for seed in (1,2,3)])
    assert response.status_code==422 and response.json()['status']=='INCONSISTENT_ENROLLMENT'
    assert not ui.gallery.repository.rows and not list(ui.gallery.files.staging.iterdir())
