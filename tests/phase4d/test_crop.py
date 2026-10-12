from dataclasses import replace
from types import SimpleNamespace
import io
import cv2
import numpy as np
import pytest
import torch
from PIL import Image
from faceattend import MEAN,STD
from faceattend.data import make_transform
from face_verification.schemas import FaceResult,Status
from faceattend_api.services.image_input import Part,decode
from faceattend_api.services.expression import rgb_input
from faceattend_analysis.crop import rgb_crop,CROP_VERSION
from faceattend_analysis.errors import AnalysisError
from faceattend_analysis.expression import CropExpression
from .conftest import METADATA

def face(box=(10.2,20.2,90.2,100.2)):
    return FaceResult(Status.OK,bbox=np.array(box),landmarks=np.array([[30,40],[70,40],[50,60],[35,80],[65,80]]),aligned=np.zeros((112,112,3),np.uint8))

def test_tight_original_resolution_bgr_to_rgb_once_no_aligned_crop():
    image=np.full((140,120,3),(17,53,201),np.uint8)
    crop=rgb_crop(image,face())
    assert crop.mode=='RGB' and crop.size==(81,81) and CROP_VERSION=='scrfd-bbox-rgb-m1-v1'
    np.testing.assert_array_equal(np.asarray(crop),cv2.cvtColor(image[20:101,10:91],cv2.COLOR_BGR2RGB))
    assert tuple(np.asarray(crop)[0,0])==(201,53,17)

def test_no_hidden_padding_rotation_or_face_reordering():
    image=np.random.default_rng(2).integers(0,255,(140,120,3),dtype=np.uint8)
    crop=rgb_crop(image,face((10,20,90,100)))
    np.testing.assert_array_equal(np.asarray(crop),image[20:100,10:90,::-1])

@pytest.mark.parametrize('change',[{'status':Status.NO_FACE},{'bbox':[-1,20,90,100]},
    {'bbox':[10,20,121,100]},{'bbox':[90,20,10,100]},{'bbox':[10,20,float('nan'),100]},
    {'landmarks':np.ones((4,2))},{'landmarks':np.zeros((5,2))},{'landmarks':np.full((5,2),float('inf'))}])
def test_invalid_or_truncated_geometry_refused(change):
    with pytest.raises(AnalysisError):rgb_crop(np.ones((140,120,3),np.uint8),replace(face(),**change))

@pytest.mark.parametrize('image',[np.ones((140,120),np.uint8),np.ones((140,120,4),np.uint8),np.ones((140,120,3),np.float32)])
def test_invalid_image_layout_refused(image):
    with pytest.raises(AnalysisError):rgb_crop(image,face())

def test_exact_saved_m1_resize_normalization_without_augmentation():
    image=np.full((140,120,3),(17,53,201),np.uint8);crop=rgb_crop(image,face())
    transform=make_transform(METADATA,training=False);tensor=transform(crop)
    assert tensor.shape==(3,224,224) and torch.isfinite(tensor).all()
    expected=torch.tensor([(value/255-mean)/std for value,mean,std in zip((201,53,17),MEAN,STD)])
    torch.testing.assert_close(tensor[:,100,100],expected)
    torch.testing.assert_close(transform(crop),tensor,rtol=0,atol=0)

@pytest.mark.parametrize('orientation',[1,3,6,8])
def test_jpeg_exif_uses_existing_opencv_decoder_once(context,orientation):
    pixels=np.random.default_rng(8).integers(30,200,(100,140,3),dtype=np.uint8)
    exif=Image.Exif();exif[274]=orientation
    stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='JPEG',exif=exif)
    data=stream.getvalue();image=decode(Part(name='image',is_file=True,mime='image/jpeg',data=bytearray(data)),context.settings)
    native=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    turns={1:0,3:2,6:3,8:1}[orientation]
    np.testing.assert_array_equal(image,np.rot90(native,k=turns))
    h,w=image.shape[:2];points=np.array([[w*.3,h*.3],[w*.7,h*.3],[w*.5,h*.5],[w*.35,h*.7],[w*.65,h*.7]])
    accepted=FaceResult(Status.OK,bbox=[0,0,w,h],landmarks=points)
    np.testing.assert_array_equal(np.asarray(rgb_crop(image,accepted)),image[:,:,::-1])
    # Historical Phase 3 intentionally stays Pillow full-image, no transpose.
    assert rgb_input(image,data).size==(140,100)

@pytest.mark.parametrize('change',[{'architecture':'other'},{'class_names':list(reversed(METADATA['class_names']))},
    {'input_size':[112,112]},{'normalization':{'mean':[0,0,0],'std':[1,1,1]}}])
def test_wrong_checkpoint_preprocessing_policy_refused(change):
    with pytest.raises(AnalysisError):CropExpression(SimpleNamespace(metadata=dict(METADATA,**change)))

def test_cpu_model_cannot_run_even_when_cuda_available(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    model=torch.nn.Linear(2,7)
    expression=CropExpression(SimpleNamespace(metadata=METADATA,model=model))
    from faceattend_api.errors import APIError
    with pytest.raises(APIError) as error:expression.predict(Image.new('RGB',(80,80)))
    assert error.value.code=='CUDA_UNAVAILABLE'

def test_cuda_missing_refuses_before_transform(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    expression=CropExpression(SimpleNamespace(metadata=METADATA))
    from faceattend_api.errors import APIError
    with pytest.raises(APIError):expression.predict(Image.new('RGB',(80,80)))
