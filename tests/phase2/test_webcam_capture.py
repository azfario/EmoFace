from types import SimpleNamespace
import numpy as np
import pytest
from face_verification import Status,VerificationError
from face_verification import webcam

@pytest.fixture
def camera(monkeypatch):
    data={'released':False,'reads':0,'windows_closed':False}
    frame=np.full((200,200,3),128,np.uint8)
    def read():
        data['reads']+=1
        return True,frame
    device=SimpleNamespace(isOpened=lambda:True,read=read,release=lambda:data.update(released=True))
    monkeypatch.setattr(webcam.cv2,'VideoCapture',lambda *a:device)
    monkeypatch.setattr(webcam.cv2,'imshow',lambda *a:None)
    monkeypatch.setattr(webcam.cv2,'putText',lambda *a:None)
    monkeypatch.setattr(webcam.cv2,'getWindowProperty',lambda *a:1)
    monkeypatch.setattr(webcam.cv2,'destroyAllWindows',lambda:data.update(windows_closed=True))
    return device,data,frame

def capture(**kw):
    return webcam.capture_one_frame(adult_consent=True,identity_consent=True,user_initiated=True,**kw)

def test_no_camera_without_operator_action(monkeypatch):
    monkeypatch.setattr(webcam.cv2,'VideoCapture',lambda *a:pytest.fail('camera must remain closed'))
    with pytest.raises(VerificationError) as exc:
        webcam.capture_one_frame()
    assert exc.value.status==Status.CONSENT_REQUIRED

def test_space_capture_releases(camera,monkeypatch):
    _,data,frame=camera
    monkeypatch.setattr(webcam.cv2,'waitKey',lambda *a:32)
    metrics={}
    assert capture(metrics=metrics) is frame
    assert data['released'] and data['windows_closed'] and metrics['camera_released']
    assert metrics['stop_reason']=='SPACE_CAPTURE' and len(metrics['read_durations_ms'])==1

def test_black_virtual_camera_frame_rejected_as_camera_error(camera,monkeypatch):
    device,data,frame=camera
    device.read=lambda:(True,np.zeros_like(frame))
    monkeypatch.setattr(webcam.cv2,'waitKey',lambda *a:32)
    metrics={}
    with pytest.raises(VerificationError) as exc:
        capture(metrics=metrics)
    assert exc.value.status==Status.INVALID_IMAGE
    assert 'all-black frame' in str(exc.value)
    assert metrics['stop_reason']=='BLACK_FRAME_REJECTED'
    assert data['released'] and metrics['camera_released'] and data['windows_closed']

@pytest.mark.parametrize('key,reason',[(27,'ESC_CANCEL'),(ord('q'),'Q_CANCEL')])
def test_key_cancel_releases(camera,monkeypatch,key,reason):
    _,data,_=camera
    monkeypatch.setattr(webcam.cv2,'waitKey',lambda *a:key)
    metrics={}
    assert capture(metrics=metrics) is None
    assert data['released'] and metrics['stop_reason']==reason

@pytest.mark.parametrize('raises',[False,True])
def test_window_close_releases(camera,monkeypatch,raises):
    _,data,_=camera
    monkeypatch.setattr(webcam.cv2,'waitKey',lambda *a:-1)
    def visibility(*a):
        if raises:
            raise webcam.cv2.error('window no longer exists')
        return 0
    monkeypatch.setattr(webcam.cv2,'getWindowProperty',visibility)
    metrics={}
    assert capture(metrics=metrics) is None
    assert data['released'] and metrics['stop_reason']=='WINDOW_CLOSED'

def test_timeout_releases(camera,monkeypatch):
    _,data,_=camera
    clock=iter([0.,2.])
    monkeypatch.setattr(webcam.time,'monotonic',lambda:next(clock))
    metrics={}
    assert capture(timeout_seconds=1,metrics=metrics) is None
    assert data['released'] and data['reads']==0 and metrics['stop_reason']=='TIMEOUT'

@pytest.mark.parametrize('problem',['unopened','read_failed','preview_failed'])
def test_failures_release(camera,monkeypatch,problem):
    device,data,_=camera
    if problem=='unopened':
        device.isOpened=lambda:False
    if problem=='read_failed':
        device.read=lambda:(False,None)
    if problem=='preview_failed':
        monkeypatch.setattr(webcam.cv2,'imshow',lambda *a:(_ for _ in ()).throw(webcam.cv2.error('GUI unavailable')))
    with pytest.raises(VerificationError) as exc:
        capture()
    assert exc.value.status==Status.INVALID_IMAGE and data['released']
