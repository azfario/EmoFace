"""One user-initiated capture. No saving, monitoring, or automatic enrollment."""
import time
import cv2
from .schemas import Status, VerificationError

def capture_one_frame(*, adult_consent=False, identity_consent=False, user_initiated=False, camera_index=0, timeout_seconds=60,metrics=None):
    if adult_consent is not True or identity_consent is not True or user_initiated is not True:
        raise VerificationError(Status.CONSENT_REQUIRED,'Explicit adult consent and a user-initiated capture are required.')
    if not 1 <= timeout_seconds <= 120:
        raise ValueError('Camera sessions must have a bounded timeout of at most 120 seconds.')
    metrics=metrics if metrics is not None else {}
    metrics.update(camera_released=False,read_durations_ms=[],stop_reason=None)
    opened=time.perf_counter_ns()
    camera = cv2.VideoCapture(camera_index)
    metrics['camera_open_ms']=(time.perf_counter_ns()-opened)/1e6
    try:
        if not camera.isOpened():
            raise VerificationError(Status.INVALID_IMAGE,'Camera is unavailable. Use a consented local image instead.')
        start = time.monotonic()
        while time.monotonic()-start < timeout_seconds:
            read_start=time.perf_counter_ns()
            ok, frame = camera.read()
            metrics['read_durations_ms'].append((time.perf_counter_ns()-read_start)/1e6)
            if not ok:
                raise VerificationError(Status.INVALID_IMAGE,'Camera did not return an image.')
            black_frame = not frame.any()
            preview = frame.copy()
            label = 'No camera video (black frame) | ESC/Q: stop' if black_frame else 'SPACE: capture once | ESC/Q: stop'
            cv2.putText(preview,label,(10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.imshow('FaceAttend opt-in capture',preview)
            key = cv2.waitKey(1) & 0xff
            if key == 32:
                if black_frame:
                    metrics['stop_reason']='BLACK_FRAME_REJECTED'
                    raise VerificationError(Status.INVALID_IMAGE,'Camera returned an all-black frame. Check the virtual-camera output connection before face processing; a working Camo Studio preview alone does not confirm video delivery to this app.')
                metrics['stop_reason']='SPACE_CAPTURE'
                return frame
            if key in (27,ord('q')):
                metrics['stop_reason']='ESC_CANCEL' if key==27 else 'Q_CANCEL'
                return None
            try:
                visible=cv2.getWindowProperty('FaceAttend opt-in capture',cv2.WND_PROP_VISIBLE)
            except cv2.error:
                visible=0  # Some Windows backends throw after the user closes it.
            if visible<1:
                metrics['stop_reason']='WINDOW_CLOSED'
                return None
        metrics['stop_reason']='TIMEOUT'
        return None
    except cv2.error as exc:
        raise VerificationError(Status.INVALID_IMAGE,'Camera driver or OpenCV preview is unavailable. Check local camera permissions and GUI support.') from exc
    finally:
        camera.release()
        metrics['camera_released']=True
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Camera has been released even if GUI teardown is unavailable.
