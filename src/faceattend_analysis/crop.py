"""Versioned tight SCRFD ROI; no ArcFace crop, padding or hidden EXIF transform."""
import cv2
import numpy as np
from PIL import Image
from face_verification.schemas import Status
from .errors import AnalysisError

CROP_VERSION='scrfd-bbox-rgb-m1-v1'

def rgb_crop(image,face):
    try:
        if face.status!=Status.OK or not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError
        box=np.asarray(face.bbox);points=np.asarray(face.landmarks)
        if box.shape!=(4,) or points.shape!=(5,2) or not np.all(np.isfinite(box)) or not np.all(np.isfinite(points)):raise ValueError
        height,width=image.shape[:2];x1,y1,x2,y2=box
        if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError
        if np.any(points[:,0]<x1) or np.any(points[:,0]>x2) or np.any(points[:,1]<y1) or np.any(points[:,1]>y2):raise ValueError
        left,top=int(np.floor(x1)),int(np.floor(y1));right,bottom=int(np.ceil(x2)),int(np.ceil(y2))
        roi=image[top:bottom,left:right]
        if roi.size==0 or min(roi.shape[:2])<2:raise ValueError
        return Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
    except Exception:raise AnalysisError('INVALID_EXPRESSION_CROP') from None
