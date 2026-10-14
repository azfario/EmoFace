from face_verification.schemas import Status
from faceattend_api.errors import core_error
from faceattend_candidates.service import CandidateService
from .crop import rgb_crop,CROP_VERSION
from .expression import CropExpression,WARNING
from .errors import AnalysisError
from .quality import expression_detection,BLUR_WARNING

class FaceAnalysis:
    def __init__(self,expression,*,candidate_service=None):
        self.expression=expression;self.candidate_service=candidate_service

    def analyze(self,image,bundle,*,adult_consent=False,research_opt_in=False,expression_opt_in=False,top_k=3,include_face_box=False,allow_blurry_expression=False):
        if any(flag is not True for flag in (adult_consent,research_opt_in,expression_opt_in)):
            raise AnalysisError('EXPRESSION_OPT_IN_REQUIRED')
        # One accepted detection is shared; no largest-face selection or second detection.
        face,blurry=expression_detection(image,bundle,allow_blurry_expression)
        if face.status==Status.NO_FACE:return self.output([])
        if face.status!=Status.OK:raise core_error(face.status)
        rgb=rgb_crop(image,face,getattr(bundle,'quality_policy',None))
        try:expression=self.expression.predict(rgb)
        finally:rgb.close()
        identity=CandidateService.output([],True,'CANDIDATE_SEARCH_NOT_REQUESTED')
        if blurry:
            identity=CandidateService.output([],True,'BLURRY_FRAME_IDENTITY_MATCHING_SKIPPED')
        elif self.candidate_service is not None:
            identity=self.candidate_service._rank_face(face,bundle,top_k)
        item={'face_index':0,'identity_diagnostic':identity,'visible_expression_prediction':expression}
        if blurry:item['input_quality']={'status':'BLUR_GATE_BYPASSED','warning':BLUR_WARNING}
        if include_face_box:
            from faceattend_ui.integration import face_box
            item['face_box']=face_box(face,image)
        return self.output([item])

    @staticmethod
    def output(faces):
        return {'status':'PER_FACE_RESEARCH_DIAGNOSTIC_COMPLETED','face_detected':bool(faces),'face_count':len(faces),
            'faces':faces,'verified':False,'threshold':None,'not_a_mood_or_diagnosis':True,
            'crop_version':CROP_VERSION,'warning':WARNING,'persistent_inference_history':False}
