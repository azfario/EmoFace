"""Point-in-time reference ranking; every identity remains unverified/unknown."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from collections import Counter
import numpy as np
from face_verification.detector import detect_single_face
from face_verification.embedder import generate_embedding
from face_verification.schemas import Status,VerificationError
from face_verification.template_store import TemplateStore
from faceattend_enrollment.models import PURPOSE
from faceattend_enrollment.repository import valid_id,valid_name,valid_locator
from faceattend_enrollment.errors import EnrollmentError
from faceattend_api.errors import core_error
from .catalog import Catalog
from .config import CandidateSettings
from .errors import CandidateError

WARNING=('Research diagnostic only. Possible matches are unverified; identity remains Unknown. '
         'No acceptance threshold is calibrated. Unknown subjects can receive ranked candidates. '
         'Scores are not identity probabilities or attendance approval.')

class BoundedReferenceFile:
    def __init__(self,path,limit):self.path=path;self.limit=limit
    def read_bytes(self):
        with self.path.open('rb') as stream:data=stream.read(self.limit+1)
        if len(data)>self.limit:raise OSError('Reference size bound exceeded.')
        return data

class BoundedStore(TemplateStore):
    def __init__(self,files,limit):
        self.files=files;self.limit=limit
        super().__init__(files.templates,key=files.key,retention_days=30)
    def _path(self,user_id):
        path=super()._path(user_id)
        return BoundedReferenceFile(self.files.child('templates',path.name),self.limit)

class CandidateService:
    def __init__(self,enrollment,settings=None,*,catalog=None,clock=lambda:datetime.now(timezone.utc)):
        self.enrollment=enrollment;self.settings=settings or CandidateSettings(enabled=True)
        self.catalog=catalog if catalog is not None else Catalog(enrollment.repository)
        self.clock=clock

    def eligible(self,row,model_id,now):
        try:
            valid_id(row.identity_id);valid_name(row.display_name);valid_locator(row.template_locator)
            return (row.status=='active' and row.purpose_version==PURPOSE and row.model_policy_fingerprint==model_id
                and row.consent_recorded_at.tzinfo is not None and row.enrolled_at.tzinfo is not None
                and row.expires_at.tzinfo is not None and row.consent_recorded_at<=row.enrolled_at<row.expires_at
                and row.expires_at-row.consent_recorded_at<=timedelta(days=30)
                and row.consent_recorded_at<=now<row.expires_at)
        except Exception:return False

    @staticmethod
    def output(candidates,face_detected,reason,eligible_count=0,excluded_count=0):
        scores=[item['cosine_similarity'] for item in candidates]
        return {'status':'CANDIDATE_DIAGNOSTIC_COMPLETED','face_detected':face_detected,
            'candidate_count':len(candidates),'eligible_reference_count':eligible_count,
            'excluded_reference_count':excluded_count,'possible_candidates':candidates,
            'identification_status':'UNCALIBRATED_DIAGNOSTIC','identity_status':'Unknown (unverified)',
            'verified':False,'threshold':None,'reason':reason,'warning':WARNING,
            'exact_top_score_tie':len(scores)>1 and scores[0]==scores[1],
            'top_score_gap':scores[0]-scores[1] if len(scores)>1 else None,
            'ambiguity_policy':'No calibrated score-gap cutoff; ordering and ties are diagnostics only.'}

    def search(self,image,bundle,*,top_k=3,adult_consent=False,research_opt_in=False,search_opt_in=False):
        if not self.settings.enabled:raise CandidateError('CANDIDATES_DISABLED')
        if adult_consent is not True or research_opt_in is not True or search_opt_in is not True:
            raise CandidateError('CANDIDATE_SEARCH_OPT_IN_REQUIRED')
        if type(top_k) is not int or not 1<=top_k<=self.settings.max_top_k:
            raise CandidateError('INVALID_CANDIDATE_REQUEST')
        if bundle.embedding_dim!=512:raise core_error(Status.MODEL_MISMATCH)
        face=detect_single_face(image,bundle)
        if face.status==Status.NO_FACE:return self.output([],False,'NO_FACE')
        if face.status!=Status.OK:raise core_error(face.status)
        return self._rank_face(face,bundle,top_k)

    def _rank_face(self,face,bundle,top_k):
        """Internal trusted-QC hook; HTTP opt-ins/gate remain enforced by each caller."""
        if not self.settings.enabled:raise CandidateError('CANDIDATES_DISABLED')
        if type(top_k) is not int or not 1<=top_k<=self.settings.max_top_k:raise CandidateError('INVALID_CANDIDATE_REQUEST')
        if face.status!=Status.OK:raise core_error(face.status)
        if bundle.embedding_dim!=512:raise core_error(Status.MODEL_MISMATCH)
        probe=generate_embedding(face,bundle)
        references=[];matrix=None
        try:
            with self.enrollment.files.locked():
                rows=self.catalog.snapshot()
                if len(rows)>self.settings.gallery_limit:raise CandidateError('CANDIDATE_GALLERY_TOO_LARGE')
                # Keep the newest version before eligibility filtering; never fall
                # back to an older active version of a revoked/expired identity.
                unique={}
                for row in rows:
                    try:
                        previous=unique.get(row.identity_id)
                        if previous is None or (row.enrolled_at,str(row.template_locator))>(previous.enrolled_at,str(previous.template_locator)):
                            unique[row.identity_id]=row
                    except Exception:continue
                shared=Counter(r.template_locator for r in unique.values())
                metadata=[r for r in unique.values() if shared[r.template_locator]==1 and self.eligible(r,bundle.model_id,self.clock())]
                if not metadata:return self.output([],True,'NO_ELIGIBLE_REFERENCES',excluded_count=len(rows))
                if self.enrollment.files.key is None:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE')
                try:store=BoundedStore(self.enrollment.files,self.settings.max_template_bytes)
                except Exception:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE') from None
                loaded=[]
                for row in metadata:
                    try:
                        payload=store.load(self.enrollment.files.user(row.template_locator))
                        vector=payload['embedding']
                        if payload['model_id']!=bundle.model_id or vector.shape!=(512,) or not np.all(np.isfinite(vector)):
                            continue
                        loaded.append(row);references.append(vector)
                    except VerificationError:continue
                    except (OSError,ValueError,TypeError):continue
                if not loaded:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE')
                current={r.identity_id:r for r in self.catalog.revalidate(loaded)}
                final=[(r,v) for r,v in zip(loaded,references) if current.get(r.identity_id)==r and self.eligible(r,bundle.model_id,self.clock())]
                if not final:return self.output([],True,'NO_ELIGIBLE_REFERENCES',excluded_count=len(rows))
                matrix=np.stack([v for r,v in final])
                scores=np.clip(matrix@probe,-1.0,1.0)
                ordered=sorted(zip(final,scores),key=lambda value:(-float(value[1]),str(value[0][0].identity_id)))
                candidates=[{'identity_id':str(row.identity_id),'display_name':row.display_name,
                    'label':'Possible match (unverified)','display_label':'Possible match (unverified): '+row.display_name,
                    'cosine_similarity':float(score),'verified':False} for ((row,vector),score) in ordered[:top_k]]
                return self.output(candidates,True,'THRESHOLD_NOT_CALIBRATED',len(final),len(rows)-len(final))
        except (CandidateError,EnrollmentError):raise
        except Exception:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE') from None
        finally:
            probe.fill(0)
            if matrix is not None:matrix.fill(0)
            for vector in references:vector.fill(0)
