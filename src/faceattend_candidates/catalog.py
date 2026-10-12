"""Bounded read-only metadata; existing session/ORM, no migration or new grants."""
from sqlalchemy import select
from faceattend_enrollment.models import Identity
from faceattend_enrollment.repository import record
from .errors import CandidateError

class Catalog:
    def __init__(self,repository):self.factory=repository.factory
    def snapshot(self):
        try:
            with self.factory() as session:
                rows=session.execute(select(Identity).order_by(Identity.identity_id).limit(101)).scalars()
                values=[record(row) for row in rows]
            if len(values)>100:raise CandidateError('CANDIDATE_GALLERY_TOO_LARGE')
            return values
        except CandidateError:raise
        except Exception:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE') from None
    def revalidate(self,entries):
        if not entries:return []
        if len(entries)>100:raise CandidateError('CANDIDATE_GALLERY_TOO_LARGE')
        try:
            with self.factory() as session:
                rows=session.execute(select(Identity).where(Identity.identity_id.in_([r.identity_id for r in entries])).limit(101)).scalars()
                current={row.identity_id:record(row) for row in rows}
            return [r for r in entries if current.get(r.identity_id)==r]
        except Exception:raise CandidateError('CANDIDATE_GALLERY_UNAVAILABLE') from None
