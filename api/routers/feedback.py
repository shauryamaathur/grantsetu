from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.db.models import NGOProfile, OpportunityFeedback
from api.db.session import get_db
from api.models.schemas import OpportunityFeedbackCreate, OpportunityFeedbackResponse

router = APIRouter(prefix="/opportunity-feedback", tags=["opportunity-feedback"])


@router.post("", response_model=OpportunityFeedbackResponse, status_code=201)
def submit_feedback(payload: OpportunityFeedbackCreate, db: Session = Depends(get_db)):
    if db.get(NGOProfile, payload.ngo_id) is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")

    existing = (
        db.query(OpportunityFeedback)
        .filter_by(
            ngo_id=payload.ngo_id,
            opportunity_source=payload.opportunity_source,
            opportunity_ref=payload.opportunity_ref,
            feedback_type=payload.feedback_type,
        )
        .first()
    )
    if existing:
        return existing  # idempotent -- hiding/reporting the same thing twice is a no-op, not an error

    feedback = OpportunityFeedback(
        ngo_id=payload.ngo_id,
        opportunity_source=payload.opportunity_source,
        opportunity_ref=payload.opportunity_ref,
        feedback_type=payload.feedback_type,
        notes=payload.notes,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


@router.get("", response_model=list[OpportunityFeedbackResponse])
def list_feedback(
    ngo_id: str = Query(...),
    feedback_type: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(OpportunityFeedback).filter_by(ngo_id=ngo_id)
    if feedback_type:
        query = query.filter_by(feedback_type=feedback_type)
    return query.all()


@router.delete("/{feedback_id}", status_code=204)
def remove_feedback(feedback_id: str, db: Session = Depends(get_db)):
    feedback = db.get(OpportunityFeedback, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=404, detail=f"Feedback '{feedback_id}' not found")
    db.delete(feedback)
    db.commit()
