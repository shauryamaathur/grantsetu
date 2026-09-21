import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.db.models import ImportedGrant, NGOProfile, SavedOpportunity
from api.db.session import get_db
from api.models.schemas import (
    SavedOpportunityCreate,
    SavedOpportunityResponse,
    SavedOpportunityUpdate,
)
from src.ranking.corpus import get_corpus

router = APIRouter(prefix="/saved-opportunities", tags=["saved-opportunities"])


def _enrich(saved: SavedOpportunity, db: Session) -> SavedOpportunityResponse:
    """Attaches title/funder/deadline/expiry/url at read time so the
    frontend can render a saved-grants list without a second round trip per
    item -- looked up from whichever corpus the save actually points at."""
    base = SavedOpportunityResponse.model_validate(saved)
    if saved.opportunity_source == "imported":
        grant = db.get(ImportedGrant, saved.opportunity_ref)
        if grant is not None:
            base.title = grant.title
            base.funder = grant.funder
            base.deadline = grant.deadline.date() if grant.deadline else None
            base.is_expired = grant.status in ("closed", "expired")
            base.url = grant.url
    else:
        corpus = get_corpus()
        row = corpus.get_opportunity(saved.opportunity_id)
        if row is not None:
            base.title = row["opportunity_title"]
            funder = row.get("agency_name")
            base.funder = funder if pd.notna(funder) else None
            base.deadline = row["close_date"].date() if pd.notna(row.get("close_date")) else None
            base.is_expired = bool(row.get("is_expired", False))
            # additional_information_url can be pandas NA (not Python None) in
            # this corpus -- assigning that straight to a str|None field would
            # coerce to the literal string "<NA>" instead of a real null.
            url = row.get("additional_information_url")
            base.url = url if pd.notna(url) else None
    return base


@router.post("", response_model=SavedOpportunityResponse, status_code=201)
def save_opportunity(payload: SavedOpportunityCreate, db: Session = Depends(get_db)):
    profile = db.get(NGOProfile, payload.ngo_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")

    if payload.opportunity_source == "imported":
        ref = payload.opportunity_ref
        if not ref:
            raise HTTPException(status_code=422, detail="opportunity_ref is required when opportunity_source='imported'")
        grant = db.get(ImportedGrant, ref)
        if grant is None:
            raise HTTPException(status_code=404, detail=f"Imported grant '{ref}' not found")
        db_opportunity_id = SavedOpportunity.IMPORTED_ID_SENTINEL
    else:
        opp_id = payload.opportunity_id
        ref = payload.opportunity_ref or (str(opp_id) if opp_id is not None else None)
        if opp_id is None or not ref:
            raise HTTPException(status_code=422, detail="opportunity_id is required when opportunity_source='historical'")
        corpus = get_corpus()
        if corpus.get_opportunity(opp_id) is None:
            raise HTTPException(status_code=404, detail=f"Opportunity '{opp_id}' not found")
        db_opportunity_id = opp_id

    existing = (
        db.query(SavedOpportunity)
        .filter_by(ngo_id=payload.ngo_id, opportunity_source=payload.opportunity_source, opportunity_ref=ref)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="This opportunity is already saved for this NGO")

    saved = SavedOpportunity(
        ngo_id=payload.ngo_id,
        opportunity_id=db_opportunity_id,
        opportunity_source=payload.opportunity_source,
        opportunity_ref=ref,
        notes=payload.notes,
        assigned_to=payload.assigned_to,
    )
    db.add(saved)
    db.commit()
    db.refresh(saved)
    return _enrich(saved, db)


@router.get("", response_model=list[SavedOpportunityResponse])
def list_saved_opportunities(ngo_id: str = Query(...), db: Session = Depends(get_db)):
    rows = db.query(SavedOpportunity).filter_by(ngo_id=ngo_id).order_by(SavedOpportunity.saved_at.desc()).all()
    return [_enrich(row, db) for row in rows]


@router.patch("/{saved_id}", response_model=SavedOpportunityResponse)
def update_saved_opportunity(saved_id: str, payload: SavedOpportunityUpdate, db: Session = Depends(get_db)):
    saved = db.get(SavedOpportunity, saved_id)
    if saved is None:
        raise HTTPException(status_code=404, detail=f"Saved opportunity '{saved_id}' not found")

    if payload.status is not None:
        saved.status = payload.status
    if payload.assigned_to is not None:
        saved.assigned_to = payload.assigned_to
    if payload.notes is not None:
        saved.notes = payload.notes

    db.commit()
    db.refresh(saved)
    return _enrich(saved, db)


@router.delete("/{saved_id}", status_code=204)
def remove_saved_opportunity(saved_id: str, db: Session = Depends(get_db)):
    saved = db.get(SavedOpportunity, saved_id)
    if saved is None:
        raise HTTPException(status_code=404, detail=f"Saved opportunity '{saved_id}' not found")
    db.delete(saved)
    db.commit()
