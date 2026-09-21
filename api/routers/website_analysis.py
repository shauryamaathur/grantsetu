from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db.models import NGOProfile, WebsiteAnalysis
from api.db.session import get_db
from api.models.schemas import (
    NGOProfileResponse,
    WebsiteAnalysisApply,
    WebsiteAnalysisResponse,
)
from src.ai.factory import get_provider_for_ngo
from src.discovery.website_analyzer import analyze_ngo_website

router = APIRouter(prefix="/ngo-profile/{ngo_id}", tags=["website-analysis"])


@router.post("/analyze-website", response_model=WebsiteAnalysisResponse, status_code=201)
def analyze_website(ngo_id: str, db: Session = Depends(get_db)):
    """Fetches the NGO's website (and a few relevant sub-pages) and produces
    a PROPOSED profile update -- see WebsiteAnalysis's docstring: this is
    never auto-applied. Uses the NGO's configured AI provider if enabled,
    otherwise a rule-based keyword extraction -- both paths always run
    (never fails outright just because no AI is configured)."""
    ngo = db.get(NGOProfile, ngo_id)
    if ngo is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")
    if not ngo.website_url:
        raise HTTPException(status_code=400, detail="This NGO profile has no website_url set yet")

    provider = get_provider_for_ngo(db, ngo_id)
    result, pages_fetched, fetch_error = analyze_ngo_website(ngo.website_url, provider)

    analysis = WebsiteAnalysis(
        ngo_id=ngo_id,
        website_url=ngo.website_url,
        summary=result.summary or None,
        detected_focus_areas=result.focus_areas,
        detected_locations=result.locations,
        detected_beneficiaries=result.beneficiaries,
        keywords=result.keywords,
        pages_analyzed=pages_fetched,
        method=result.method,
        status="pending_review" if fetch_error is None else "dismissed",
        error=fetch_error,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    if fetch_error:
        raise HTTPException(status_code=502, detail=fetch_error)

    return analysis


@router.get("/website-analyses", response_model=list[WebsiteAnalysisResponse])
def list_website_analyses(ngo_id: str, db: Session = Depends(get_db)):
    if db.get(NGOProfile, ngo_id) is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")
    return (
        db.query(WebsiteAnalysis)
        .filter_by(ngo_id=ngo_id)
        .order_by(WebsiteAnalysis.created_at.desc())
        .all()
    )


@router.post("/website-analyses/{analysis_id}/apply", response_model=NGOProfileResponse)
def apply_website_analysis(
    ngo_id: str, analysis_id: str, payload: WebsiteAnalysisApply, db: Session = Depends(get_db)
):
    """Applies selected fields from a REVIEWED analysis onto the NGO
    profile. The user chooses which categories to apply (focus_areas /
    locations / beneficiaries independently) -- nothing is overwritten
    silently, and fields are MERGED (union with existing values, not
    replaced), so a user's own manual entries are never discarded."""
    ngo = db.get(NGOProfile, ngo_id)
    if ngo is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")

    analysis = db.get(WebsiteAnalysis, analysis_id)
    if analysis is None or analysis.ngo_id != ngo_id:
        raise HTTPException(status_code=404, detail=f"Website analysis '{analysis_id}' not found for this NGO")

    def _merge(existing: list[str], new: list[str]) -> list[str]:
        existing_lower = {e.lower() for e in existing}
        merged = list(existing)
        for item in new:
            if item.lower() not in existing_lower:
                merged.append(item)
                existing_lower.add(item.lower())
        return merged

    if payload.apply_focus_areas:
        ngo.focus_areas = _merge(ngo.focus_areas, analysis.detected_focus_areas)
    if payload.apply_locations:
        ngo.locations = _merge(ngo.locations, analysis.detected_locations)
    if payload.apply_beneficiaries:
        ngo.beneficiaries = _merge(ngo.beneficiaries, analysis.detected_beneficiaries)

    analysis.status = "applied"
    db.commit()
    db.refresh(ngo)
    return ngo
