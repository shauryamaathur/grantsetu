"""The floating GrantSetu AI assistant's single endpoint. See
src/ai/assistant.py for the grounding/answering discipline -- this router's
only job is resolving WHICH real data to ground the answer in, based on
`context.type`, then delegating to that module.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db.models import GrantSource, ImportedGrant, NGOProfile, OpportunityChange
from api.db.session import get_db
from api.models.schemas import AssistantRequest, AssistantResponse
from src.ai.assistant import (
    answer_question,
    build_analytics_context,
    build_ngo_context,
    build_opportunity_context_from_historical,
    build_opportunity_context_from_imported,
    build_search_context,
)
from src.ai.factory import get_provider_for_ngo
from src.ranking.corpus import get_corpus
from src.ranking.live_match import RECOMMENDABLE_STATUSES, rank_live_grants

router = APIRouter(prefix="/assistant", tags=["assistant"])


@router.post("/ask", response_model=AssistantResponse)
def ask_assistant(payload: AssistantRequest, db: Session = Depends(get_db)):
    provider = None
    if payload.ngo_id is not None:
        if db.get(NGOProfile, payload.ngo_id) is None:
            raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")
        provider = get_provider_for_ngo(db, payload.ngo_id)

    facts: dict = {}
    grounded_on: list[str] = []
    ctx = payload.context

    if ctx is None or ctx.type == "general":
        facts = {
            "product": "GrantSetu",
            "capabilities": [
                "Live Feed: continuously discovered funding opportunities from Grants.gov, NGOBox, and other configured sources",
                "AI Funding Search: natural-language search across live and historical opportunities",
                "Grant Intelligence: interactive analysis of the historical opportunity corpus and live opportunities",
                "NGO Profile: personalizes recommendations by focus area, beneficiaries, and location",
                "Saved Opportunities: track opportunities through saved -> planning_to_apply -> applied -> awarded/rejected",
            ],
        }
        grounded_on = ["GrantSetu product capabilities"]

    elif ctx.type == "opportunity":
        if ctx.opportunity_source == "imported":
            grant = db.get(ImportedGrant, ctx.opportunity_ref) if ctx.opportunity_ref else None
            if grant is None or grant.is_sample_data:
                raise HTTPException(status_code=404, detail="Opportunity not found")
            source = db.get(GrantSource, grant.source_id)
            changes = (
                db.query(OpportunityChange)
                .filter_by(imported_grant_id=grant.imported_grant_id)
                .order_by(OpportunityChange.detected_at.desc())
                .limit(5)
                .all()
            )
            facts = build_opportunity_context_from_imported(grant, source, changes)
            grounded_on = [f"Live opportunity: {grant.title}"]
        elif ctx.opportunity_source == "historical":
            try:
                opp_id = int(ctx.opportunity_ref)
            except (TypeError, ValueError):
                raise HTTPException(status_code=422, detail="opportunity_ref must be an integer for historical opportunities")
            row = get_corpus().get_opportunity(opp_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Opportunity not found")
            facts = build_opportunity_context_from_historical(row)
            grounded_on = [f"Historical opportunity #{opp_id}"]
        else:
            raise HTTPException(status_code=422, detail="context.opportunity_source must be 'imported' or 'historical'")

    elif ctx.type == "ngo":
        if payload.ngo_id is None:
            raise HTTPException(status_code=422, detail="ngo_id is required for context.type='ngo'")
        profile = db.get(NGOProfile, payload.ngo_id)
        candidates = (
            db.query(ImportedGrant)
            .filter_by(is_sample_data=False)
            .filter(ImportedGrant.status.in_(RECOMMENDABLE_STATUSES))
            .all()
        )
        matches = rank_live_grants(
            candidates, profile.focus_areas, profile.beneficiaries,
            locations=profile.locations, registration_type=profile.registration_type, limit=5,
        )
        facts = build_ngo_context(profile, [m.grant.title for m in matches])
        grounded_on = [f"NGO profile: {profile.name}", f"{len(matches)} currently recommended opportunities"]

    elif ctx.type == "search":
        facts = build_search_context(ctx.search_query or "", ctx.search_results or [])
        grounded_on = [f"Your last search: \"{ctx.search_query}\"" if ctx.search_query else "Your last search results"]

    elif ctx.type == "analytics":
        facts = build_analytics_context(ctx.analytics or {})
        grounded_on = ["Grant Intelligence analytics you're currently viewing"]

    else:
        raise HTTPException(status_code=422, detail=f"Unknown context.type '{ctx.type}'")

    result = answer_question(payload.message, facts, grounded_on, provider)

    return AssistantResponse(
        answer=result.answer,
        method=result.method,
        ai_disclaimer=result.ai_disclaimer,
        grounded_on=result.grounded_on,
        data_available=result.data_available,
    )
