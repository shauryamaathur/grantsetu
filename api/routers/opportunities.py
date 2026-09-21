import math
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.db.models import GrantSource, NGOProfile, OpportunityFeedback, SearchHistory
from api.db.session import get_db
from api.models.schemas import (
    AISearchIntentEcho,
    AISearchRequest,
    AISearchResponse,
    AISearchResultItem,
    MatchScore,
    MatchScoreComponent,
    OpportunityDetail,
    OpportunitySummary,
    RankedImportedGrant,
    SearchResultsSummary,
)
from api.routers.grant_sources import imported_grant_to_summary
from src.ai.factory import get_provider_for_ngo
from src.ai.grounded_explain import explain_result
from src.ai.query_understanding import SCOPE_HISTORICAL, extract_search_intent
from src.ranking.ai_search import search as ai_search_run
from src.ranking.corpus import get_corpus
from src.ranking.live_match import build_canonical_match_scores

router = APIRouter(prefix="/opportunities", tags=["opportunities"])

# Every opportunity in this endpoint family comes from the historical
# grants.gov-style corpus (see docs/data_audit_report.md) -- that fact is
# stated once, honestly, here and in README/docs, rather than as a
# recurring "this is fake" banner on every screen. Per-opportunity status
# is still communicated accurately via status_label/source_label below.
HISTORICAL_SOURCE_LABEL = "Historical opportunity"

# Maps src/ai/grounded_explain.py's coarser 4-tier vocabulary (used when a
# result has no canonical src/ranking/live_match.py score to draw a tier id
# from -- i.e. no ngo_id was given) onto the same tier-id namespace
# live_match.py uses, so the frontend can style every AI-search result
# with one shared CSS mapping regardless of which system produced its tier.
_LEGACY_TIER_ID_MAP = {
    "Strong match": "strong",
    "Possible match": "moderate",
    "Weak match": "low",
    "Eligibility unclear": "minimal",
}


def _clean_float(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return float(v)


def _status_label(is_expired: bool, close_date) -> str:
    if is_expired:
        return "Currently closed"
    if close_date is None:
        return "Deadline unavailable"
    return "Open"


def _eligibility_status_label(eligible_applicants) -> str:
    if not eligible_applicants or (isinstance(eligible_applicants, float) and math.isnan(eligible_applicants)):
        return "Eligibility requires verification"
    return "Eligibility stated by source"


def _row_to_summary(row: pd.Series) -> OpportunitySummary:
    close_date = row["close_date"].date() if pd.notna(row.get("close_date")) else None
    is_expired = bool(row.get("is_expired", False))
    return OpportunitySummary(
        opportunity_id=int(row["opportunity_id"]),
        opportunity_title=row["opportunity_title"],
        agency_name=row.get("agency_name"),
        opportunity_category=row.get("opportunity_category"),
        close_date=close_date,
        is_expired=is_expired,
        award_size_bucket=row.get("award_size_bucket"),
        status_label=_status_label(is_expired, close_date),
        source_label=HISTORICAL_SOURCE_LABEL,
    )


def _row_to_detail(row: pd.Series) -> OpportunityDetail:
    summary = _row_to_summary(row)
    return OpportunityDetail(
        **summary.model_dump(),
        opportunity_number=row.get("opportunity_number"),
        funding_instrument_type=row.get("funding_instrument_type"),
        category_of_funding_activity=row.get("category_of_funding_activity"),
        eligible_applicants=row.get("eligible_applicants"),
        eligible_applicants_type=row.get("eligible_applicants_type"),
        eligibility_status_label=_eligibility_status_label(row.get("eligible_applicants")),
        post_date=row["post_date"].date() if pd.notna(row.get("post_date")) else None,
        award_ceiling=_clean_float(row.get("award_ceiling")),
        award_floor=_clean_float(row.get("award_floor")),
        estimated_total_program_funding=_clean_float(row.get("estimated_total_program_funding")),
        additional_information_url=row.get("additional_information_url"),
    )


@router.get("/search", response_model=list[OpportunitySummary])
def search_opportunities(
    q: str = Query(default="", description="Keyword search query"),
    category: str | None = Query(default=None),
    funder: str | None = Query(default=None, description="Filter by funder/agency name substring"),
    limit: int = Query(default=25, ge=1, le=200),
):
    corpus = get_corpus()
    results = corpus.search(q, category=category, funder=funder, top_k=limit)
    return [_row_to_summary(row) for _, row in results.iterrows()]


@router.get("/recommended", response_model=list[RankedImportedGrant])
def recommended_opportunities(
    ngo_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """LIVE opportunities only, ranked by relevance to the NGO's profile.

    This is deliberately NOT the historical corpus and NEVER includes
    sample/fixture data or closed/expired grants (see RECOMMENDABLE_STATUSES
    in src/ranking/live_match.py) -- Recommended means "real, currently-
    applicable opportunities relevant to you," not "anything in the
    database." When no live source has been synced yet, or nothing synced
    shares any vocabulary with this NGO's profile, the honest result is an
    empty list -- never a historical-data or sample-data substitute. Use
    GET /opportunities/live to browse the full, unranked live-discovery
    feed, or GET /opportunities/search for the separate historical archive.
    """
    profile = db.get(NGOProfile, ngo_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")

    # Canonical scores (src/ranking/live_match.py::build_canonical_match_scores)
    # -- the SAME computation GET /opportunities/live/{id} and POST
    # /opportunities/ai-search read from, so a grant's score here is never
    # allowed to be A Different Number elsewhere.
    canonical = build_canonical_match_scores(
        db, profile.focus_areas, profile.beneficiaries,
        locations=profile.locations, registration_type=profile.registration_type,
    )

    hidden_refs = {
        ref
        for (ref,) in db.query(OpportunityFeedback.opportunity_ref)
        .filter_by(ngo_id=ngo_id, opportunity_source="imported", feedback_type="hidden")
        .all()
    }
    # Hidden filtering happens HERE, on the output list, never on the
    # scoring corpus above -- a grant's canonical score must not depend on
    # some other, unrelated grant having been hidden by this NGO.
    matches = [m for m in canonical.values() if m.grant.imported_grant_id not in hidden_refs]
    matches.sort(key=lambda m: m.match_score.score_100, reverse=True)
    matches = matches[:limit]

    source_ids = {m.grant.source_id for m in matches}
    sources_by_id = {
        s.source_id: s for s in db.query(GrantSource).filter(GrantSource.source_id.in_(source_ids)).all()
    } if source_ids else {}

    return [
        RankedImportedGrant(
            opportunity=imported_grant_to_summary(m.grant, sources_by_id.get(m.grant.source_id)),
            match_summary=f"{m.match_score.score_100}/100 -- {m.match_score.tier_label}",
            match_reasons=m.match_reasons,
            score=m.score,
            match_score=MatchScore(
                score_100=m.match_score.score_100,
                tier=m.match_score.tier,
                tier_label=m.match_score.tier_label,
                components=[
                    MatchScoreComponent(label=c.label, points=c.points, detail=c.detail)
                    for c in m.match_score.components
                ],
            ),
        )
        for m in matches
    ]


@router.post("/ai-search", response_model=AISearchResponse)
def ai_search(payload: AISearchRequest, db: Session = Depends(get_db)):
    """Natural-language grant search: extracts structured intent from the
    query (AI-assisted if the given/inferred NGO has a provider configured,
    otherwise a deterministic keyword/regex extractor -- see
    src/ai/query_understanding.py), retrieves and ranks CURRENT-only
    candidates from live ImportedGrant data (+ the open web, once
    configured -- src/ranking/ai_search.py -- never sample data, never
    closed/expired live grants), and generates a grounded, tier-labeled
    explanation for each shortlisted result (src/ai/grounded_explain.py) --
    the match tier is ALWAYS computed deterministically from real signals,
    never an AI judgment call. Works fully without any AI configured (a
    documented product requirement throughout this codebase);
    `ai_available` in the response says whether AI was actually used
    anywhere for this call.

    HARD RULE: Ask GrantSetu is current-funding discovery ONLY. A query
    classified as SCOPE_HISTORICAL (src/ai/query_understanding.py) -- even
    one that explicitly asks for past/archived funding -- is intercepted
    HERE, before src/ranking/ai_search.py::search() is ever called, and
    answered with a redirect notice pointing at Grant Intelligence instead
    of a historical-database search. This is not a post-hoc filter: the
    historical corpus is never queried, never scored, and never present in
    `results` for this endpoint, regardless of scope or how few current
    results exist -- see ai_search.py's module docstring for the full
    rationale."""
    provider = None
    profile = None
    if payload.ngo_id is not None:
        profile = db.get(NGOProfile, payload.ngo_id)
        if profile is None:
            raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")
        provider = get_provider_for_ngo(db, payload.ngo_id)

    intent = extract_search_intent(payload.query, provider)

    db.add(SearchHistory(
        ngo_id=payload.ngo_id,
        query_text=payload.query,
        filters={
            "method": intent.method,
            "search_scope": intent.search_scope,
            "locations": intent.locations,
            "causes": intent.causes,
            "applicant_type": intent.applicant_type,
            "deadline_within_days": intent.deadline_within_days,
            "international_ok": intent.international_ok,
        },
    ))
    db.commit()

    intent_echo = AISearchIntentEcho(
        method=intent.method,
        keywords=intent.keywords,
        locations=intent.locations,
        causes=intent.causes,
        applicant_type=intent.applicant_type,
        deadline_within_days=intent.deadline_within_days,
        international_ok=intent.international_ok,
        funding_min=intent.funding_min,
        funding_max=intent.funding_max,
        search_scope=intent.search_scope,
    )

    if intent.search_scope == SCOPE_HISTORICAL:
        # Never reaches ai_search_run -- see docstring's HARD RULE.
        return AISearchResponse(
            intent=intent_echo,
            results=[],
            ai_available=intent.method == "ai",
            summary=SearchResultsSummary(
                total=0, strong_matches=0, closing_within_30_days=0, open_to_india=0,
                current_count=0, upcoming_count=0, needs_verification_count=0,
            ),
            scope_notice=(
                "Ask GrantSetu focuses on current funding opportunities, so it doesn't search "
                "historical records. For past awards, funder trends, and historical analysis, "
                "use Grant Intelligence."
            ),
        )

    results = ai_search_run(db, intent, limit=payload.limit)

    # THE single source of truth for a live opportunity's score against
    # this NGO -- the SAME function GET /opportunities/recommended and GET
    # /opportunities/live/{id} use (src/ranking/live_match.py). Only
    # meaningful when we have a real profile to score against; without one,
    # results keep ai_search.py's own TF-IDF-based relevance score/tier.
    canonical = (
        build_canonical_match_scores(
            db, profile.focus_areas, profile.beneficiaries,
            locations=profile.locations, registration_type=profile.registration_type,
        )
        if profile is not None else {}
    )

    ai_used = intent.method == "ai"
    items = []
    now = datetime.utcnow()
    for r in results:
        canonical_match = canonical.get(r.ref) if r.source == "imported" else None
        tier_override = canonical_match.match_score.tier_label if canonical_match else None

        explanation = explain_result(payload.query, r, provider, tier_override=tier_override)
        ai_used = ai_used or explanation.method == "ai"

        if canonical_match:
            score_100 = canonical_match.match_score.score_100
            tier_id = canonical_match.match_score.tier
        else:
            # Only when relevance_known -- see AISearchResultItem's
            # docstring: a filter-only match's internal 1.0 sentinel score
            # must never be displayed as if it were a real 100/100.
            score_100 = round(r.score * 100) if r.relevance_known else None
            tier_id = _LEGACY_TIER_ID_MAP.get(explanation.tier, "moderate")

        items.append(AISearchResultItem(
            source=r.source,
            ref=r.ref,
            title=r.title,
            funder=r.funder,
            category=r.category,
            country=r.country,
            deadline=r.deadline,
            url=r.url,
            status_label=r.status_label,
            source_label=r.source_label,
            is_expired=r.is_expired,
            match_tier=explanation.tier,
            match_tier_id=tier_id,
            match_score_100=score_100,
            match_explanation=explanation.summary,
            explanation_method=explanation.method,
            ai_disclaimer=explanation.ai_disclaimer,
            hard_filters_applied=r.hard_filters_applied,
            status_bucket=r.status_bucket,
        ))

    summary = SearchResultsSummary(
        total=len(items),
        strong_matches=sum(1 for it in items if it.match_tier_id in {"strong", "very_strong", "exceptional"}),
        closing_within_30_days=sum(
            1 for it in items if it.deadline is not None and 0 <= (it.deadline - now.date()).days <= 30
        ),
        open_to_india=sum(1 for it in items if it.country == "India"),
        current_count=sum(1 for it in items if it.status_bucket == "open"),
        upcoming_count=sum(1 for it in items if it.status_bucket == "upcoming"),
        needs_verification_count=sum(1 for it in items if it.status_bucket == "needs_verification"),
    )

    return AISearchResponse(
        intent=intent_echo,
        results=items,
        ai_available=ai_used,
        summary=summary,
    )


@router.get("/{opportunity_id}", response_model=OpportunityDetail)
def get_opportunity(opportunity_id: int):
    corpus = get_corpus()
    row = corpus.get_opportunity(opportunity_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Opportunity '{opportunity_id}' not found")
    return _row_to_detail(row)
