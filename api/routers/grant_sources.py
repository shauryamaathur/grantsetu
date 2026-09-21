import logging
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.admin_auth import require_admin
from api.db.models import GrantSource, ImportedGrant, NGOProfile, OpportunityChange, RefreshJob
from api.db.session import SessionLocal, get_db
from api.models.schemas import (
    GrantSourceCreate,
    GrantSourceResponse,
    ImportedGrantDetail,
    ImportedGrantSummary,
    MatchScore,
    MatchScoreComponent,
    OpportunityChangeResponse,
    RefreshJobResponse,
    SourceSyncSummary,
    SyncResultResponse,
)
from src.email.notify import notify_subscribers_of_new_grants
from src.ingestion.sync_grants import sync_source

logger = logging.getLogger("grantsetu.grant_sources")

router = APIRouter(tags=["grant-sources"])


@router.get("/grant-sources", response_model=list[GrantSourceResponse])
def list_grant_sources(db: Session = Depends(get_db)):
    return db.query(GrantSource).all()


@router.post(
    "/admin/grant-sources",
    response_model=GrantSourceResponse,
    status_code=201,
    dependencies=[Depends(require_admin)],
)
def create_grant_source(payload: GrantSourceCreate, db: Session = Depends(get_db)):
    if db.query(GrantSource).filter_by(name=payload.name).first():
        raise HTTPException(status_code=409, detail=f"A grant source named '{payload.name}' already exists")
    source = GrantSource(
        name=payload.name,
        source_url=payload.source_url,
        connector_type=payload.connector_type,
        category=payload.category,
        country=payload.country,
        config=payload.config,
        notes=payload.notes,
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


@router.post(
    "/admin/sync-source/{source_id}",
    response_model=SyncResultResponse,
    dependencies=[Depends(require_admin)],
)
def sync_grant_source(source_id: str, db: Session = Depends(get_db)):
    source = db.get(GrantSource, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail=f"Grant source '{source_id}' not found")

    result = sync_source(db, source)

    notifications_sent = 0
    if result.new_grants:
        logs = notify_subscribers_of_new_grants(db, result.new_grants)
        notifications_sent = len(logs)

    return SyncResultResponse(
        status=result.status,
        fetched=result.fetched,
        new=result.new,
        updated=result.updated,
        duplicates=result.duplicates,
        already_imported=result.already_imported,
        error=result.error,
        notifications_sent=notifications_sent,
    )


def _execute_refresh_job(db: Session, job_id: str) -> None:
    """Does the actual, potentially slow, multi-source sync work. Runs
    OUTSIDE the request/response cycle (called from a FastAPI BackgroundTasks
    callback with its own DB session, or from api/scheduler.py) so an admin
    endpoint's HTTP response is never held open for however long every
    connected source takes to fetch and parse -- see this module's
    module-level note on why that pattern doesn't scale.

    Never lets one source's failure stop the rest: sync_source() already
    turns a connector error into a SyncResult with status="error" rather
    than raising, so one broken source degrades to "0 new from this source"
    in its own summary row instead of blocking every other source's refresh
    or failing the whole job.
    """
    job = db.get(RefreshJob, job_id)
    if job is None:
        logger.error("Refresh job %s vanished before it could run", job_id)
        return

    try:
        job.status = "running"
        db.commit()

        sources = db.query(GrantSource).all()
        job.sources_total = len(sources)
        db.commit()

        all_new_grants: list[ImportedGrant] = []
        per_source: list[dict] = []
        for source in sources:
            result = sync_source(db, source)
            all_new_grants.extend(result.new_grants)
            per_source.append({
                "source_id": source.source_id,
                "name": source.name,
                "status": result.status,
                "fetched": result.fetched,
                "new": result.new,
                "updated": result.updated,
                "error": result.error,
            })
            job.sources_checked += 1
            job.total_fetched += result.fetched
            job.total_new += result.new
            job.total_updated += result.updated
            job.per_source = per_source
            db.commit()

        notifications_sent = 0
        if all_new_grants:
            logs = notify_subscribers_of_new_grants(db, all_new_grants)
            notifications_sent = len(logs)

        job.notifications_sent = notifications_sent
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()
    except Exception as exc:  # noqa: BLE001 -- a job must record its own failure, never crash silently
        logger.exception("Refresh job %s failed", job_id)
        db.rollback()
        job = db.get(RefreshJob, job_id)
        if job is not None:
            job.status = "failed"
            job.error = str(exc)
            job.completed_at = datetime.utcnow()
            db.commit()


def _run_refresh_job_in_background(job_id: str) -> None:
    """BackgroundTasks entrypoint: opens its OWN session, since the
    request-scoped `get_db` session is closed before a BackgroundTasks
    callback runs (same reasoning as api/routers/subscribers.py's
    verification-email background task)."""
    db = SessionLocal()
    try:
        _execute_refresh_job(db, job_id)
    finally:
        db.close()


@router.post("/admin/sync-all-sources", response_model=RefreshJobResponse, status_code=202)
def sync_all_grant_sources(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Refreshes every connected GrantSource -- what powers the All
    Opportunities page's single "Refresh All Sources" control. Loops over
    whatever GrantSource rows actually exist in the database at request
    time (never a hardcoded list of connector names like "Grants.gov"/
    "NGOBox"/"MSJE e-ANUDAAN"), so a newly added connector is picked up
    automatically -- no frontend or backend change needed when a source is
    added.

    Returns immediately with a job_id (HTTP 202) instead of blocking the
    request for however long every source's sync takes -- poll
    GET /admin/refresh-jobs/{job_id} for progress/completion. The actual
    work runs via FastAPI BackgroundTasks, so it survives this request
    ending but NOT the process exiting (a platform that recycles the
    process mid-refresh -- e.g. a Render free-tier spin-down -- would leave
    the job "running" forever; the same external cron/EventBridge trigger
    that calls this endpoint should also alert on a job stuck in "running"
    past a few multiples of a normal refresh's duration).

    Deliberately NOT behind admin-token auth (api/admin_auth.py) -- unlike
    the single-source sync/create-source endpoints below, this is the
    ordinary user-facing "Refresh All Sources" button, reachable by any
    visitor to the All Opportunities page, same as before this change.
    """
    job = RefreshJob(status="queued", trigger="manual")
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_refresh_job_in_background, job.job_id)

    return job


@router.get("/admin/refresh-jobs/{job_id}", response_model=RefreshJobResponse)
def get_refresh_job(job_id: str, db: Session = Depends(get_db)):
    """Polled by the frontend after POST /admin/sync-all-sources to show
    live "i/N sources refreshed" progress and the final summary -- not
    behind admin-token auth, for the same reason sync-all-sources isn't."""
    job = db.get(RefreshJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Refresh job '{job_id}' not found")
    return job


_STATUS_LABELS = {
    "forecasted": "Forecasted (not yet open for applications)",
    "open": "Open",
    "closed": "Currently closed",
    "expired": "Currently closed",
    "unknown": "Status unknown -- verify with the source",
}


def _imported_status_label(grant: ImportedGrant) -> str:
    label = _STATUS_LABELS.get(grant.status)
    if label:
        return label
    if not grant.deadline:
        return "Deadline unavailable"
    return "Status unknown -- verify with the source"


def _imported_source_label(grant: ImportedGrant) -> str:
    if grant.is_sample_data:
        return "Sample data (not a live opportunity)"
    if grant.extraction_status != "complete":
        return "Source information incomplete"
    return "Discovered opportunity"


_CLOSED_LIKE_STATUSES = {"closed", "expired"}


def imported_grant_to_summary(grant: ImportedGrant, source: GrantSource | None = None) -> ImportedGrantSummary:
    """Shared ImportedGrant -> ImportedGrantSummary mapping, used both by
    this module's /opportunities/live and by /opportunities/recommended
    (api/routers/opportunities.py) so the two feeds never drift in how they
    label the same underlying record. `source`, when given, fills in the
    GrantSource-level fields (name/category) the frontend needs to show
    "Source: NGOBox (Aggregator)" instead of a bare source_id."""
    return ImportedGrantSummary(
        imported_grant_id=grant.imported_grant_id,
        title=grant.title,
        funder=grant.funder,
        category=grant.category,
        country=grant.country,
        deadline=grant.deadline.date() if grant.deadline else None,
        url=grant.url,
        status=grant.status,
        is_sample_data=grant.is_sample_data,
        source_id=grant.source_id,
        source_name=source.name if source else None,
        source_category=source.category if source else None,
        extraction_method=grant.extraction_method,
        extraction_status=grant.extraction_status,
        ai_enriched=grant.extraction_method == "ai_inferred",
        evidence_snippet=grant.evidence_snippet,
        status_label=_imported_status_label(grant),
        source_label=_imported_source_label(grant),
        date_collected=grant.date_collected,
        last_verified_at=grant.last_verified_at,
        updated_at=grant.updated_at,
    )


_LIVE_SORT_COLUMNS = {
    "date_collected": ImportedGrant.date_collected,
    "updated_at": ImportedGrant.updated_at,
    "deadline": ImportedGrant.deadline,
    "title": ImportedGrant.title,
}


@router.get("/opportunities/live", response_model=list[ImportedGrantSummary])
def list_live_opportunities(
    ngo_id: str | None = Query(default=None, description="If given, hides opportunities this NGO has hidden"),
    q: str | None = Query(default=None, description="Substring filter on title"),
    category: str | None = Query(default=None),
    funder: str | None = Query(default=None, description="Substring filter on funder name"),
    status: str | None = Query(default=None, description="forecasted | open | closed | expired | unknown"),
    country: str | None = Query(default=None, description="e.g. India | United States | unknown"),
    source_type: str | None = Query(
        default=None, description="government | csr | foundation | aggregator | international | web"
    ),
    active_only: bool = Query(
        default=True,
        description="Default feed excludes closed/expired opportunities. Set false (or pass an explicit "
        "`status`) to see everything, e.g. for a historical/expired view.",
    ),
    sort: str = Query(default="date_collected", description="date_collected | updated_at | deadline | title"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Grants imported through a GrantSource connector -- kept explicitly
    separate from the historical corpus served by /opportunities/search and
    /opportunities/recommended, since the two have different provenance and
    freshness guarantees. Empty until a source has been synced via
    POST /admin/sync-source/{source_id}.

    The DEFAULT view is "what's available right now": `active_only=True`
    excludes closed/expired rows unless a `status` filter is explicitly
    given (an explicit request for status=closed is always honored, since
    active_only is a DEFAULT-view convenience, not a hard restriction).

    ALWAYS excludes is_sample_data=True rows -- sample/fixture data exists
    solely to prove the sync/dedup/notification pipeline works in
    automated tests and manual admin verification; it must never reach a
    real user through this (or any other) user-facing listing. There is
    deliberately no query parameter to opt back into seeing it here -- an
    admin who genuinely needs to inspect sample rows can query the
    database directly."""
    query = db.query(ImportedGrant).filter_by(is_sample_data=False)
    if q:
        query = query.filter(ImportedGrant.title.ilike(f"%{q}%"))
    if category:
        query = query.filter(ImportedGrant.category.ilike(f"%{category}%"))
    if funder:
        query = query.filter(ImportedGrant.funder.ilike(f"%{funder}%"))
    if country:
        query = query.filter(ImportedGrant.country == country)
    if status:
        query = query.filter_by(status=status)
    elif active_only:
        query = query.filter(ImportedGrant.status.notin_(_CLOSED_LIKE_STATUSES))
    if source_type:
        query = query.join(GrantSource, GrantSource.source_id == ImportedGrant.source_id).filter(
            GrantSource.category == source_type
        )

    sort_col = _LIVE_SORT_COLUMNS.get(sort, ImportedGrant.date_collected)
    query = query.order_by(sort_col.desc() if sort != "title" else sort_col.asc())

    fetch_limit = limit * 2 if ngo_id else limit
    rows = query.offset(offset).limit(fetch_limit).all()

    if ngo_id:
        from api.db.models import OpportunityFeedback

        hidden_refs = {
            ref
            for (ref,) in db.query(OpportunityFeedback.opportunity_ref)
            .filter_by(ngo_id=ngo_id, opportunity_source="imported", feedback_type="hidden")
            .all()
        }
        rows = [r for r in rows if r.imported_grant_id not in hidden_refs][:limit]

    source_ids = {r.source_id for r in rows}
    sources_by_id = {
        s.source_id: s for s in db.query(GrantSource).filter(GrantSource.source_id.in_(source_ids)).all()
    } if source_ids else {}

    return [imported_grant_to_summary(r, sources_by_id.get(r.source_id)) for r in rows]


@router.get("/opportunities/live/{imported_grant_id}", response_model=ImportedGrantDetail)
def get_live_opportunity(
    imported_grant_id: str,
    ngo_id: str | None = Query(default=None, description="If given, includes this NGO's own match score/breakdown"),
    db: Session = Depends(get_db),
):
    """Full detail for one live opportunity -- powers the opportunity detail
    page. Returns 404 for sample-data rows too (is_sample_data=True is never
    a real, viewable opportunity, same rule as GET /opportunities/live).
    When `ngo_id` is given and the NGO profile exists, includes that NGO's
    own match score (freshly computed against just this one opportunity --
    the same scoring function /opportunities/recommended uses, so the two
    never disagree about the same grant/NGO pair) and its recent change
    history (most recent 5)."""
    grant = db.get(ImportedGrant, imported_grant_id)
    if grant is None or grant.is_sample_data:
        raise HTTPException(status_code=404, detail=f"Live opportunity '{imported_grant_id}' not found")

    source = db.get(GrantSource, grant.source_id)
    summary = imported_grant_to_summary(grant, source)

    match_score = None
    if ngo_id:
        profile = db.get(NGOProfile, ngo_id)
        if profile is not None:
            from src.ranking.live_match import build_canonical_match_scores

            # THE canonical scoring call (src/ranking/live_match.py) -- the
            # SAME function GET /opportunities/recommended and POST
            # /opportunities/ai-search read from, so this grant's score can
            # never disagree with what those surfaces just showed for it.
            # A grant absent from the canonical set (e.g. since closed, so
            # outside RECOMMENDABLE_STATUSES) correctly shows no score here
            # either, rather than a one-off recomputation that could
            # disagree with /recommended's own honest "not recommendable"
            # treatment of the same grant.
            match = build_canonical_match_scores(
                db, profile.focus_areas, profile.beneficiaries,
                locations=profile.locations, registration_type=profile.registration_type,
            ).get(imported_grant_id)
            if match:
                m = match.match_score
                match_score = MatchScore(
                    score_100=m.score_100, tier=m.tier, tier_label=m.tier_label,
                    components=[MatchScoreComponent(label=c.label, points=c.points, detail=c.detail) for c in m.components],
                )

    changes = (
        db.query(OpportunityChange)
        .filter_by(imported_grant_id=imported_grant_id)
        .order_by(OpportunityChange.detected_at.desc())
        .limit(5)
        .all()
    )

    return ImportedGrantDetail(
        **summary.model_dump(),
        description=grant.description,
        eligibility_text=grant.eligibility_text,
        source_url=source.source_url if source else None,
        match_score=match_score,
        recent_changes=changes,
    )


@router.get("/opportunities/live/{imported_grant_id}/changes", response_model=list[OpportunityChangeResponse])
def list_opportunity_changes(imported_grant_id: str, db: Session = Depends(get_db)):
    """Change history for one live opportunity (src/ingestion/sync_grants.py
    ::_apply_changes) -- e.g. "deadline_changed" from an earlier date to a
    later one, "reopened", "closed". Newest first. Empty (not 404) for a
    grant that has never changed since it was first discovered -- that's a
    normal, common case, not an error."""
    if db.get(ImportedGrant, imported_grant_id) is None:
        raise HTTPException(status_code=404, detail=f"Live opportunity '{imported_grant_id}' not found")
    rows = (
        db.query(OpportunityChange)
        .filter_by(imported_grant_id=imported_grant_id)
        .order_by(OpportunityChange.detected_at.desc())
        .all()
    )
    return rows
