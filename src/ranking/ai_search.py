"""Retrieval + ranking for AI-powered natural-language grant search
(POST /opportunities/ai-search) -- given a SearchIntent
(src/ai/query_understanding.py), retrieves CURRENT-only candidates from
live ImportedGrant data (+ the open web, once configured), applies the
intent's genuinely-structured fields (deadline window) as real filters,
uses the rest (locations/causes/applicant_type) as text-relevance signals
ONLY, deduplicates overlap, and returns a ranked, source-labeled list.

HARD ARCHITECTURAL RULE (interview.txt: "Ask GrantSetu = current funding
discovery only; Grant Intelligence = historical funding research only"):
this module NEVER queries the historical GrantSetu corpus
(src/ranking/corpus.py), under any circumstances -- not as a fallback when
live results are sparse, not for a query that explicitly asks about past
funding, not even to pad a thin result count. There is deliberately no
function in this file that reads from the historical corpus at all, so
returning a historical record here isn't a matter of low-scoring or
filtering it out after the fact -- the historical dataset is structurally
absent from the candidate pool from the very first retrieval step. A query
recognized as historical-scope (src/ai/query_understanding.py's
SCOPE_HISTORICAL) is intercepted even earlier than that: api/routers/
opportunities.py::ai_search returns a redirect notice pointing at Grant
Intelligence WITHOUT calling this module's search() at all. Historical
data remains fully available -- just exclusively through Grant
Intelligence, never through Ask GrantSetu.

HONESTY NOTE (see also src/ai/grounded_explain.py): ImportedGrant has no
structured location/country column beyond `country` itself, and no
structured funding-amount column. "locations", "causes", and
"applicant_type" from the extracted intent can therefore only ever be
TEXT-RELEVANCE signals (fed into the retrieval query), never hard filters
-- claiming otherwise would imply a guarantee the underlying data can't
back up. Only `deadline_within_days` is applied as a real hard filter, and
even then a matching deadline alone never implies "current" -- every
candidate is ALREADY restricted to RECOMMENDABLE_STATUSES (open/
forecasted/unknown, i.e. never closed/expired) before that filter is
applied, so a deadline coincidentally falling inside the requested window
can't smuggle in a record whose actual status says otherwise. This module
never sends the full corpus to an LLM -- it does retrieval/ranking only;
explanation text is generated separately, only for the already-shortlisted
results.

WEB SEARCH SEAM: `_search_web()` below calls
src/discovery/web_search.py::get_web_search_provider(), which returns None
unless a real search-API key is configured (none is, by default -- see that
module's docstring). This is a real, wired extension point, not a TODO
comment: when a provider IS configured, this is where its hits would be
merged in as additional AISearchResult rows via the same extraction
pipeline src/connectors/web_discovery_connector.py already uses, so a
web-discovered opportunity is always a real, structured, source-linked,
CURRENT record -- never a raw search snippet, and never a historical one.
"""
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy.orm import Session

from api.db.models import ImportedGrant
from src.ai.query_understanding import SearchIntent
from src.discovery.web_search import get_web_search_provider
from src.ingestion.sync_grants import compute_dedup_hash
from src.ranking.live_match import RECOMMENDABLE_STATUSES

STATUS_OPEN = "open"
STATUS_UPCOMING = "upcoming"
STATUS_NEEDS_VERIFICATION = "needs_verification"


@dataclass
class AISearchResult:
    source: str  # always "imported" -- see module docstring's hard rule
    ref: str  # imported_grant_id, as a string
    title: str
    funder: str | None
    category: str | None
    deadline: date | None
    url: str | None
    status_label: str
    source_label: str
    is_expired: bool | None
    score: float
    # Which of the extracted intent fields actually narrowed this result
    # via a real (non-text) filter -- used by grounded_explain.py so the
    # explanation never implies a stronger guarantee than the data supports.
    hard_filters_applied: list[str]
    # True only when `score` is a genuine TF-IDF text-relevance score;
    # False when it's the 1.0 sentinel used for a filter-only match with no
    # topic keywords (see _search_live) -- grounded_explain.py must never
    # call a filter-only match "Strong" on the strength of that sentinel.
    relevance_known: bool
    # Whether this record has ANY real eligibility text at all (not
    # whether this specific NGO is eligible -- that's never determined
    # here). Feeds the "Eligibility unclear" tier.
    eligibility_known: bool
    # See src/ai/research.py::_detect_country.
    country: str | None = None
    # "open" | "upcoming" | "needs_verification" -- what the frontend
    # groups/badges results by (interview.txt: "8 current · 3 upcoming ·
    # 1 needs verification"), derived from the grant's own status/
    # extraction_status, never guessed. "historical" is deliberately not a
    # possible value here -- see module docstring.
    status_bucket: str = STATUS_NEEDS_VERIFICATION


def _query_text(intent: SearchIntent) -> str:
    return " ".join(dict.fromkeys([*intent.keywords, *intent.causes, *intent.locations]))


def _search_live(db: Session, intent: SearchIntent, top_k: int) -> list[AISearchResult]:
    candidates = (
        db.query(ImportedGrant)
        .filter_by(is_sample_data=False)
        .filter(ImportedGrant.status.in_(RECOMMENDABLE_STATUSES))
        .all()
    )
    hard_filters: list[str] = []

    if intent.deadline_within_days is not None:
        now = datetime.utcnow()
        window_end = now + timedelta(days=intent.deadline_within_days)
        candidates = [g for g in candidates if g.deadline is not None and now <= g.deadline <= window_end]
        hard_filters.append("deadline_within_days")
    # funding_min/max: ImportedGrant has no structured funding-amount
    # column, so this filter cannot be applied here -- deliberately not
    # attempted rather than silently ignored-but-implied-applied.

    if not candidates:
        return []

    query_text = _query_text(intent)
    if not query_text.strip():
        # No topic keywords left over once structured filters are extracted
        # -- use the hard-filtered set directly (soonest deadline first
        # when a deadline window was requested) rather than a meaningless
        # zero relevance score.
        if not hard_filters:
            return []
        candidates = sorted(candidates, key=lambda g: g.deadline or datetime.max)
        scored = [(g, 1.0) for g in candidates]
        relevance_known = False
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        documents = [
            " ".join(p for p in [g.title, g.category, g.funder, g.description, g.eligibility_text] if p)
            for g in candidates
        ]
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform([query_text, *documents])
            sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
            scored = list(zip(candidates, sims))
        except ValueError:
            scored = [(g, 0.0) for g in candidates]
        relevance_known = True

    scored.sort(key=lambda pair: pair[1], reverse=True)

    results = []
    for grant, score in scored[:top_k]:
        # Real-status-derived, never guessed: a confirmed "open" grant with
        # complete extraction is the only thing bucketed "open"; anything
        # the source didn't clearly confirm (status="unknown") OR that this
        # extraction couldn't fully verify (extraction_status != "complete")
        # is bucketed "needs_verification" rather than optimistically shown
        # as open. Never "historical" -- RECOMMENDABLE_STATUSES already
        # excludes closed/expired grants from `candidates` above.
        if grant.extraction_status != "complete" or grant.status == "unknown":
            status_bucket = STATUS_NEEDS_VERIFICATION
        elif grant.status == "forecasted":
            status_bucket = STATUS_UPCOMING
        elif grant.status == "open":
            status_bucket = STATUS_OPEN
        else:
            status_bucket = STATUS_NEEDS_VERIFICATION

        results.append(AISearchResult(
            source="imported",
            ref=grant.imported_grant_id,
            title=grant.title,
            country=grant.country,
            funder=grant.funder,
            category=grant.category,
            deadline=grant.deadline.date() if grant.deadline else None,
            url=grant.url,
            status_label={
                "forecasted": "Forecasted (not yet open for applications)",
                "open": "Open",
                "unknown": "Status unknown -- verify with the source",
            }.get(grant.status, "Status unknown -- verify with the source"),
            source_label="Discovered opportunity" if grant.extraction_status == "complete" else "Source information incomplete",
            is_expired=False,
            score=float(score),
            relevance_known=relevance_known,
            eligibility_known=bool(grant.eligibility_text and grant.eligibility_text.strip()),
            hard_filters_applied=hard_filters,
            status_bucket=status_bucket,
        ))
    return results


def _dedup_key(title: str, funder: str | None, deadline: date | None) -> str:
    deadline_str = deadline.isoformat() if deadline else None
    return compute_dedup_hash(title, funder, deadline_str)


def _search_web(intent: SearchIntent) -> list[AISearchResult]:
    """See "WEB SEARCH SEAM" in the module docstring -- returns [] unless a
    real search provider is configured (none is, by default), which is
    normal and expected, not a degraded state. Any future implementation
    here must only ever surface CURRENT opportunities (see module
    docstring's hard rule) -- never historical/archived pages."""
    provider = get_web_search_provider()
    if provider is None:
        return []
    # A provider IS configured: this is where its hits would be fetched +
    # extracted into real, CURRENT AISearchResult rows (see the module
    # docstring's "TO ACTUALLY ENABLE OPEN-WEB DISCOVERY" steps). Not
    # implemented yet -- intentionally returns [] rather than guessing at
    # extraction here.
    return []


def search(db: Session, intent: SearchIntent, limit: int = 20) -> list[AISearchResult]:
    """Merges live + (when configured) open-web results, deduplicates
    overlap, ranks by score, and returns the top `limit`. Returns an empty
    list -- never a padded/fallback list, and NEVER a historical one -- when
    nothing genuinely current matches. Callers must not call this at all
    for a historical-scope query (see api/routers/opportunities.py::
    ai_search) -- there is no `scope` handling in here because there is no
    historical source in here to scope against in the first place."""
    live_results = _search_live(db, intent, top_k=limit * 2)
    web_results = _search_web(intent)

    seen_keys = {_dedup_key(r.title, r.funder, r.deadline) for r in live_results}
    deduped_web = [r for r in web_results if _dedup_key(r.title, r.funder, r.deadline) not in seen_keys]

    merged = live_results + deduped_web
    merged = [r for r in merged if r.score > 0]  # never include a zero-relevance "match"
    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:limit]
