"""Pydantic request/response schemas for the GrantSetu API."""
from datetime import date, datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ai.providers import SUPPORTED_PROVIDERS

VALID_REGISTRATION_TYPES = {"trust", "society", "section_8_company", "other"}
# Simple application-tracking vocabulary (premium-upgrade phase). Previously:
# researching/drafting/submitted/outcome_known -- renamed in place to match
# the simpler states requested for the Saved Grants screen.
VALID_STATUSES = {"saved", "planning_to_apply", "applied", "awarded", "rejected"}


def _validate_url(v: str | None, field_name: str = "URL") -> str | None:
    if v is None or v == "":
        return None
    parsed = urlparse(v)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be a valid http(s) URL")
    return v


class NGOProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    registration_type: str | None = Field(default=None)
    website_url: str | None = Field(default=None, max_length=500)
    focus_areas: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    beneficiaries: list[str] = Field(default_factory=list)

    @field_validator("registration_type")
    @classmethod
    def check_registration_type(cls, v):
        if v is not None and v.lower() not in VALID_REGISTRATION_TYPES:
            raise ValueError(f"registration_type must be one of {sorted(VALID_REGISTRATION_TYPES)}")
        return v.lower() if v else v

    @field_validator("website_url")
    @classmethod
    def check_website_url(cls, v):
        return _validate_url(v, "website_url")

    @field_validator("focus_areas")
    @classmethod
    def check_focus_areas(cls, v):
        if not v:
            raise ValueError("focus_areas must include at least one focus area for meaningful matching")
        return v


class NGOProfileResponse(NGOProfileCreate):
    ngo_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ScoreBreakdown(BaseModel):
    total_score: float
    text_relevance: float
    category_overlap: float
    recency_score: float
    eligibility_multiplier: float


class OpportunitySummary(BaseModel):
    opportunity_id: int
    opportunity_title: str
    agency_name: str | None = None
    opportunity_category: str | None = None
    close_date: date | None = None
    is_expired: bool
    award_size_bucket: str | None = None
    # Honest, opportunity-level status labels (never a blanket "this whole
    # product is fake" message) -- see api/routers/opportunities.py::_status_labels.
    status_label: str
    source_label: str


class RankedOpportunity(BaseModel):
    opportunity: OpportunitySummary
    match_summary: str
    match_reasons: list[str]
    score_breakdown: ScoreBreakdown


class OpportunityDetail(OpportunitySummary):
    opportunity_number: str | None = None
    funding_instrument_type: str | None = None
    category_of_funding_activity: str | None = None
    eligible_applicants: str | None = None
    eligible_applicants_type: str | None = None
    eligibility_status_label: str
    post_date: date | None = None
    award_ceiling: float | None = None
    award_floor: float | None = None
    estimated_total_program_funding: float | None = None
    additional_information_url: str | None = None


VALID_SAVE_SOURCES = {"historical", "imported"}


class SavedOpportunityCreate(BaseModel):
    ngo_id: str
    # opportunity_id is kept for backward compatibility with the original
    # historical-only save flow -- when opportunity_source="historical" and
    # opportunity_ref is omitted, opportunity_id is used as the ref too.
    opportunity_id: int | None = None
    opportunity_source: str = "historical"
    opportunity_ref: str | None = None
    notes: str | None = None
    assigned_to: str | None = None

    @field_validator("opportunity_source")
    @classmethod
    def check_source(cls, v):
        if v not in VALID_SAVE_SOURCES:
            raise ValueError(f"opportunity_source must be one of {sorted(VALID_SAVE_SOURCES)}")
        return v


class SavedOpportunityUpdate(BaseModel):
    status: str | None = None
    assigned_to: str | None = None
    notes: str | None = None

    @field_validator("status")
    @classmethod
    def check_status(cls, v):
        if v is not None and v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v


class SavedOpportunityResponse(BaseModel):
    saved_id: str
    ngo_id: str
    opportunity_id: int
    opportunity_source: str
    opportunity_ref: str | None
    status: str
    assigned_to: str | None
    notes: str | None
    saved_at: datetime
    # Enriched at read time so the frontend never needs a second fetch to
    # render a saved-grants list -- see api/routers/saved.py::_enrich.
    title: str | None = None
    funder: str | None = None
    deadline: date | None = None
    is_expired: bool | None = None
    url: str | None = None

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Subscribers / email notifications
# ---------------------------------------------------------------------------

class SubscribeRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    ngo_id: str | None = None

    @field_validator("email")
    @classmethod
    def check_email_shape(cls, v):
        # Deliberately simple validation (not a full RFC 5322 parser) --
        # good enough to catch obvious typos without an extra dependency.
        if "@" not in v or "." not in v.split("@")[-1] or " " in v:
            raise ValueError("email does not look like a valid email address")
        return v.strip().lower()


class SubscriberResponse(BaseModel):
    subscriber_id: str
    email: str
    ngo_id: str | None
    status: str  # pending | verified | unsubscribed | bounced | disabled
    is_verified: bool  # derived from status -- kept for backward compatibility
    is_unsubscribed: bool  # derived from status -- kept for backward compatibility
    notify_new_grants: bool
    notify_digest: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SubscriberPreferencesUpdate(BaseModel):
    ngo_id: str | None = None
    notify_new_grants: bool | None = None
    notify_digest: bool | None = None


# ---------------------------------------------------------------------------
# Grant sources / imported (live/current) grants
# ---------------------------------------------------------------------------

VALID_SOURCE_CATEGORIES = {"government", "csr", "foundation", "aggregator", "international", "web"}


class GrantSourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    source_url: str | None = None
    connector_type: str
    category: str | None = None
    country: str | None = None
    config: dict = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("category")
    @classmethod
    def check_category(cls, v):
        if v is not None and v not in VALID_SOURCE_CATEGORIES:
            raise ValueError(f"category must be one of {sorted(VALID_SOURCE_CATEGORIES)}")
        return v


class GrantSourceResponse(BaseModel):
    source_id: str
    name: str
    source_url: str | None
    connector_type: str
    category: str | None = None
    country: str | None = None
    config: dict
    status: str
    last_synced_at: datetime | None
    last_sync_summary: str | None
    notes: str | None

    model_config = ConfigDict(from_attributes=True)


class SyncResultResponse(BaseModel):
    status: str
    fetched: int
    new: int
    updated: int = 0
    duplicates: int
    already_imported: int
    error: str | None
    notifications_sent: int


class SourceSyncSummary(BaseModel):
    """One GrantSource's outcome within a POST /admin/sync-all-sources
    background job -- kept per-source so a single failing connector is
    visible without hiding the other sources' results (see
    RefreshJobResponse.per_source)."""
    source_id: str
    name: str
    status: str  # not_configured | active | error
    fetched: int
    new: int
    updated: int
    error: str | None


class RefreshJobResponse(BaseModel):
    """Status/result of one POST /admin/sync-all-sources background job
    (api/db/models.py::RefreshJob) -- what powers the All Opportunities
    page's single "Refresh All Sources" control (never a hardcoded
    per-connector button). Returned immediately (status=queued) by the POST
    that creates the job, then polled via GET /admin/refresh-jobs/{job_id}
    until status is completed|failed. `per_source` lists every GrantSource
    row that was actually queried, so a newly added connector appears here
    automatically without any frontend change."""
    model_config = ConfigDict(from_attributes=True)

    job_id: str
    status: str  # queued | running | completed | failed
    trigger: str  # manual | scheduled
    sources_checked: int
    sources_total: int
    total_fetched: int
    total_new: int
    total_updated: int
    notifications_sent: int
    per_source: list[SourceSyncSummary]
    error: str | None
    created_at: datetime
    completed_at: datetime | None


class OpportunityChangeResponse(BaseModel):
    change_id: str
    change_type: str
    old_value: str | None
    new_value: str | None
    detected_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ImportedGrantSummary(BaseModel):
    imported_grant_id: str
    title: str
    funder: str | None
    category: str | None
    country: str = "unknown"
    deadline: date | None
    url: str | None
    status: str
    is_sample_data: bool
    source_id: str
    source_name: str | None = None
    source_category: str | None = None
    extraction_method: str
    extraction_status: str
    ai_enriched: bool = False
    evidence_snippet: str | None
    status_label: str
    source_label: str
    date_collected: datetime
    last_verified_at: datetime
    updated_at: datetime | None = None


class MatchScoreComponent(BaseModel):
    label: str
    points: int
    detail: str


class MatchScore(BaseModel):
    """An explainable 0-100 relevance score (never a decimal, never framed
    as a probability of winning funding) -- see
    src/ranking/live_match.py::build_match_score. `components` lists only
    the factors that actually contributed; nothing is fabricated to fill
    out a breakdown."""
    score_100: int = Field(..., ge=0, le=100)
    tier: str  # minimal | low | moderate | good | strong | very_strong | exceptional
    tier_label: str
    components: list[MatchScoreComponent]


class ImportedGrantDetail(ImportedGrantSummary):
    """Full detail for one live opportunity (GET /opportunities/live/{id}) --
    everything ImportedGrantSummary has, plus the longer text fields and,
    when `ngo_id` was given, this NGO's own match score against it (the
    same score/breakdown shape as /opportunities/recommended, computed
    fresh -- never cached across NGOs, since it's genuinely per-profile)."""
    description: str | None = None
    eligibility_text: str | None = None
    source_url: str | None = None
    match_score: MatchScore | None = None
    recent_changes: list[OpportunityChangeResponse] = Field(default_factory=list)


class RankedImportedGrant(BaseModel):
    """A live-discovered (ImportedGrant) opportunity ranked for one NGO's
    profile -- the shape /opportunities/recommended returns."""
    opportunity: ImportedGrantSummary
    match_summary: str
    match_reasons: list[str]
    score: float
    match_score: MatchScore


# ---------------------------------------------------------------------------
# Opportunity feedback (hide / report incorrect)
# ---------------------------------------------------------------------------

VALID_FEEDBACK_TYPES = {"hidden", "reported_incorrect"}
VALID_OPPORTUNITY_SOURCES = {"historical", "imported"}


class OpportunityFeedbackCreate(BaseModel):
    ngo_id: str
    opportunity_source: str
    opportunity_ref: str
    feedback_type: str
    notes: str | None = None

    @field_validator("opportunity_source")
    @classmethod
    def check_source(cls, v):
        if v not in VALID_OPPORTUNITY_SOURCES:
            raise ValueError(f"opportunity_source must be one of {sorted(VALID_OPPORTUNITY_SOURCES)}")
        return v

    @field_validator("feedback_type")
    @classmethod
    def check_type(cls, v):
        if v not in VALID_FEEDBACK_TYPES:
            raise ValueError(f"feedback_type must be one of {sorted(VALID_FEEDBACK_TYPES)}")
        return v


class OpportunityFeedbackResponse(BaseModel):
    feedback_id: str
    ngo_id: str
    opportunity_source: str
    opportunity_ref: str
    feedback_type: str
    notes: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# AI provider configuration
# ---------------------------------------------------------------------------

class AIProviderConfigCreate(BaseModel):
    provider: str
    api_key: str = Field(..., min_length=8)
    model: str | None = None
    base_url: str | None = None
    is_enabled: bool = True

    @field_validator("provider")
    @classmethod
    def check_provider(cls, v):
        if v not in SUPPORTED_PROVIDERS:
            raise ValueError(f"provider must be one of {SUPPORTED_PROVIDERS}")
        return v

    @field_validator("base_url")
    @classmethod
    def check_base_url(cls, v):
        return _validate_url(v, "base_url")


class AIProviderConfigUpdate(BaseModel):
    api_key: str | None = Field(default=None, min_length=8)
    model: str | None = None
    base_url: str | None = None
    is_enabled: bool | None = None

    @field_validator("base_url")
    @classmethod
    def check_base_url(cls, v):
        return _validate_url(v, "base_url")


class AIProviderConfigResponse(BaseModel):
    """NEVER includes the API key, encrypted or otherwise -- only whether
    one is stored."""
    ngo_id: str
    provider: str
    model: str | None
    base_url: str | None
    is_enabled: bool
    has_api_key: bool
    last_test_status: str | None
    last_test_message: str | None
    last_tested_at: datetime | None


class AIProviderTestResult(BaseModel):
    ok: bool
    message: str


# ---------------------------------------------------------------------------
# Website analysis
# ---------------------------------------------------------------------------

class WebsiteAnalysisResponse(BaseModel):
    analysis_id: str
    ngo_id: str
    website_url: str
    summary: str | None
    detected_focus_areas: list[str]
    detected_locations: list[str]
    detected_beneficiaries: list[str]
    keywords: list[str]
    pages_analyzed: list[str]
    method: str
    status: str
    error: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class WebsiteAnalysisApply(BaseModel):
    apply_focus_areas: bool = True
    apply_locations: bool = True
    apply_beneficiaries: bool = True


# ---------------------------------------------------------------------------
# AI-powered natural-language grant search
# ---------------------------------------------------------------------------

class AISearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    ngo_id: str | None = None
    limit: int = Field(default=20, ge=1, le=50)


class AISearchIntentEcho(BaseModel):
    """The extracted filters, echoed back so the UI/caller can show the user
    what was understood from their query -- never hidden state."""
    method: str  # "ai" | "rule_based"
    keywords: list[str]
    locations: list[str]
    causes: list[str]
    applicant_type: str | None
    deadline_within_days: int | None
    international_ok: bool | None
    funding_min: float | None
    funding_max: float | None
    search_scope: str  # "current" | "historical" | "research" -- see src/ai/query_understanding.py


class AISearchResultItem(BaseModel):
    source: str  # "historical" | "imported"
    ref: str
    title: str
    funder: str | None
    category: str | None
    country: str | None = None
    deadline: date | None
    url: str | None
    status_label: str
    source_label: str
    is_expired: bool | None
    match_tier: str
    # A short, stable id for CSS/styling purposes ("strong", "moderate", ...)
    # -- see src/ranking/live_match.py's TIER_* constants. `match_tier`
    # above stays the human-readable label shown in the UI; this is never
    # displayed directly.
    match_tier_id: str
    # Integer 0-100. When `ngo_id` was given and this is a live ("imported")
    # opportunity, this is the CANONICAL score (src/ranking/live_match.py::
    # build_canonical_match_scores) -- the identical number GET
    # /opportunities/recommended and GET /opportunities/live/{id} show for
    # the same grant+NGO pair, never independently recalculated. Otherwise
    # (no ngo_id, or a historical-corpus result with no canonical
    # equivalent) it's the raw TF-IDF relevance score, and null when even
    # that isn't a genuine relevance number (AISearchResult.relevance_known
    # =False, e.g. a pure deadline-window query with no topic keywords) --
    # never a misleadingly precise score built from the internal 1.0
    # sentinel.
    match_score_100: int | None = None
    match_explanation: str
    explanation_method: str  # "ai" | "rule_based"
    ai_disclaimer: str | None
    hard_filters_applied: list[str]
    status_bucket: str  # "open" | "upcoming" | "needs_verification" | "historical" -- see src/ranking/ai_search.py


class SearchResultsSummary(BaseModel):
    """The contextual "47 found, 18 open to Indian NGOs, 7 closing within 30
    days, 12 strong matches" line -- every count computed from the actual
    result set, never estimated or fabricated. `open_to_india` only counts
    opportunities confidently tagged country="India" (see src/ai/
    research.py::_detect_country). current/upcoming/needs_verification
    counts partition the same result set by `status_bucket` (they sum to
    `total`) -- what powers the "8 current · 3 upcoming · 1 needs
    verification" summary line. There is deliberately no historical_count:
    Ask GrantSetu never returns historical opportunities at all (see
    src/ranking/ai_search.py's module docstring), so a historical bucket
    would always be zero by construction."""
    total: int
    strong_matches: int
    closing_within_30_days: int
    open_to_india: int
    current_count: int
    upcoming_count: int
    needs_verification_count: int


# ---------------------------------------------------------------------------
# Google authentication (optional -- guest usage always fully works)
# ---------------------------------------------------------------------------

class AuthConfigResponse(BaseModel):
    google_enabled: bool


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str | None
    picture_url: str | None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MigrateGuestRequest(BaseModel):
    ngo_id: str


class AISearchResponse(BaseModel):
    intent: AISearchIntentEcho
    results: list[AISearchResultItem]
    summary: SearchResultsSummary
    ai_available: bool  # whether an AI provider was actually used anywhere in this response
    # Set ONLY when the query was recognized as historical-scope
    # (src/ai/query_understanding.py's SCOPE_HISTORICAL) -- Ask GrantSetu
    # never searches historical data, so `results` is always empty
    # whenever this is set, and the frontend should show this message with
    # a link to Grant Intelligence instead of an opportunity list.
    scope_notice: str | None = None


# ---------------------------------------------------------------------------
# Floating AI assistant (grounded Q&A)
# ---------------------------------------------------------------------------

VALID_ASSISTANT_CONTEXT_TYPES = {"opportunity", "ngo", "search", "analytics", "general"}


class AssistantContext(BaseModel):
    type: str = "general"  # opportunity | ngo | search | analytics | general
    # context.type == "opportunity"
    opportunity_source: str | None = None  # "imported" | "historical"
    opportunity_ref: str | None = None
    # context.type == "search" -- the frontend passes its OWN already-fetched
    # /opportunities/ai-search results here, so this endpoint never re-runs
    # the search (staying grounded in exactly what the user is looking at).
    search_query: str | None = None
    search_results: list[dict] | None = None
    # context.type == "analytics" -- the frontend passes an already-computed
    # /analytics/* payload here; this endpoint never recomputes it.
    analytics: dict | None = None

    @field_validator("type")
    @classmethod
    def check_type(cls, v):
        if v not in VALID_ASSISTANT_CONTEXT_TYPES:
            raise ValueError(f"context.type must be one of {sorted(VALID_ASSISTANT_CONTEXT_TYPES)}")
        return v


class AssistantRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    ngo_id: str | None = None
    context: AssistantContext | None = None


class AssistantResponse(BaseModel):
    answer: str
    method: str  # "ai" | "rule_based"
    ai_disclaimer: str | None
    grounded_on: list[str]
    data_available: bool
