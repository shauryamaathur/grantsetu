"""SQLAlchemy ORM models.

Originally scoped to the MVP schema (blueprint Section K): NGOProfile,
SavedOpportunity, SearchHistory. Extended in the "premium upgrade" phase with:
Subscriber + EmailLog (email subscription/notification foundation) and
GrantSource + ImportedGrant (current/live-grant discovery foundation, fed by
connectors in src/connectors/).

`opportunities` (the ~75k historical corpus) is still NOT a database table --
it stays in the in-process Corpus object (src/ranking/corpus.py), since it is
a bulk-rebuilt, mostly-read-only artifact. ImportedGrant is a SEPARATE table
for grants added incrementally through a source connector (Section 4 of the
premium-upgrade request) -- this is the mechanism for "current" grants, kept
deliberately distinct from the historical demo corpus so the two are never
confused in the API or UI.
"""
import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class NGOProfile(Base):
    __tablename__ = "ngo_profiles"

    ngo_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False)
    registration_type: Mapped[str | None] = mapped_column(String, nullable=True)
    website_url: Mapped[str | None] = mapped_column(String, nullable=True)
    focus_areas: Mapped[list] = mapped_column(JSON, default=list)
    locations: Mapped[list] = mapped_column(JSON, default=list)
    beneficiaries: Mapped[list] = mapped_column(JSON, default=list)
    # NULL means this profile was created by a guest (no sign-in) -- the
    # ORIGINAL, still fully-supported way to use GrantSetu (see interview.txt:
    # "SIGN-IN MUST NOT BE REQUIRED"). Set only via an explicit, user-confirmed
    # POST /auth/migrate-guest call -- NEVER silently associated with a user
    # just because they happen to be signed in when the profile was created.
    user_id: Mapped[str | None] = mapped_column(String, ForeignKey("users.user_id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    saved_opportunities: Mapped[list["SavedOpportunity"]] = relationship(back_populates="ngo")
    subscribers: Mapped[list["Subscriber"]] = relationship(back_populates="ngo")


class User(Base):
    """A Google-authenticated GrantSetu account. Existing purely for
    PERSISTENCE (see interview.txt Section 17: "Signing in should provide
    persistence, not permission") -- every feature GrantSetu offers already
    works fully for a guest with no User row at all; signing in only lets an
    NGOProfile/SavedOpportunity/etc be reattached to a durable identity
    instead of a browser's localStorage. `google_sub` is Google's own stable
    subject identifier (the ID token's `sub` claim) -- the correct field to
    key off of, NOT email (a Google account's email can change)."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    google_sub: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    email: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[str | None] = mapped_column(String, nullable=True)
    picture_url: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UserSession(Base):
    """A signed-in browser session. Same hashed-token pattern as
    Subscriber.verify_token (src/email/tokens.py) -- the raw session token
    exists only in the response the browser stores (localStorage) and is
    NEVER persisted in plaintext; only its SHA-256 hash is stored here, so a
    database compromise alone can't be used to impersonate a signed-in user."""

    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)  # SHA-256 hash of the raw token
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.user_id"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SavedOpportunity(Base):
    """A grant an NGO has bookmarked. Can point at either the historical
    corpus (int `opportunity_id`, the original MVP shape -- kept for
    backward compatibility) or a live `ImportedGrant` row. `opportunity_source`
    + `opportunity_ref` (added later; see api/db/migrations.py) together
    identify the target the same way OpportunityFeedback does, since the two
    corpora use different ID spaces. For a historical save, `opportunity_id`
    is the real int id and `opportunity_ref` mirrors it as a string; for an
    imported/live save, `opportunity_id` holds the sentinel -1 (the column
    predates this change and is still NOT NULL) and `opportunity_ref` holds
    the real ImportedGrant.imported_grant_id."""

    __tablename__ = "saved_opportunities"
    __table_args__ = (
        UniqueConstraint(
            "ngo_id", "opportunity_source", "opportunity_ref",
            name="uq_saved_opportunities_target",
        ),
    )

    IMPORTED_ID_SENTINEL = -1

    # Status vocabulary matches the simple application-tracking states asked
    # for in the premium-upgrade request: saved -> planning_to_apply ->
    # applied -> awarded / rejected. (Previously: researching/drafting/
    # submitted/outcome_known -- renamed in place, not duplicated.)
    saved_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    ngo_id: Mapped[str] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), nullable=False)
    opportunity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    opportunity_source: Mapped[str] = mapped_column(String, default="historical")  # historical|imported
    opportunity_ref: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="saved")
    assigned_to: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    saved_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    ngo: Mapped["NGOProfile"] = relationship(back_populates="saved_opportunities")


class SearchHistory(Base):
    __tablename__ = "search_history"

    search_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    ngo_id: Mapped[str | None] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), nullable=True)
    query_text: Mapped[str | None] = mapped_column(String, nullable=True)
    filters: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Subscriber(Base):
    """One email address on GrantSetu's notification list. Optionally linked
    to an NGOProfile (a subscriber can sign up before creating a profile --
    the landing-page email capture -- or link one later).

    `status` is the source of truth for the subscriber's lifecycle state
    (pending -> verified -> unsubscribed, or bounced/disabled from delivery
    feedback). `is_verified`/`is_unsubscribed` are kept as read-only derived
    properties (below) purely for API backward compatibility -- existing
    responses/tests that read those two booleans keep working unchanged.

    SECURITY: `verify_token` stores a SHA-256 HASH of the token, never the
    raw value -- the raw token exists only inside the verification email
    link and is never persisted (see src/email/tokens.py). It also expires
    (`verify_token_expires_at`) and is single-use (`verify_token_used_at`).
    `unsubscribe_token` is NOT hashed: unlike the verification token, it is
    a long-lived capability embedded in EVERY future alert email (not a
    one-time proof), so the raw value must remain re-derivable for as long
    as the subscriber exists -- the same pattern used by mailing-list
    unsubscribe links industry-wide. It is high-entropy
    (`secrets.token_urlsafe`, not a UUID) and is never written to logs."""

    __tablename__ = "subscribers"

    STATUS_PENDING = "pending"
    STATUS_VERIFIED = "verified"
    STATUS_UNSUBSCRIBED = "unsubscribed"
    STATUS_BOUNCED = "bounced"
    STATUS_DISABLED = "disabled"

    subscriber_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    ngo_id: Mapped[str | None] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), nullable=True)

    status: Mapped[str] = mapped_column(String, default=STATUS_PENDING, nullable=False)
    notify_new_grants: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_digest: Mapped[bool] = mapped_column(Boolean, default=False)

    verify_token: Mapped[str | None] = mapped_column(String, nullable=True, unique=True)
    verify_token_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verify_token_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_verification_email_sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    unsubscribe_token: Mapped[str] = mapped_column(String, default=_uuid, unique=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    ngo: Mapped["NGOProfile | None"] = relationship(back_populates="subscribers")

    @property
    def is_verified(self) -> bool:
        return self.status == self.STATUS_VERIFIED

    @property
    def is_unsubscribed(self) -> bool:
        return self.status == self.STATUS_UNSUBSCRIBED


class EmailLog(Base):
    """A record of every email the system attempted to send (mock or real).
    `dedup_key` is checked before sending a notification so the same
    subscriber is never notified twice about the same grant (Section 5's
    "duplicate notification prevention" requirement)."""

    __tablename__ = "email_logs"
    __table_args__ = (UniqueConstraint("dedup_key", name="uq_email_logs_dedup_key"),)

    log_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    subscriber_id: Mapped[str] = mapped_column(String, ForeignKey("subscribers.subscriber_id"), nullable=False)
    email_type: Mapped[str] = mapped_column(String, nullable=False)  # welcome | new_grant_alert | digest
    subject: Mapped[str] = mapped_column(String, nullable=False)
    dedup_key: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)  # sent_mock | sent | failed
    provider: Mapped[str] = mapped_column(String, nullable=False)  # mock | resend | ...
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GrantSource(Base):
    """Configuration for one external grant-data connector (src/connectors/).
    A row here does NOT mean live data is flowing -- `status` reflects
    whether the connector is actually configured (e.g. has an API key) and
    has been successfully synced at least once. `config` holds
    connector-specific settings (e.g. seed_urls for a web-crawl connector)
    as JSON, so new connector types don't need new columns."""

    __tablename__ = "grant_sources"

    source_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    source_url: Mapped[str | None] = mapped_column(String, nullable=True)
    connector_type: Mapped[str] = mapped_column(String, nullable=False)
    # Registry metadata (V2 source-registry upgrade): free-form, honest
    # classification of WHAT KIND of source this is and WHERE it applies --
    # used by /opportunities/live's `source_type`/`country` filters and by
    # the frontend to label cards ("Government", "CSR", "Foundation",
    # "Aggregator", "International", "Web") without hardcoding a
    # per-connector-type lookup table. Optional/nullable rather than an
    # enum: an admin adding a new source type shouldn't be blocked from
    # creating the GrantSource row until this taxonomy is extended.
    category: Mapped[str | None] = mapped_column(String, nullable=True)  # government|csr|foundation|aggregator|international|web
    country: Mapped[str | None] = mapped_column(String, nullable=True)  # India|International|Global|unknown
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String, default="not_configured")  # not_configured|configured|active|error
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_sync_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ImportedGrant(Base):
    """A grant opportunity brought in through a GrantSource connector, kept
    separate from the historical demo corpus (see module docstring). Records
    from the bundled sample/test connector are flagged `is_sample_data=True`
    and must always be shown to users as sample/test data, never as a real
    opportunity."""

    __tablename__ = "imported_grants"
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="uq_imported_grants_source_external"),
    )

    imported_grant_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    source_id: Mapped[str] = mapped_column(String, ForeignKey("grant_sources.source_id"), nullable=False)
    external_id: Mapped[str] = mapped_column(String, nullable=False)

    title: Mapped[str] = mapped_column(String, nullable=False)
    funder: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    eligibility_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    # Best-effort, never-fabricated geography signal -- "India" only when the
    # source text (or, for the AI path, the model reading that same text)
    # actually names India/an Indian state or city, or the source domain is
    # under .in / .gov.in / .org.in (see src/ai/research.py::_detect_country).
    # "unknown" (the honest default), never guessed from context alone.
    country: Mapped[str] = mapped_column(String, default="unknown")
    deadline: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    url: Mapped[str | None] = mapped_column(String, nullable=True)

    status: Mapped[str] = mapped_column(String, default="unknown")  # forecasted|open|closed|expired|unknown
    dedup_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    is_sample_data: Mapped[bool] = mapped_column(Boolean, default=False)

    # Provenance (Section 10 of the "real product" upgrade): every field
    # extracted from a live web page must be traceable back to HOW it was
    # obtained and, where practical, the exact text it came from.
    extraction_method: Mapped[str] = mapped_column(String, default="direct")  # direct|ai_inferred|rule_based
    evidence_snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_status: Mapped[str] = mapped_column(String, default="complete")  # complete|partial|failed

    date_collected: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_verified_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    # Bumped only when a re-sync actually changes a tracked field (see
    # OpportunityChange below) -- distinct from last_verified_at, which
    # advances on EVERY re-sync that re-encounters this record whether or
    # not anything changed. This is what "recently updated" in the live
    # feed should sort/filter on, not last_verified_at.
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OpportunityChange(Base):
    """One detected change to an already-imported grant, found on a later
    sync of its source (src/ingestion/sync_grants.py::sync_source). Exists
    so the live feed can honestly show "deadline changed"/"reopened"/etc.
    instead of silently overwriting a field with no record that it moved --
    see interview.txt's change-detection requirement. `change_type` is a
    short, fixed vocabulary (not free text) so the frontend can render a
    label without parsing `old_value`/`new_value`."""

    __tablename__ = "opportunity_changes"

    change_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    imported_grant_id: Mapped[str] = mapped_column(
        String, ForeignKey("imported_grants.imported_grant_id"), nullable=False, index=True
    )
    # deadline_changed|status_changed|eligibility_changed|url_changed|
    # title_changed|funder_changed|description_changed|category_changed|
    # reopened|closed
    change_type: Mapped[str] = mapped_column(String, nullable=False)
    old_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    new_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OpportunityFeedback(Base):
    """Lets an NGO hide an irrelevant opportunity or report incorrect
    information, for either a historical-corpus opportunity or an
    ImportedGrant. `opportunity_source` + `opportunity_ref` together
    identify the target, since the two corpora use different ID spaces
    (int opportunity_id vs. str imported_grant_id) and live in different
    places (in-memory Corpus vs. this database)."""

    __tablename__ = "opportunity_feedback"
    __table_args__ = (
        UniqueConstraint(
            "ngo_id", "opportunity_source", "opportunity_ref", "feedback_type",
            name="uq_opportunity_feedback_target",
        ),
    )

    feedback_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    ngo_id: Mapped[str] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), nullable=False)
    opportunity_source: Mapped[str] = mapped_column(String, nullable=False)  # historical|imported
    opportunity_ref: Mapped[str] = mapped_column(String, nullable=False)
    feedback_type: Mapped[str] = mapped_column(String, nullable=False)  # hidden|reported_incorrect
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AIProviderConfig(Base):
    """Per-NGO AI provider configuration for the research/extraction/
    explanation pipeline (src/ai/). ONE config per NGO profile. The API key
    is NEVER stored or returned in plaintext -- see src/ai/crypto.py.
    `is_enabled=False` (or no row at all) means the system uses the
    rule-based fallback path everywhere AI could otherwise help."""

    __tablename__ = "ai_provider_configs"

    ngo_id: Mapped[str] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), primary_key=True)
    provider: Mapped[str] = mapped_column(String, nullable=False)  # openai|anthropic|gemini|deepseek|groq|...
    api_key_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(String, nullable=True)
    base_url: Mapped[str | None] = mapped_column(String, nullable=True)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    last_test_status: Mapped[str | None] = mapped_column(String, nullable=True)  # ok|failed|untested
    last_test_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RefreshJob(Base):
    """One "Refresh All Sources" run, tracked so the API can return
    immediately with a job_id instead of blocking the request for the
    duration of every source's sync (which can be a real network-bound
    operation per source -- see src/ingestion/sync_grants.py). The frontend
    (and any external cron/EventBridge trigger) polls
    GET /admin/refresh-jobs/{job_id} until status is completed|failed.
    Created by POST /admin/sync-all-sources and executed by a FastAPI
    BackgroundTasks callback (api/routers/grant_sources.py::_run_refresh_job)
    or by the optional in-process scheduler (api/scheduler.py)."""

    __tablename__ = "refresh_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|completed|failed
    trigger: Mapped[str] = mapped_column(String, default="manual")  # manual|scheduled
    sources_checked: Mapped[int] = mapped_column(Integer, default=0)
    sources_total: Mapped[int] = mapped_column(Integer, default=0)
    total_fetched: Mapped[int] = mapped_column(Integer, default=0)
    total_new: Mapped[int] = mapped_column(Integer, default=0)
    total_updated: Mapped[int] = mapped_column(Integer, default=0)
    notifications_sent: Mapped[int] = mapped_column(Integer, default=0)
    per_source: Mapped[list] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class WebsiteAnalysis(Base):
    """The result of analyzing an NGO's website (Section 7 of the "real
    product" upgrade). Stored as a PROPOSAL -- never auto-applied to the
    NGOProfile. `status` tracks whether the user has reviewed it yet."""

    __tablename__ = "website_analyses"

    analysis_id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    ngo_id: Mapped[str] = mapped_column(String, ForeignKey("ngo_profiles.ngo_id"), nullable=False)
    website_url: Mapped[str] = mapped_column(String, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    detected_focus_areas: Mapped[list] = mapped_column(JSON, default=list)
    detected_locations: Mapped[list] = mapped_column(JSON, default=list)
    detected_beneficiaries: Mapped[list] = mapped_column(JSON, default=list)
    keywords: Mapped[list] = mapped_column(JSON, default=list)
    pages_analyzed: Mapped[list] = mapped_column(JSON, default=list)  # list of URLs actually fetched
    method: Mapped[str] = mapped_column(String, nullable=False)  # ai | rule_based
    status: Mapped[str] = mapped_column(String, default="pending_review")  # pending_review|applied|dismissed
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
