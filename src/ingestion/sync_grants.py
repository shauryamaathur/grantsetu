"""Syncs a GrantSource by running its connector, de-duplicating results
against the imported_grants table, and inserting genuinely new records.

Deduplication has two layers:
  1. (source_id, external_id) uniqueness -- the same source re-reporting the
     same record on a later sync is never re-inserted (checked first, cheap).
  2. `dedup_hash` (title + funder + deadline, normalized) -- catches the same
     underlying grant being listed by TWO DIFFERENT sources, which
     (source_id, external_id) uniqueness alone would not catch.

"New-grant detection" is simply: whatever survives both checks in one sync
run is new, and is returned so the caller can notify subscribers about it.

Change detection: a record matched by (source_id, external_id) on a LATER
sync is not just re-verified (last_verified_at bump) -- _apply_changes()
compares every mutable field against what's stored and, for anything that
actually differs, updates the row AND writes an OpportunityChange audit row
(api/db/models.py) so the feed can honestly show "deadline changed"/
"reopened"/etc. rather than silently mutating data with no record it moved.
A field is only ever updated when the fresh value is non-empty and
different -- a partial re-fetch that extracted LESS than before never
erases previously-known good data.

AI enrichment (opt-in): AIProviderConfig is per-NGO (api/db/models.py), but a
GrantSource is admin-level and isn't owned by any one NGO -- so there is no
automatic "the" AI key to use for a source's extraction. An operator opts a
source into AI-assisted extraction by setting
`GrantSource.config["ai_ngo_id"]` to an NGOProfile.ngo_id that has AI enabled
(same config an NGO sets up for AI search, via api/routers/ai_config.py) --
that NGO's provider is then used ONLY to improve extraction quality for this
source's free-text pages (src/ai/research.py::extract_opportunity_fields),
never to decide what counts as a grant or to fabricate fields. Without
`ai_ngo_id` (the default), or if the referenced NGO has no working AI config,
sync falls back to the always-available rule-based extraction exactly as
before -- a source's sync never fails or degrades because of this.
"""
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from api.db.models import GrantSource, ImportedGrant, OpportunityChange
from src.ai.factory import get_provider_for_ngo
from src.connectors.base import ConnectorNotConfiguredError, RawGrantRecord
from src.connectors.registry import get_connector

logger = logging.getLogger("grantsetu.sync")


@dataclass
class SyncResult:
    status: str  # not_configured | active | error
    fetched: int = 0
    new: int = 0
    duplicates: int = 0
    already_imported: int = 0
    # Of `already_imported`, how many actually had a tracked field change
    # detected (see _apply_changes) -- distinct from already_imported
    # itself, which also counts re-synced records that came back identical.
    updated: int = 0
    error: str | None = None
    new_grants: list[ImportedGrant] = field(default_factory=list)


def compute_dedup_hash(title: str, funder: str | None, deadline: str | None) -> str:
    normalized = f"{(title or '').strip().lower()}|{(funder or '').strip().lower()}|{(deadline or '').strip()}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _parse_deadline(deadline: str | None):
    if not deadline:
        return None
    try:
        return datetime.fromisoformat(deadline)
    except ValueError:
        logger.warning("Could not parse deadline %r; storing as None", deadline)
        return None


def _resolve_status(opportunity_status: str, deadline, now: datetime) -> str:
    """Connectors that KNOW the real status (e.g. grants.gov's oppStatus)
    set `opportunity_status` explicitly -- trust it (a source-confirmed
    "forecasted" must never be silently overridden into "open" just because
    it happens to have a future date field). Connectors that don't know
    (opportunity_status left at the RawGrantRecord default, "unknown") get
    one safe, non-inventive downgrade: if we have a real deadline and it has
    already passed, mark "expired" rather than leaving a plainly-past
    opportunity looking active (Phase 10: "If a deadline has passed, do not
    continue presenting the opportunity as active"). Otherwise "unknown"
    stays "unknown" -- never upgraded to "open" without the source saying so."""
    if opportunity_status and opportunity_status != "unknown":
        return opportunity_status
    if deadline is not None and deadline < now:
        return "expired"
    return "unknown"


# (ImportedGrant attribute, human change_type) for every mutable field a
# re-sync can update. Deliberately excludes source_id/external_id/dedup_hash
# (identity, never changes for the same row) and is_sample_data/
# extraction_method/extraction_status/evidence_snippet (provenance of HOW
# the CURRENT values were obtained, tracked but not itself "a change" worth
# surfacing to a user the way a moved deadline is).
_TRACKED_FIELDS = [
    ("title", "title_changed"),
    ("funder", "funder_changed"),
    ("description", "description_changed"),
    ("eligibility_text", "eligibility_changed"),
    ("category", "category_changed"),
    ("url", "url_changed"),
    ("country", "country_changed"),
]


def _status_change_type(old_status: str, new_status: str) -> str:
    """"reopened"/"closed" whenever a status genuinely crosses the
    closed-like <-> not-closed-like boundary -- including into "unknown"
    (a source no longer confirming a status either way is not the same as
    a confirmed close, and a previously-expired grant that's no longer past
    its deadline on re-sync is meaningfully "reopened" even if the source
    never explicitly says "open"). Anything else (e.g. open -> forecasted)
    is a plain status_changed."""
    closed_like = {"closed", "expired"}
    if old_status in closed_like and new_status not in closed_like:
        return "reopened"
    if old_status not in closed_like and new_status in closed_like:
        return "closed"
    return "status_changed"


def _apply_changes(db: Session, grant: ImportedGrant, rec: RawGrantRecord, parsed_deadline, new_status: str, now: datetime) -> bool:
    """Compares `rec`'s fresh values against the already-persisted `grant`
    row, updates whatever genuinely changed, and logs one OpportunityChange
    row per changed field. Never overwrites an existing non-empty value with
    an empty one -- a later fetch that happened to extract LESS than a
    previous one (e.g. a partial re-fetch) must not erase previously-known
    good data; it just leaves that field alone. Returns True if anything
    changed (title/description/etc OR status OR deadline), so the caller can
    count it as `updated` and bump `updated_at`."""
    changed = False

    for attr, change_type in _TRACKED_FIELDS:
        new_value = getattr(rec, attr, None)
        old_value = getattr(grant, attr)
        if new_value and new_value != old_value:
            db.add(OpportunityChange(
                imported_grant_id=grant.imported_grant_id,
                change_type=change_type,
                old_value=old_value,
                new_value=new_value,
            ))
            setattr(grant, attr, new_value)
            changed = True

    if parsed_deadline is not None and parsed_deadline != grant.deadline:
        db.add(OpportunityChange(
            imported_grant_id=grant.imported_grant_id,
            change_type="deadline_changed",
            old_value=grant.deadline.isoformat() if grant.deadline else None,
            new_value=parsed_deadline.isoformat(),
        ))
        grant.deadline = parsed_deadline
        changed = True

    if new_status != grant.status:
        db.add(OpportunityChange(
            imported_grant_id=grant.imported_grant_id,
            change_type=_status_change_type(grant.status, new_status),
            old_value=grant.status,
            new_value=new_status,
        ))
        grant.status = new_status
        changed = True

    if changed:
        grant.dedup_hash = compute_dedup_hash(grant.title, grant.funder, grant.deadline.isoformat() if grant.deadline else None)
        grant.updated_at = now

    return changed


def sync_source(db: Session, source: GrantSource) -> SyncResult:
    connector = get_connector(source.connector_type, source.config)

    if not connector.is_configured():
        source.status = "not_configured"
        db.commit()
        return SyncResult(status="not_configured")

    provider = None
    ai_ngo_id = source.config.get("ai_ngo_id") if source.config else None
    if ai_ngo_id:
        # get_provider_for_ngo already returns None (never raises) for a
        # missing NGO, a missing/disabled AI config, or an undecryptable
        # key -- so an operator misconfiguring or later deleting the
        # referenced NGO just silently reverts this source to rule-based
        # extraction rather than breaking the sync.
        provider = get_provider_for_ngo(db, ai_ngo_id)

    try:
        records: list[RawGrantRecord] = connector.fetch_records(provider=provider)
    except ConnectorNotConfiguredError as exc:
        source.status = "not_configured"
        db.commit()
        return SyncResult(status="not_configured", error=str(exc))
    except Exception as exc:  # noqa: BLE001 -- a connector failure must not crash the sync endpoint
        logger.error("Connector %s failed: %s", source.connector_type, exc)
        source.status = "error"
        db.commit()
        return SyncResult(status="error", error=str(exc))

    result = SyncResult(status="active", fetched=len(records))
    now = datetime.utcnow()

    for rec in records:
        already = (
            db.query(ImportedGrant)
            .filter_by(source_id=source.source_id, external_id=rec.external_id)
            .first()
        )
        if already:
            already.last_verified_at = now
            parsed_deadline = _parse_deadline(rec.deadline)
            new_status = _resolve_status(rec.opportunity_status, parsed_deadline, now)
            if _apply_changes(db, already, rec, parsed_deadline, new_status, now):
                result.updated += 1
            result.already_imported += 1
            continue

        dedup_hash = compute_dedup_hash(rec.title, rec.funder, rec.deadline)
        cross_source_duplicate = db.query(ImportedGrant).filter_by(dedup_hash=dedup_hash).first()
        if cross_source_duplicate:
            result.duplicates += 1
            continue

        parsed_deadline = _parse_deadline(rec.deadline)
        grant = ImportedGrant(
            source_id=source.source_id,
            external_id=rec.external_id,
            title=rec.title,
            funder=rec.funder,
            description=rec.description,
            eligibility_text=rec.eligibility_text,
            category=rec.category,
            deadline=parsed_deadline,
            url=rec.url,
            country=rec.country,
            status=_resolve_status(rec.opportunity_status, parsed_deadline, now),
            dedup_hash=dedup_hash,
            is_sample_data=connector.is_sample_data,
            extraction_method=rec.extraction_method,
            evidence_snippet=rec.evidence_snippet,
            extraction_status=rec.extraction_status,
            date_collected=now,
            last_verified_at=now,
            updated_at=now,
        )
        db.add(grant)
        result.new += 1
        result.new_grants.append(grant)

    source.status = "active"
    source.last_synced_at = now
    source.last_sync_summary = (
        f"fetched={result.fetched} new={result.new} updated={result.updated} "
        f"duplicates={result.duplicates} already_imported={result.already_imported}"
    )
    db.commit()

    for grant in result.new_grants:
        db.refresh(grant)

    return result
