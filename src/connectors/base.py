"""Connector interface for external grant-data sources.

A connector's ONLY job is: given its configuration (API keys, URLs -- read
from environment variables, never hard-coded), return a list of
`RawGrantRecord`s. `src/ingestion/sync_grants.py` handles de-duplication,
persistence, and notification -- connectors do not touch the database.

`is_configured()` must return False (not raise, not fake success) whenever
required credentials/config are missing, so the sync layer can report
"not_configured" honestly instead of pretending to have synced live data.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.ai.providers import AIProvider


@dataclass
class RawGrantRecord:
    external_id: str
    title: str
    funder: str | None = None
    description: str | None = None
    eligibility_text: str | None = None
    category: str | None = None
    deadline: str | None = None  # ISO 8601 date string, or None
    url: str | None = None
    # Best-effort geography signal -- "India" | "unknown" (never any other
    # value invented without real evidence; see
    # src/ai/research.py::_detect_country). Connectors backed by a
    # single-country structured API (e.g. grants.gov) set this directly and
    # confidently; connectors that parse free text default to "unknown"
    # unless the text/domain actually names a country.
    country: str = "unknown"
    # forecasted | open | closed | expired | unknown -- connectors that can
    # distinguish these (e.g. grants.gov's oppStatus) should set this
    # explicitly rather than leaving it to be inferred from the deadline
    # alone; a FORECASTED opportunity has no application window yet even if
    # it superficially looks "open" (see GrantsGovAPIConnector).
    opportunity_status: str = "unknown"
    # Provenance (Section 10, "real product" upgrade): how this record's
    # fields were obtained, and, where practical, the source text they came
    # from. sync_grants.py copies these straight onto the ImportedGrant row.
    extraction_method: str = "direct"  # direct | ai_inferred | rule_based
    evidence_snippet: str | None = None
    extraction_status: str = "complete"  # complete | partial | failed | rejected


class ConnectorNotConfiguredError(RuntimeError):
    """Raised by fetch_records() if called on an unconfigured connector."""


class ConnectorUnavailableError(RuntimeError):
    """Raised by fetch_records() when a source's PRIMARY listing/entry point
    could not be fetched at all (blocked, unreachable, HTTP error) -- as
    opposed to a normal sync that reaches the source fine and simply finds
    nothing new. Distinct from silently returning an empty list so
    src/ingestion/sync_grants.py can report an honest "temporarily
    unavailable" (GrantSource.status="error", with the real reason) instead
    of a misleading "active, 0 fetched" that reads identically to a
    successful sync with no new opportunities this pass."""


class GrantSourceConnector(ABC):
    connector_type: str
    is_sample_data: bool = False  # True only for connectors that return fixture/test data

    def __init__(self, config: dict | None = None):
        """`config` is GrantSource.config (JSON column) -- connector-specific
        settings, e.g. {"seed_urls": [...]} for the web-discovery connector.
        Connectors that need no config (sample fixture, env-var-only real
        APIs) simply ignore it."""
        self.config = config or {}

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True only if this connector has everything it needs
        (API keys, URLs, etc.) to make a real fetch."""

    @abstractmethod
    def fetch_records(self, provider: AIProvider | None = None) -> list[RawGrantRecord]:
        """Return the current list of records from the source. Must raise
        ConnectorNotConfiguredError (not return fake data) if
        is_configured() is False.

        `provider`: an already-resolved AIProvider to use for AI-ASSISTED
        extraction, or None. Resolution (whose AI config to use, since a
        GrantSource is admin-level, not tied to one NGO) happens in
        src/ingestion/sync_grants.py, via an opt-in `ai_ngo_id` in
        GrantSource.config -- see that module. A connector whose extraction
        is entirely structured/API-based (e.g. GrantsGovAPIConnector,
        SampleFixtureConnector) has no use for this and simply ignores it;
        connectors that parse free-text HTML (WebDiscoveryConnector,
        ListingPageDiscoveryConnector) pass it to
        src/ai/research.py::extract_opportunity_fields, which itself falls
        back to rule-based extraction if `provider` is None or fails."""
