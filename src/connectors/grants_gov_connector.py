"""A REAL, working connector for the U.S. Grants.gov public search API.

Grants.gov's `search2` and `fetchOpportunity` endpoints are PUBLIC -- the
same ones grants.gov's own website calls from the browser -- and require NO
API key. This connector is intentionally NOT gated behind any credential
(Phase 7 of the "free-first discovery" upgrade explicitly requires this:
"Do not block this integration on an API key"). `is_configured()` always
returns True; the only way this connector fails is a real network/HTTP
problem, reported honestly via ConnectorNotConfiguredError... actually via
letting the real exception propagate to sync_grants.py's existing
try/except, which already turns any connector exception into a
status="error" sync result rather than crashing.

WHAT THIS DOES NOT SOLVE: grants.gov is a U.S. federal source. It does not
make GrantSetu's opportunity data India-specific -- see
docs/data_audit_report.md and the README's "Data sources" table. It is
included because it is a real, genuinely public, no-key API that
demonstrates the full source-registry -> connector -> sync -> ImportedGrant
pipeline working against a live, authoritative government data source,
which is valuable on its own merits regardless of geography.

ENDPOINTS USED (public, no auth, verified against the live API during
development -- see interview.txt for the actual response captured):
  POST https://api.grants.gov/v1/api/search2
    Body: {"rows": N, "oppStatuses": "forecasted|posted", ...filters}
    Returns a list of opportunity summaries (id, title, agency, dates,
    status) -- NOT the full description/eligibility text.
  POST https://api.grants.gov/v1/api/fetchOpportunity
    Body: {"opportunityId": <id>}
    Returns the full opportunity record, including synopsis (description),
    eligibility text, and award information.

STATUS MAPPING: grants.gov's `oppStatus` is mapped onto
RawGrantRecord.opportunity_status as follows -- a FORECASTED opportunity is
never presented as currently open, per Phase 7's explicit requirement:
    "forecasted" -> "forecasted"
    "posted"     -> "open"
    "closed"     -> "closed"
    "archived"   -> "expired"
    (anything else) -> "unknown"
"""
import logging

import httpx

from src.connectors.base import GrantSourceConnector, RawGrantRecord

logger = logging.getLogger("grantsetu.connectors.grants_gov")

SEARCH_URL = "https://api.grants.gov/v1/api/search2"
FETCH_URL = "https://api.grants.gov/v1/api/fetchOpportunity"
REQUEST_TIMEOUT = 20.0

STATUS_MAP = {
    "forecasted": "forecasted",
    "posted": "open",
    "closed": "closed",
    "archived": "expired",
}

DEFAULT_ROWS = 25


class GrantsGovAPIConnector(GrantSourceConnector):
    connector_type = "grants_gov_api"
    is_sample_data = False

    def is_configured(self) -> bool:
        # Deliberately always True: this is a public API with no key
        # requirement. Configuration (keyword filters, row count) is
        # optional, read from `self.config`.
        return True

    def _search(self) -> list[dict]:
        keyword = self.config.get("keyword", "")
        opp_statuses = self.config.get("opp_statuses", "forecasted|posted")
        rows = int(self.config.get("rows", DEFAULT_ROWS))

        body = {"rows": rows, "oppStatuses": opp_statuses}
        if keyword:
            body["keyword"] = keyword

        response = httpx.post(SEARCH_URL, json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        # The real API wraps results as {"errorcode":0,"data":{"oppHits":[...],"hitCount":N}}
        hits = (data.get("data") or {}).get("oppHits", [])
        return hits

    def _fetch_detail(self, opportunity_id) -> dict | None:
        try:
            response = httpx.post(
                FETCH_URL, json={"opportunityId": opportunity_id}, timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data")
        except Exception as exc:  # noqa: BLE001 -- a detail-fetch failure shouldn't drop the whole record
            logger.warning("grants.gov fetchOpportunity failed for id=%s: %s", opportunity_id, exc)
            return None

    def fetch_records(self, provider=None) -> list[RawGrantRecord]:
        # `provider` is accepted for interface compatibility (see
        # GrantSourceConnector.fetch_records) but unused -- grants.gov's API
        # already returns structured fields; there is nothing free-text to
        # extract with AI here.
        hits = self._search()
        fetch_details = self.config.get("fetch_details", True)

        records = []
        for hit in hits:
            opp_id = hit.get("id")
            if opp_id is None:
                continue

            raw_status = (hit.get("oppStatus") or "").lower()
            mapped_status = STATUS_MAP.get(raw_status, "unknown")

            title = hit.get("title") or f"Opportunity {opp_id}"
            # Verified against the live API (2026-09): the search2 hit shape
            # has "agency" (full name, e.g. "National Institutes of Health")
            # and "agencyCode" (e.g. "HHS-NIH11") -- NOT "agencyName".
            agency = hit.get("agency") or hit.get("agencyCode")
            close_date = hit.get("closeDate")  # e.g. "11/17/2026" (MM/DD/YYYY) per the live API
            deadline_iso = _mmddyyyy_to_iso(close_date)

            description = None
            eligibility_text = None
            url = f"https://www.grants.gov/search-results-detail/{opp_id}"

            if fetch_details:
                detail = self._fetch_detail(opp_id)
                synopsis = detail.get("synopsis") if detail and isinstance(detail.get("synopsis"), dict) else None
                if synopsis:
                    description = _strip_html(synopsis.get("synopsisDesc"))
                    # Verified against the live API: the field is
                    # "applicantEligibilityDesc", not "eligibilityDesc".
                    eligibility_text = _strip_html(synopsis.get("applicantEligibilityDesc"))

            records.append(
                RawGrantRecord(
                    external_id=str(opp_id),
                    title=title,
                    funder=agency,
                    description=description,
                    eligibility_text=eligibility_text,
                    # NOTE: grants.gov's search2 hit does not include a real
                    # subject/topic category -- "docType" (e.g. "synopsis")
                    # is a document-type marker, not a topic, so it is
                    # deliberately NOT used here (using it would mislabel
                    # every record with the same meaningless "category").
                    category=None,
                    deadline=deadline_iso,
                    url=url,
                    # grants.gov is exclusively U.S. federal opportunities --
                    # a confident, direct fact from the source itself, not a
                    # guess (unlike the free-text connectors' _detect_country).
                    country="United States",
                    opportunity_status=mapped_status,
                    extraction_method="direct",
                    evidence_snippet=None,
                    extraction_status="complete" if description or eligibility_text else "partial",
                )
            )
        return records


def _mmddyyyy_to_iso(value: str | None) -> str | None:
    if not value:
        return None
    parts = value.strip().split("/")
    if len(parts) != 3:
        return None
    month, day, year = parts
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return None


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    import re

    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None
