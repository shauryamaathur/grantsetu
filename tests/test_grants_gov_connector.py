"""Tests for src/connectors/grants_gov_connector.py. httpx.post is
monkeypatched to return FIXTURES CAPTURED FROM THE REAL, LIVE grants.gov
API during development (see interview.txt) -- so these tests check the
connector's field-mapping logic against genuine response shapes without
making a real network call on every test run."""
import httpx
import pytest

from src.connectors.grants_gov_connector import GrantsGovAPIConnector

# Trimmed real response from a live POST to
# https://api.grants.gov/v1/api/search2 with {"rows": 1, "oppStatuses": "posted"}
REAL_SEARCH_RESPONSE = {
    "errorcode": 0,
    "msg": "Webservice Succeeds",
    "data": {
        "hitCount": 977,
        "oppHits": [
            {
                "id": "357305",
                "number": "PAR-25-274",
                "title": "Feasibility Clinical Trials of Mind and Body Interventions",
                "agencyCode": "HHS-NIH11",
                "agency": "National Institutes of Health",
                "openDate": "11/21/2024",
                "closeDate": "11/17/2026",
                "oppStatus": "posted",
                "docType": "synopsis",
                "cfdaList": ["93.213"],
            }
        ],
    },
}

# Trimmed real response from a live POST to
# https://api.grants.gov/v1/api/fetchOpportunity with {"opportunityId": 357305}
REAL_FETCH_RESPONSE = {
    "errorcode": 0,
    "data": {
        "id": 357305,
        "opportunityNumber": "PAR-25-274",
        "opportunityTitle": "Feasibility Clinical Trials of Mind and Body Interventions",
        "synopsis": {
            "synopsisDesc": "<p>The goal of this notice of funding opportunity (NOFO) is to support "
            "feasibility trials of complementary and integrative health approaches.</p>",
            "applicantEligibilityDesc": "Other Eligible Applicants include the following: "
            "Nonprofits having a 501(c)(3) status with the IRS, other than institutions of higher "
            "education; State governments.",
        },
    },
}


class _FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code
        self.content = b"x"
        self.text = str(json_data)

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://api.grants.gov/v1/api/search2")
            raise httpx.HTTPStatusError("error", request=request, response=httpx.Response(self.status_code, request=request))


def test_is_configured_is_always_true_no_key_required():
    """Grants.gov's search2/fetchOpportunity are public -- this connector
    must never require an API key (Phase 7's explicit requirement)."""
    connector = GrantsGovAPIConnector()
    assert connector.is_configured() is True


def test_fetch_records_maps_real_search_and_detail_shapes_correctly(monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(url)
        if "search2" in url:
            return _FakeResponse(REAL_SEARCH_RESPONSE)
        if "fetchOpportunity" in url:
            return _FakeResponse(REAL_FETCH_RESPONSE)
        raise AssertionError(f"unexpected URL {url}")

    monkeypatch.setattr(httpx, "post", fake_post)

    connector = GrantsGovAPIConnector(config={"rows": 1})
    records = connector.fetch_records()

    assert len(records) == 1
    record = records[0]
    assert record.external_id == "357305"
    assert record.title == "Feasibility Clinical Trials of Mind and Body Interventions"
    assert record.funder == "National Institutes of Health"  # from "agency", NOT "agencyCode"
    assert record.deadline == "2026-11-17"  # MM/DD/YYYY -> ISO
    assert record.opportunity_status == "open"  # "posted" -> "open"
    assert record.url == "https://www.grants.gov/search-results-detail/357305"
    assert "feasibility trials" in record.description.lower()
    assert "<p>" not in record.description  # HTML stripped
    assert "501(c)(3)" in record.eligibility_text
    assert record.extraction_status == "complete"
    assert record.extraction_method == "direct"
    assert record.category is None  # docType is not a real topic -- must not be mislabeled as one


def test_forecasted_status_is_never_reported_as_open(monkeypatch):
    forecasted_response = {
        "data": {
            "oppHits": [
                {"id": "999", "title": "Future Opportunity", "agency": "Some Agency",
                 "closeDate": "", "oppStatus": "forecasted", "docType": "forecast"}
            ]
        }
    }

    def fake_post(url, json=None, timeout=None):
        return _FakeResponse(forecasted_response)

    monkeypatch.setattr(httpx, "post", fake_post)

    connector = GrantsGovAPIConnector(config={"fetch_details": False})
    records = connector.fetch_records()
    assert records[0].opportunity_status == "forecasted"
    assert records[0].opportunity_status != "open"


def test_closed_and_archived_status_mapping(monkeypatch):
    response = {
        "data": {
            "oppHits": [
                {"id": "1", "title": "Closed one", "agency": "A", "closeDate": "01/01/2020", "oppStatus": "closed"},
                {"id": "2", "title": "Archived one", "agency": "A", "closeDate": "01/01/2019", "oppStatus": "archived"},
            ]
        }
    }
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: _FakeResponse(response))

    connector = GrantsGovAPIConnector(config={"fetch_details": False})
    records = connector.fetch_records()
    assert records[0].opportunity_status == "closed"
    assert records[1].opportunity_status == "expired"  # "archived" -> "expired"


def test_missing_detail_response_marks_extraction_partial_not_failed(monkeypatch):
    """If fetchOpportunity fails/returns nothing, the record is still
    produced (never dropped) -- description/eligibility are just missing,
    and extraction_status honestly reflects that."""
    def fake_post(url, json=None, timeout=None):
        if "search2" in url:
            return _FakeResponse(REAL_SEARCH_RESPONSE)
        raise httpx.TimeoutException("simulated timeout")

    monkeypatch.setattr(httpx, "post", fake_post)

    connector = GrantsGovAPIConnector(config={"fetch_details": True})
    records = connector.fetch_records()
    assert len(records) == 1
    assert records[0].description is None
    assert records[0].extraction_status == "partial"


def test_search_http_error_propagates_and_is_handled_by_sync_layer(monkeypatch):
    """fetch_records() does not swallow a real search failure -- it's the
    sync layer's job (src/ingestion/sync_grants.py) to catch this and
    report status='error' rather than crash."""
    def fake_post(url, json=None, timeout=None):
        return _FakeResponse({}, status_code=500)

    monkeypatch.setattr(httpx, "post", fake_post)

    connector = GrantsGovAPIConnector()
    with pytest.raises(httpx.HTTPStatusError):
        connector.fetch_records()


def test_sync_source_reports_error_status_on_connector_failure(client, monkeypatch):
    """End-to-end: sync_source() must turn a real connector exception into
    a clean status='error' SyncResult, never crash the endpoint."""
    from api.db.models import GrantSource
    from api.db.session import SessionLocal
    from src.ingestion import sync_grants

    def fake_post(url, json=None, timeout=None):
        raise httpx.ConnectError("simulated network failure")

    monkeypatch.setattr(httpx, "post", fake_post)

    db = SessionLocal()
    source = GrantSource(name="Grants.gov error test", connector_type="grants_gov_api")
    db.add(source)
    db.commit()
    db.refresh(source)

    result = sync_grants.sync_source(db, source)
    assert result.status == "error"
    assert result.error is not None
    db.close()
