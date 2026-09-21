"""End-to-end tests for POST /opportunities/ai-search."""
from datetime import datetime, timedelta

import pytest

from api.db.models import GrantSource, ImportedGrant, SearchHistory
from api.db.session import SessionLocal


@pytest.fixture
def live_grant(client):
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    source = GrantSource(name="AI Search Endpoint Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    grant = ImportedGrant(
        source_id=source.source_id, external_id="endpoint-1", title="India Education Access Grant",
        funder="Test Foundation", category="Education",
        description="Supports education programs in India for youth.",
        eligibility_text="Registered nonprofits in India.",
        status="open", dedup_hash="endpoint-hash-1", is_sample_data=False,
        deadline=datetime.utcnow() + timedelta(days=10),
    )
    db.add(grant)
    db.commit()
    grant_id = grant.imported_grant_id
    db.close()

    yield grant_id

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.query(SearchHistory).delete()
    db.commit()
    db.close()


def test_ai_search_without_ngo_or_ai_still_returns_real_ranked_results(client, live_grant):
    res = client.post("/opportunities/ai-search", json={"query": "education grants in India"})
    assert res.status_code == 200
    body = res.json()
    assert body["intent"]["method"] == "rule_based"
    assert body["ai_available"] is False
    assert any(r["ref"] == live_grant for r in body["results"])
    matched = next(r for r in body["results"] if r["ref"] == live_grant)
    assert matched["explanation_method"] == "rule_based"
    assert matched["ai_disclaimer"] is None
    assert matched["match_tier"] in {"Strong match", "Possible match", "Weak match", "Eligibility unclear"}


def test_ai_search_irrelevant_query_returns_empty_results_not_error(client, live_grant):
    res = client.post("/opportunities/ai-search", json={"query": "wildlife conservation marine biology"})
    assert res.status_code == 200
    assert res.json()["results"] == []


def test_ai_search_writes_search_history_row(client, live_grant):
    db = SessionLocal()
    before = db.query(SearchHistory).count()
    db.close()

    client.post("/opportunities/ai-search", json={"query": "education grants in India"})

    db = SessionLocal()
    after = db.query(SearchHistory).count()
    row = db.query(SearchHistory).order_by(SearchHistory.created_at.desc()).first()
    db.close()

    assert after == before + 1
    assert row.query_text == "education grants in India"
    assert row.filters["method"] == "rule_based"


def test_ai_search_unknown_ngo_id_returns_404(client):
    res = client.post("/opportunities/ai-search", json={"query": "education grants", "ngo_id": "does-not-exist"})
    assert res.status_code == 404


def test_ai_search_empty_query_rejected(client):
    res = client.post("/opportunities/ai-search", json={"query": ""})
    assert res.status_code == 422


def test_ai_search_deadline_only_query_returns_filter_matched_results(client, live_grant):
    res = client.post("/opportunities/ai-search", json={"query": "grants with deadlines in the next 30 days"})
    assert res.status_code == 200
    body = res.json()
    assert body["intent"]["deadline_within_days"] == 30
    assert any(r["ref"] == live_grant for r in body["results"])
    matched = next(r for r in body["results"] if r["ref"] == live_grant)
    # A filter-only match must never be labeled "Strong match" -- no real
    # relevance score backs that claim.
    assert matched["match_tier"] != "Strong match"


def test_ai_search_never_returns_sample_data(client, live_grant):
    db = SessionLocal()
    src = db.query(GrantSource).first()
    db.add(ImportedGrant(
        source_id=src.source_id, external_id="sample-endpoint-1", title="Education Grant Sample",
        category="Education", status="open", dedup_hash="sample-endpoint-hash", is_sample_data=True,
    ))
    db.commit()
    db.close()

    res = client.post("/opportunities/ai-search", json={"query": "education grants"})
    assert res.status_code == 200
    assert all(r["title"] != "Education Grant Sample" for r in res.json()["results"])


def test_ai_search_summary_counts_are_real(client):
    """The contextual "N found, M strong matches, K closing soon, J open to
    India" summary must be computed from the actual result set -- never a
    fabricated/estimated figure."""
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    source = GrantSource(name="Summary Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    db.add(ImportedGrant(
        source_id=source.source_id, external_id="summary-1", title="India Health Access Grant",
        funder="Test Fund", category="Health", country="India",
        description="Supports health access programs for underserved communities in India.",
        eligibility_text="Registered nonprofits.",
        status="open", dedup_hash="summary-hash-1", is_sample_data=False,
        deadline=datetime.utcnow() + timedelta(days=10),
    ))
    db.commit()
    db.close()

    res = client.post("/opportunities/ai-search", json={"query": "health access grants for underserved communities"})
    assert res.status_code == 200
    body = res.json()
    assert body["summary"]["total"] == len(body["results"])
    assert body["summary"]["open_to_india"] >= 1
    assert body["summary"]["closing_within_30_days"] >= 1
    for r in body["results"]:
        if r["ref"] and r["match_tier"] == "Strong match":
            assert r["match_score_100"] is not None
            assert 0 <= r["match_score_100"] <= 100


def test_ai_search_filter_only_match_has_no_numeric_score(client, live_grant):
    res = client.post("/opportunities/ai-search", json={"query": "grants with deadlines in the next 30 days"})
    body = res.json()
    matched = next(r for r in body["results"] if r["ref"] == live_grant)
    assert matched["match_score_100"] is None


def test_ai_search_historical_query_returns_no_results_and_a_redirect_notice(client, live_grant):
    """Non-negotiable product rule: Ask GrantSetu never returns historical
    opportunities, EVEN when the query explicitly asks for them. A query
    recognized as historical-scope gets an empty result set and a
    scope_notice pointing at Grant Intelligence, not a historical-database
    search -- see src/ranking/ai_search.py's module docstring."""
    for query in [
        "Show me historical grants for education",
        "Show me past U.S. government grants",
        "What grants did this organization receive previously?",
    ]:
        res = client.post("/opportunities/ai-search", json={"query": query})
        assert res.status_code == 200
        body = res.json()
        assert body["intent"]["search_scope"] == "historical", query
        assert body["results"] == [], query
        assert body["summary"]["total"] == 0
        assert body["scope_notice"] is not None
        assert "grant intelligence" in body["scope_notice"].lower()


def test_ai_search_never_returns_a_historical_source_result(client, live_grant):
    """Even for a query that ISN'T historical-scope, no result may ever
    have source="historical" -- src/ranking/ai_search.py has no code path
    that reaches the historical corpus at all."""
    res = client.post("/opportunities/ai-search", json={"query": "education grants in India"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) > 0
    assert all(r["source"] != "historical" for r in body["results"])
    assert all(r["source_label"] != "Historical opportunity" for r in body["results"])


def test_ai_search_does_not_pad_thin_current_results_with_historical(client, live_grant):
    """If only a handful of genuinely current opportunities match, the
    summary must report exactly that count -- never backfilled with
    historical records to make the number look bigger."""
    res = client.post("/opportunities/ai-search", json={"query": "education grants in India"})
    body = res.json()
    assert body["summary"]["total"] == len(body["results"])
    # No historical_count field at all any more -- it would always be zero
    # by construction, so it was removed rather than kept as dead weight.
    assert "historical_count" not in body["summary"]


def test_ai_search_score_matches_recommended_score_for_same_ngo_and_grant(client, live_grant):
    """The exact bug this test guards against: Ask GrantSetu used to score
    a live opportunity via its own ad-hoc TF-IDF-cosine formula
    (src/ranking/ai_search.py), while GET /opportunities/recommended and
    the detail page scored the SAME grant+NGO pair via the canonical
    src/ranking/live_match.py pipeline -- two different formulas, so the
    same opportunity could show e.g. 51/100 in a search result and a
    different number in Details. When an ngo_id is given, both endpoints
    must now agree exactly."""
    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Education NGO", "registration_type": "trust", "focus_areas": ["education"], "locations": ["India"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]

    ai_res = client.post("/opportunities/ai-search", json={"query": "education grants in India", "ngo_id": ngo_id})
    assert ai_res.status_code == 200
    ai_match = next(r for r in ai_res.json()["results"] if r["ref"] == live_grant)

    recommended = client.get("/opportunities/recommended", params={"ngo_id": ngo_id}).json()
    recommended_match = next(r for r in recommended if r["opportunity"]["imported_grant_id"] == live_grant)

    assert ai_match["match_score_100"] == recommended_match["match_score"]["score_100"]
    assert ai_match["match_tier_id"] == recommended_match["match_score"]["tier"]

    detail = client.get(f"/opportunities/live/{live_grant}", params={"ngo_id": ngo_id}).json()
    assert detail["match_score"]["score_100"] == ai_match["match_score_100"]
    assert detail["match_score"]["tier"] == ai_match["match_tier_id"]
