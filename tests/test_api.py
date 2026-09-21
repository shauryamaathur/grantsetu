"""API-level tests for the original MVP endpoints (NGO profiles, search,
recommendations, saved opportunities). Shared fixtures (`client`,
`patch_corpus`) live in tests/conftest.py so other test files can reuse them."""
import pytest

from api.db.models import GrantSource, ImportedGrant
from api.db.session import SessionLocal


@pytest.fixture
def real_live_grant():
    """Seeds one non-sample ImportedGrant -- GET /opportunities/recommended
    only ever ranks LIVE, non-sample opportunities (see
    src/ranking/live_match.py), so any test exercising it needs a real
    ImportedGrant row, not the historical corpus fixture."""
    db = SessionLocal()
    source = GrantSource(name="Test Recommended Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    grant = ImportedGrant(
        source_id=source.source_id,
        external_id="rec-1",
        title="Environmental Youth Restoration Grant",
        funder="Green Futures Fund",
        description="Supports youth-led environmental restoration projects.",
        category="Environment",
        status="open",
        dedup_hash="rec-test-hash-1",
        is_sample_data=False,
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    grant_id = grant.imported_grant_id
    db.close()

    yield grant_id

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["opportunities_indexed"] == 3


def test_create_ngo_profile_requires_focus_areas(client):
    res = client.post("/ngo-profile", json={"name": "Test NGO", "focus_areas": []})
    assert res.status_code == 422


def test_create_and_fetch_ngo_profile(client):
    res = client.post("/ngo-profile", json={
        "name": "Green Earth Trust",
        "registration_type": "trust",
        "focus_areas": ["environment"],
        "locations": ["Maharashtra"],
        "beneficiaries": ["youth"],
    })
    assert res.status_code == 201
    ngo_id = res.json()["ngo_id"]

    res2 = client.get(f"/ngo-profile/{ngo_id}")
    assert res2.status_code == 200
    assert res2.json()["name"] == "Green Earth Trust"


def test_get_ngo_profile_not_found(client):
    res = client.get("/ngo-profile/does-not-exist")
    assert res.status_code == 404


def test_recommended_feed_returns_relevant_live_match(client, real_live_grant):
    """Recommended must rank a REAL live (ImportedGrant) opportunity that
    textually matches the NGO's profile -- never the historical corpus
    (see api/routers/opportunities.py::recommended_opportunities)."""
    profile_res = client.post("/ngo-profile", json={
        "name": "Green Earth Trust",
        "registration_type": "trust",
        "focus_areas": ["environment"],
        "locations": ["Maharashtra"],
        "beneficiaries": ["youth"],
    })
    ngo_id = profile_res.json()["ngo_id"]

    res = client.get(f"/opportunities/recommended?ngo_id={ngo_id}")
    assert res.status_code == 200
    results = res.json()
    assert len(results) >= 1
    assert results[0]["opportunity"]["imported_grant_id"] == real_live_grant
    assert results[0]["score"] > 0


def test_recommended_feed_excludes_sample_data(client):
    """A profile whose focus areas only match sample/fixture data (not
    seeded here at all, matching production) gets an honest empty result,
    never a sample-data substitute."""
    profile_res = client.post("/ngo-profile", json={
        "name": "Any NGO", "focus_areas": ["health"], "beneficiaries": ["children"],
    })
    ngo_id = profile_res.json()["ngo_id"]
    res = client.get(f"/opportunities/recommended?ngo_id={ngo_id}")
    assert res.status_code == 200
    assert res.json() == []


def test_recommended_feed_excludes_closed_and_expired_grants(client):
    from api.db.models import GrantSource, ImportedGrant
    from api.db.session import SessionLocal

    db = SessionLocal()
    source = GrantSource(name="Closed Grant Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    closed = ImportedGrant(
        source_id=source.source_id, external_id="closed-1", title="Closed Environment Grant",
        category="Environment", status="closed", dedup_hash="closed-hash-1", is_sample_data=False,
    )
    expired = ImportedGrant(
        source_id=source.source_id, external_id="expired-1", title="Expired Environment Grant",
        category="Environment", status="expired", dedup_hash="expired-hash-1", is_sample_data=False,
    )
    db.add_all([closed, expired])
    db.commit()
    db.close()

    profile_res = client.post("/ngo-profile", json={"name": "Env NGO", "focus_areas": ["environment"]})
    ngo_id = profile_res.json()["ngo_id"]
    res = client.get(f"/opportunities/recommended?ngo_id={ngo_id}")
    assert res.status_code == 200
    assert res.json() == []


def test_recommended_feed_unknown_ngo_returns_404(client):
    res = client.get("/opportunities/recommended?ngo_id=nonexistent")
    assert res.status_code == 404


def test_search_returns_matching_opportunity(client):
    res = client.get("/opportunities/search?q=environmental youth")
    assert res.status_code == 200
    titles = [r["opportunity_title"] for r in res.json()]
    assert "Environmental Youth Grant" in titles


def test_opportunity_detail_not_found(client):
    res = client.get("/opportunities/999999")
    assert res.status_code == 404


def test_opportunity_detail_missing_award_shows_not_specified_semantics(client):
    res = client.get("/opportunities/101")
    assert res.status_code == 200
    assert res.json()["award_ceiling"] == 50000.0


def test_save_and_list_and_update_saved_opportunity(client):
    profile_res = client.post("/ngo-profile", json={
        "name": "Green Earth Trust",
        "focus_areas": ["environment"],
    })
    ngo_id = profile_res.json()["ngo_id"]

    save_res = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 101})
    assert save_res.status_code == 201
    saved_id = save_res.json()["saved_id"]

    list_res = client.get(f"/saved-opportunities?ngo_id={ngo_id}")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 1

    update_res = client.patch(f"/saved-opportunities/{saved_id}", json={"status": "planning_to_apply"})
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "planning_to_apply"

    delete_res = client.delete(f"/saved-opportunities/{saved_id}")
    assert delete_res.status_code == 204

    list_after_delete = client.get(f"/saved-opportunities?ngo_id={ngo_id}")
    assert list_after_delete.json() == []


def test_remove_unknown_saved_opportunity_404(client):
    res = client.delete("/saved-opportunities/does-not-exist")
    assert res.status_code == 404


def test_save_duplicate_opportunity_conflicts(client):
    profile_res = client.post("/ngo-profile", json={"name": "Test NGO", "focus_areas": ["health"]})
    ngo_id = profile_res.json()["ngo_id"]

    client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 102})
    res2 = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 102})
    assert res2.status_code == 409


def test_save_opportunity_for_unknown_ngo_returns_404(client):
    res = client.post("/saved-opportunities", json={"ngo_id": "no-such-ngo", "opportunity_id": 101})
    assert res.status_code == 404


def test_update_saved_opportunity_invalid_status_rejected(client):
    profile_res = client.post("/ngo-profile", json={"name": "Test NGO", "focus_areas": ["health"]})
    ngo_id = profile_res.json()["ngo_id"]
    save_res = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 102})
    saved_id = save_res.json()["saved_id"]

    res = client.patch(f"/saved-opportunities/{saved_id}", json={"status": "not-a-real-status"})
    assert res.status_code == 422
