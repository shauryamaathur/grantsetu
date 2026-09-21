"""Coverage for GET /opportunities/live's search/filter/sort parameters,
added alongside the Live Grant Discovery frontend upgrade."""
import pytest

from api.db.models import GrantSource, ImportedGrant
from api.db.session import SessionLocal


@pytest.fixture
def two_imported_grants():
    # All tests in the pytest session share one SQLite file (see
    # conftest.py), and ImportedGrant rows from other test files (e.g.
    # test_grant_sync.py's sample-connector import) can still be present
    # depending on test order -- wipe first so query-text/category/status
    # filter assertions below only ever see rows this fixture created.
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()

    source = GrantSource(name="Test Live Filter Source", connector_type="sample_fixture", category="aggregator")
    db.add(source)
    db.commit()
    db.refresh(source)

    g1 = ImportedGrant(
        source_id=source.source_id, external_id="e1", title="Coastal Health Grant",
        funder="Ocean Trust", category="Health", status="open", country="India",
        dedup_hash="h1", is_sample_data=False,
    )
    g2 = ImportedGrant(
        source_id=source.source_id, external_id="e2", title="Rural Education Fund",
        funder="Education Board", category="Education", status="closed", country="unknown",
        dedup_hash="h2", is_sample_data=False,
    )
    db.add_all([g1, g2])
    db.commit()
    db.close()

    yield

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_filter_by_query_text(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"q": "coastal"})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Coastal Health Grant"}


def test_filter_by_category(client, two_imported_grants):
    # active_only=false: this fixture's "Rural Education Fund" is status
    # "closed", and the default feed (active_only=True) now excludes
    # closed/expired rows -- see test_default_feed_excludes_closed_by_default.
    # This test is about category filtering, not status, so it opts out of
    # that default explicitly.
    res = client.get("/opportunities/live", params={"category": "Education", "active_only": False})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Rural Education Fund"}


def test_default_feed_excludes_closed_by_default(client, two_imported_grants):
    res = client.get("/opportunities/live")
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Coastal Health Grant"}  # closed "Rural Education Fund" excluded


def test_explicit_status_filter_overrides_active_only_default(client, two_imported_grants):
    """An explicit status=closed request must still work even though
    active_only defaults to True -- the default is a convenience for the
    unfiltered view, not a hard restriction overriding an explicit ask."""
    res = client.get("/opportunities/live", params={"status": "closed"})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Rural Education Fund"}


def test_active_only_false_shows_everything(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"active_only": False})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Coastal Health Grant", "Rural Education Fund"}


def test_filter_by_country(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"country": "India"})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Coastal Health Grant"}


def test_filter_by_source_type(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"source_type": "aggregator", "active_only": False})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Coastal Health Grant", "Rural Education Fund"}

    res2 = client.get("/opportunities/live", params={"source_type": "government"})
    assert res2.json() == []


def test_summary_includes_source_name_and_ai_enriched_flag(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"active_only": False})
    assert res.status_code == 200
    by_title = {g["title"]: g for g in res.json()}
    assert by_title["Coastal Health Grant"]["source_name"] == "Test Live Filter Source"
    assert by_title["Coastal Health Grant"]["source_category"] == "aggregator"
    assert by_title["Coastal Health Grant"]["ai_enriched"] is False
    assert by_title["Coastal Health Grant"]["country"] == "India"


def test_opportunity_changes_endpoint_empty_for_never_changed_grant(client, two_imported_grants):
    db = SessionLocal()
    grant = db.query(ImportedGrant).filter_by(external_id="e1").first()
    grant_id = grant.imported_grant_id
    db.close()

    res = client.get(f"/opportunities/live/{grant_id}/changes")
    assert res.status_code == 200
    assert res.json() == []


def test_opportunity_changes_endpoint_404_for_unknown_grant(client, two_imported_grants):
    res = client.get("/opportunities/live/does-not-exist/changes")
    assert res.status_code == 404


def test_live_opportunity_detail_endpoint(client, two_imported_grants):
    db = SessionLocal()
    grant = db.query(ImportedGrant).filter_by(external_id="e1").first()
    grant_id = grant.imported_grant_id
    db.close()

    res = client.get(f"/opportunities/live/{grant_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["title"] == "Coastal Health Grant"
    assert body["source_name"] == "Test Live Filter Source"
    assert body["match_score"] is None  # no ngo_id given
    assert body["recent_changes"] == []


def test_live_opportunity_detail_includes_match_score_for_ngo(client, two_imported_grants):
    db = SessionLocal()
    grant = db.query(ImportedGrant).filter_by(external_id="e1").first()
    grant_id = grant.imported_grant_id
    db.close()

    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Coastal NGO", "registration_type": "trust", "focus_areas": ["health"], "locations": ["India"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]

    res = client.get(f"/opportunities/live/{grant_id}", params={"ngo_id": ngo_id})
    assert res.status_code == 200
    body = res.json()
    assert body["match_score"] is not None
    assert 0 <= body["match_score"]["score_100"] <= 100
    assert body["match_score"]["tier"] in {
        "minimal", "low", "moderate", "good", "strong", "very_strong", "exceptional",
    }
    assert len(body["match_score"]["components"]) > 0


def test_detail_match_score_is_identical_to_recommended_list_score(client):
    """Regression test for a real inconsistency bug: GET /opportunities/live/
    {id}?ngo_id=... used to rank the grant against ONLY itself
    (rank_live_grants([grant], ...)), while GET /opportunities/recommended
    ranks the same grant against the full candidate pool -- and TF-IDF
    text_relevance depends on the whole candidate set for IDF weighting, so
    the two produced DIFFERENT scores for the exact same grant+NGO pair
    (e.g. a search result showing 51/100 and the detail page showing a
    different number for the identical opportunity). This test uses three
    candidates specifically so a single-grant TF-IDF fit would score
    differently than the real multi-candidate fit, catching a regression a
    2-grant fixture could miss."""
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()

    source = GrantSource(name="Consistency Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    db.add_all([
        ImportedGrant(
            source_id=source.source_id, external_id="c1", title="Rural Health Access Grant",
            funder="Ocean Trust", category="Health",
            description="Supports rural health and healthcare access programs in India.",
            eligibility_text="Registered nonprofits.",
            status="open", country="India", dedup_hash="ch1", is_sample_data=False,
        ),
        ImportedGrant(
            source_id=source.source_id, external_id="c2", title="Urban Infrastructure Fund",
            funder="City Trust", category="Infrastructure",
            description="Funds road and infrastructure construction projects.",
            status="open", country="India", dedup_hash="ch2", is_sample_data=False,
        ),
        ImportedGrant(
            source_id=source.source_id, external_id="c3", title="Youth Sports Program",
            funder="Sports Council", category="Sports",
            description="Supports youth sports and recreation programs.",
            status="open", country="India", dedup_hash="ch3", is_sample_data=False,
        ),
    ])
    db.commit()
    target = db.query(ImportedGrant).filter_by(external_id="c1").first()
    grant_id = target.imported_grant_id
    db.close()

    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Health NGO", "registration_type": "trust", "focus_areas": ["health", "rural communities"], "locations": ["India"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]

    recommended = client.get("/opportunities/recommended", params={"ngo_id": ngo_id}).json()
    recommended_match = next(r for r in recommended if r["opportunity"]["imported_grant_id"] == grant_id)

    detail = client.get(f"/opportunities/live/{grant_id}", params={"ngo_id": ngo_id}).json()

    assert detail["match_score"]["score_100"] == recommended_match["match_score"]["score_100"]
    assert detail["match_score"]["tier"] == recommended_match["match_score"]["tier"]

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_live_opportunity_detail_404_for_unknown_grant(client):
    res = client.get("/opportunities/live/does-not-exist")
    assert res.status_code == 404


def test_live_opportunity_detail_404_for_sample_data(client):
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    source = GrantSource(name="Sample Detail Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    grant = ImportedGrant(
        source_id=source.source_id, external_id="sample-detail-1", title="[SAMPLE] Detail Test",
        status="open", dedup_hash="sample-detail-hash", is_sample_data=True,
    )
    db.add(grant)
    db.commit()
    grant_id = grant.imported_grant_id
    db.close()

    res = client.get(f"/opportunities/live/{grant_id}")
    assert res.status_code == 404

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_filter_by_status(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"status": "closed"})
    assert res.status_code == 200
    titles = {g["title"] for g in res.json()}
    assert titles == {"Rural Education Fund"}


def test_sort_by_title_ascending(client, two_imported_grants):
    res = client.get("/opportunities/live", params={"sort": "title"})
    assert res.status_code == 200
    titles = [g["title"] for g in res.json()]
    assert titles == sorted(titles)
