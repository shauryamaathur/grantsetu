"""Saved-opportunities coverage for the new opportunity_source/opportunity_ref
path (historical vs. imported/live grants) added alongside Live Grant
Discovery. The original historical-only flow is already covered by
tests/test_api.py -- these tests focus specifically on the imported-grant
path and on the two sources not colliding with each other."""
import pytest

from api.db.models import GrantSource, ImportedGrant
from api.db.session import SessionLocal


@pytest.fixture
def imported_grant():
    db = SessionLocal()
    source = GrantSource(name="Test Live Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    grant = ImportedGrant(
        source_id=source.source_id,
        external_id="ext-1",
        title="Live Community Health Grant",
        funder="Test Funder",
        category="Health",
        status="open",
        dedup_hash="test-hash-1",
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


def _create_ngo(client):
    res = client.post("/ngo-profile", json={"name": "Test NGO", "focus_areas": ["health"]})
    return res.json()["ngo_id"]


def test_save_imported_grant(client, imported_grant):
    ngo_id = _create_ngo(client)
    res = client.post(
        "/saved-opportunities",
        json={"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": imported_grant},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["opportunity_source"] == "imported"
    assert body["opportunity_ref"] == imported_grant
    assert body["title"] == "Live Community Health Grant"
    assert body["funder"] == "Test Funder"
    assert body["is_expired"] is False


def test_save_unknown_imported_grant_404(client):
    ngo_id = _create_ngo(client)
    res = client.post(
        "/saved-opportunities",
        json={"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": "does-not-exist"},
    )
    assert res.status_code == 404


def test_save_imported_grant_missing_ref_422(client):
    ngo_id = _create_ngo(client)
    res = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_source": "imported"})
    assert res.status_code == 422


def test_duplicate_imported_save_rejected(client, imported_grant):
    ngo_id = _create_ngo(client)
    payload = {"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": imported_grant}
    first = client.post("/saved-opportunities", json=payload)
    assert first.status_code == 201
    second = client.post("/saved-opportunities", json=payload)
    assert second.status_code == 409


def test_historical_and_imported_saves_coexist(client, imported_grant):
    """Same NGO can save a historical-corpus opportunity AND an imported
    grant without them colliding on the shared saved_opportunities table."""
    ngo_id = _create_ngo(client)
    hist = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 101})
    assert hist.status_code == 201
    live = client.post(
        "/saved-opportunities",
        json={"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": imported_grant},
    )
    assert live.status_code == 201

    listing = client.get(f"/saved-opportunities?ngo_id={ngo_id}")
    assert listing.status_code == 200
    rows = listing.json()
    assert len(rows) == 2
    sources = {r["opportunity_source"] for r in rows}
    assert sources == {"historical", "imported"}
    titles = {r["title"] for r in rows}
    assert "Environmental Youth Grant" in titles
    assert "Live Community Health Grant" in titles


def test_save_historical_with_pandas_na_url_serializes_as_null(client, monkeypatch):
    """Regression test: the real 75k-row corpus stores
    additional_information_url as a pandas nullable-string column, so a
    missing value is pandas.NA, not Python None. Assigning that straight to
    a Pydantic str|None field coerced it to the literal string "<NA>"
    instead of a real null -- found via a live smoke test against the real
    dev database, not something the all-Python-None fixture in conftest.py
    would ever catch."""
    import pandas as pd

    from api.routers import saved as saved_router

    df = pd.DataFrame({
        "opportunity_id": pd.array([9001], dtype="Int64"),
        "opportunity_title": pd.array(["NA-URL Test Grant"], dtype="string"),
        "agency_name": pd.array([pd.NA], dtype="string"),
        "close_date": pd.to_datetime([None]),
        "is_expired": [False],
        "additional_information_url": pd.array([pd.NA], dtype="string"),
    })

    class _FakeCorpus:
        def get_opportunity(self, opportunity_id):
            matches = df[df["opportunity_id"] == opportunity_id]
            return None if matches.empty else matches.iloc[0]

    fake = _FakeCorpus()
    monkeypatch.setattr(saved_router, "get_corpus", lambda: fake)

    ngo_id = _create_ngo(client)
    res = client.post("/saved-opportunities", json={"ngo_id": ngo_id, "opportunity_id": 9001})
    assert res.status_code == 201
    body = res.json()
    assert body["url"] is None
    assert body["funder"] is None


def test_remove_imported_save(client, imported_grant):
    ngo_id = _create_ngo(client)
    saved = client.post(
        "/saved-opportunities",
        json={"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": imported_grant},
    ).json()
    delete_res = client.delete(f"/saved-opportunities/{saved['saved_id']}")
    assert delete_res.status_code == 204
    listing = client.get(f"/saved-opportunities?ngo_id={ngo_id}").json()
    assert listing == []
