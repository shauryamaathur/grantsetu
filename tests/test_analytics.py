"""Grant Intelligence (analytics) endpoints -- all computed from the fake
3-row corpus fixture in conftest.py, so exact numbers are asserted rather
than just status codes."""


def test_overview_real_numbers(client):
    r = client.get("/analytics/overview")
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 3
    assert data["open_count"] == 2
    assert data["expired_count"] == 1
    assert data["unique_funders"] == 3
    assert data["unique_categories"] == 3


def test_trends_uses_real_dates(client):
    r = client.get("/analytics/trends")
    assert r.status_code == 200
    data = r.json()
    months = {row["month"] for row in data["postings_by_month"]}
    assert "2024-01" in months
    assert "2010-01" in months


def test_funders_ranks_by_count_and_funding(client):
    r = client.get("/analytics/funders")
    assert r.status_code == 200
    data = r.json()
    assert {f["funder"] for f in data["top_by_opportunity_count"]} == {"Agency A", "Agency B", "Agency C"}
    top_funding = data["top_by_total_estimated_funding"][0]
    assert top_funding["funder"] == "Agency C"  # 2,000,000 estimated funding, the largest


def test_categories_breakdown(client):
    r = client.get("/analytics/categories")
    assert r.status_code == 200
    cats = {c["category"] for c in r.json()["categories"]}
    assert cats == {"Environment", "Health", "Infrastructure"}


def test_eligibility_breakdown(client):
    r = client.get("/analytics/eligibility")
    assert r.status_code == 200
    types = {e["type"] for e in r.json()["eligible_applicant_types"]}
    assert "Non-Government Organization" in types


def test_explorer_pagination_and_filter(client):
    r = client.get("/analytics/explorer", params={"category": "Health"})
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 1
    assert data["rows"][0]["opportunity_title"] == "Rural Health Initiative"


def test_export_csv_returns_real_rows(client):
    r = client.get("/analytics/export.csv")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    body = r.text
    assert "Environmental Youth Grant" in body
    assert body.count("\n") >= 4  # header + 3 data rows (+ trailing newline)


def test_export_json_returns_real_rows(client):
    r = client.get("/analytics/export.json")
    assert r.status_code == 200
    import json
    rows = json.loads(r.text)
    assert len(rows) == 3
    titles = {row["opportunity_title"] for row in rows}
    assert "Urban Roads Fund" in titles


def test_analysis_skips_clustering_below_threshold(client):
    # The fixture corpus only has 3 rows -- far below the 20-row minimum the
    # endpoint requires before it will claim a cluster shape means anything.
    r = client.get("/analytics/analysis")
    assert r.status_code == 200
    data = r.json()
    assert data["sample_size"] == 3
    assert data["clusters"] is None
    assert "skipped" in data["clustering_note"].lower()


def test_analysis_runs_clustering_with_enough_rows(client, monkeypatch):
    import numpy as np
    import pandas as pd

    from api.routers import analytics as analytics_module

    rng = np.random.RandomState(0)
    n = 60
    df = pd.DataFrame({
        "award_ceiling": rng.uniform(10_000, 500_000, n),
        "award_floor": rng.uniform(1_000, 50_000, n),
        "opportunity_category": rng.choice(["Health", "Education", "Environment"], n),
        "opportunity_title": [f"Grant {i}" for i in range(n)],
        "estimated_total_program_funding": rng.uniform(10_000, 500_000, n),
    })

    class _FakeCorpus:
        pass

    fake = _FakeCorpus()
    fake.df = df
    monkeypatch.setattr(analytics_module, "get_corpus", lambda: fake)

    r = client.get("/analytics/analysis")
    assert r.status_code == 200
    data = r.json()
    assert data["sample_size"] == n
    assert data["clusters"] is not None
    assert data["clusters"]["k"] >= 2
    assert len(data["clusters"]["points"]) == n
    assert data["award_ceiling_vs_total_funding_correlation"] is not None


def test_live_summary_empty_by_default(client):
    r = client.get("/analytics/live-summary")
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 0


def test_overview_filters_by_category(client):
    r = client.get("/analytics/overview", params={"category": "Health"})
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 1
    assert data["unique_funders"] == 1
    assert data["filters_applied"] == {"category": "Health", "funder": None, "year": None}


def test_overview_filters_by_year(client):
    r = client.get("/analytics/overview", params={"year": 2010})
    assert r.status_code == 200
    assert r.json()["total_rows"] == 1  # only the Infrastructure row was posted in 2010


def test_available_years_reflects_real_data(client):
    r = client.get("/analytics/available-years")
    assert r.status_code == 200
    assert set(r.json()["years"]) == {2010, 2024}


def test_trends_filters_by_category(client):
    r = client.get("/analytics/trends", params={"category": "Infrastructure"})
    assert r.status_code == 200
    months = {row["month"] for row in r.json()["postings_by_month"]}
    assert months == {"2010-01"}


def test_funders_filters_by_category(client):
    r = client.get("/analytics/funders", params={"category": "Environment"})
    assert r.status_code == 200
    data = r.json()
    assert {f["funder"] for f in data["top_by_opportunity_count"]} == {"Agency A"}


def test_compare_two_categories(client):
    r = client.get("/analytics/compare", params={"dimension": "category", "a": "Health", "b": "Infrastructure"})
    assert r.status_code == 200
    data = r.json()
    assert data["a"]["label"] == "Health"
    assert data["a"]["total_rows"] == 1
    assert data["a"]["open_count"] == 1
    assert data["b"]["label"] == "Infrastructure"
    assert data["b"]["total_rows"] == 1
    assert data["b"]["expired_count"] == 1


def test_compare_rejects_unsupported_dimension(client):
    r = client.get("/analytics/compare", params={"dimension": "geography", "a": "India", "b": "US"})
    assert r.status_code == 422


def test_live_geography_empty_by_default(client):
    r = client.get("/analytics/live/geography")
    assert r.status_code == 200
    assert r.json() == {"total_rows": 0, "by_country": [], "note": r.json()["note"]}


def test_live_geography_real_counts(client):
    from api.db.models import GrantSource, ImportedGrant
    from api.db.session import SessionLocal

    db = SessionLocal()
    source = GrantSource(name="GI Geo Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    db.add_all([
        ImportedGrant(source_id=source.source_id, external_id="g1", title="India Grant", status="open", country="India", dedup_hash="gigeo1", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="g2", title="Unknown Grant", status="open", country="unknown", dedup_hash="gigeo2", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="g3", title="Sample Grant", status="open", country="India", dedup_hash="gigeo3", is_sample_data=True),
    ])
    db.commit()
    db.close()

    r = client.get("/analytics/live/geography")
    assert r.status_code == 200
    data = r.json()
    assert data["total_rows"] == 2  # sample row excluded
    by_country = {row["country"]: row["count"] for row in data["by_country"]}
    assert by_country == {"India": 1, "unknown": 1}

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_live_deadlines_buckets_real_dates(client):
    from datetime import datetime, timedelta

    from api.db.models import GrantSource, ImportedGrant
    from api.db.session import SessionLocal

    db = SessionLocal()
    source = GrantSource(name="GI Deadline Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    now = datetime.utcnow()
    db.add_all([
        ImportedGrant(source_id=source.source_id, external_id="d1", title="Soon", status="open", deadline=now + timedelta(days=3), dedup_hash="gid1", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="d2", title="This month", status="open", deadline=now + timedelta(days=20), dedup_hash="gid2", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="d3", title="No deadline", status="unknown", deadline=None, dedup_hash="gid3", is_sample_data=False),
    ])
    db.commit()
    db.close()

    r = client.get("/analytics/live/deadlines")
    assert r.status_code == 200
    data = r.json()
    assert data["buckets"]["closing_this_week"] == 1
    assert data["buckets"]["closing_this_month"] == 1
    assert data["buckets"]["no_deadline"] == 1
    assert data["total_active_rows"] == 3

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_live_funders_ranks_real_counts(client):
    from api.db.models import GrantSource, ImportedGrant
    from api.db.session import SessionLocal

    db = SessionLocal()
    source = GrantSource(name="GI Funder Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    db.add_all([
        ImportedGrant(source_id=source.source_id, external_id="f1", title="A", funder="Ford Foundation", status="open", dedup_hash="gif1", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="f2", title="B", funder="Ford Foundation", status="open", dedup_hash="gif2", is_sample_data=False),
        ImportedGrant(source_id=source.source_id, external_id="f3", title="C", funder=None, status="open", dedup_hash="gif3", is_sample_data=False),
    ])
    db.commit()
    db.close()

    r = client.get("/analytics/live/funders")
    assert r.status_code == 200
    data = r.json()
    assert data["top_funders"][0] == {"funder": "Ford Foundation", "count": 2}
    assert data["unattributed"] == 1

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()
