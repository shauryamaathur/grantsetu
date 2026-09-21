"""Tests for the grant-source connector/sync pipeline (src/connectors/,
src/ingestion/sync_grants.py) and its notification hook."""
import pytest

from api.db.models import EmailLog, GrantSource, ImportedGrant, Subscriber
from api.db.session import SessionLocal
from src.connectors.sample_connector import SampleFixtureConnector
from src.ingestion.sync_grants import compute_dedup_hash, sync_source


def _mark_verified(email: str):
    """Test helper: only VERIFIED subscribers receive grant alerts (Phase 3
    requirement -- see src/email/notify.py::notify_subscribers_of_new_grants).
    The actual verification MECHANISM (token hashing/expiry/single-use) is
    tested thoroughly in tests/test_subscribers.py; these sync/notification
    tests just need a subscriber that IS verified, so a direct DB update is
    legitimate test isolation rather than re-deriving the token flow here."""
    db = SessionLocal()
    sub = db.query(Subscriber).filter_by(email=email).first()
    sub.status = Subscriber.STATUS_VERIFIED
    db.commit()
    db.close()


@pytest.fixture(autouse=True)
def clean_grant_tables(client):
    """All tests in this file share one SQLite file for the pytest session
    (see conftest.py), but ImportedGrant.dedup_hash is a GLOBAL uniqueness
    check across sources -- without this, one test's sample-connector import
    would look like a "duplicate" to the next test. Wipe grant-source/
    imported-grant rows before every test in this file so each test's
    dedup/new-grant-detection assertions are self-contained."""
    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()
    yield


def test_sample_connector_is_configured_and_returns_records():
    connector = SampleFixtureConnector()
    assert connector.is_configured() is True
    records = connector.fetch_records()
    assert len(records) == 3
    assert all(r.title.startswith("[SAMPLE]") for r in records)


def test_compute_dedup_hash_is_stable_and_case_insensitive():
    h1 = compute_dedup_hash("Community Grant", "Acme Foundation", "2027-01-01")
    h2 = compute_dedup_hash("community grant", "ACME FOUNDATION", "2027-01-01")
    assert h1 == h2


def test_compute_dedup_hash_differs_for_different_grants():
    h1 = compute_dedup_hash("Grant A", "Funder A", "2027-01-01")
    h2 = compute_dedup_hash("Grant B", "Funder A", "2027-01-01")
    assert h1 != h2


def test_sync_source_not_configured_status(client):
    # grants_gov_api is deliberately ALWAYS configured (it's a public API
    # with no key requirement -- see src/connectors/grants_gov_connector.py),
    # so it is not a valid "unconfigured" example any more. web_discovery
    # with no seed_urls genuinely has nothing to work with, and -- unlike
    # grants_gov_api -- correctly reports not_configured without making any
    # network call, which also keeps this test fast and offline-safe.
    db = SessionLocal()
    source = GrantSource(name="Unconfigured Test Source", connector_type="web_discovery", config={})
    db.add(source)
    db.commit()
    db.refresh(source)

    result = sync_source(db, source)
    assert result.status == "not_configured"
    assert result.new == 0
    db.close()


def test_sync_source_sample_fixture_imports_new_grants(client):
    db = SessionLocal()
    source = GrantSource(name="Test Sample Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    result = sync_source(db, source)
    assert result.status == "active"
    assert result.fetched == 3
    assert result.new == 3
    assert result.duplicates == 0
    assert len(result.new_grants) == 3
    assert all(g.is_sample_data for g in result.new_grants)

    stored = db.query(ImportedGrant).filter_by(source_id=source.source_id).count()
    assert stored == 3
    db.close()


def test_sync_source_second_run_finds_no_new_grants(client):
    db = SessionLocal()
    source = GrantSource(name="Test Sample Source 2", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    first = sync_source(db, source)
    assert first.new == 3

    second = sync_source(db, source)
    assert second.new == 0
    assert second.already_imported == 3
    db.close()


def test_sync_via_admin_endpoint_imports_sample_data_but_never_serves_it_live(client):
    """The sample/fixture connector still syncs successfully (proving the
    sync/dedup pipeline works, its original purpose), but
    GET /opportunities/live -- the real, user-facing feed -- must NEVER
    return sample rows, regardless of any query parameter. Sample data
    stays reachable only via direct DB/admin inspection, never through a
    production listing endpoint."""
    create_res = client.post(
        "/admin/grant-sources",
        json={"name": "Admin Test Source", "connector_type": "sample_fixture"},
    )
    assert create_res.status_code == 201
    source_id = create_res.json()["source_id"]

    sync_res = client.post(f"/admin/sync-source/{source_id}")
    assert sync_res.status_code == 200
    body = sync_res.json()
    assert body["status"] == "active"
    assert body["new"] == 3

    live_res = client.get("/opportunities/live")
    assert live_res.status_code == 200
    assert live_res.json() == []  # only sample data was synced, so the real feed is honestly empty


def test_sync_all_sources_refreshes_every_connected_source_in_one_call(client):
    """POST /admin/sync-all-sources (the All Opportunities page's single
    "Refresh All Sources" control) must queue a background job that syncs
    EVERY GrantSource row that exists at request time -- never a hardcoded
    subset -- and report a per-source breakdown plus aggregate totals once
    complete. Under TestClient, BackgroundTasks run synchronously as part of
    finishing the request, so the job is already done by the time
    GET /admin/refresh-jobs/{id} is polled -- see subscribers' verification-
    email tests for the same pattern."""
    ids = []
    for name in ["Source A", "Source B", "Source C"]:
        res = client.post("/admin/grant-sources", json={"name": name, "connector_type": "sample_fixture"})
        assert res.status_code == 201
        ids.append(res.json()["source_id"])

    res = client.post("/admin/sync-all-sources")
    assert res.status_code == 202
    job_id = res.json()["job_id"]

    job = client.get(f"/admin/refresh-jobs/{job_id}").json()
    assert job["status"] == "completed"
    assert job["sources_checked"] == 3
    # dedup_hash is a GLOBAL uniqueness check across sources (see
    # clean_grant_tables above), and the sample connector always returns
    # the same 3 fixture records -- so only the FIRST source's records
    # count as genuinely new; the other two sources see the same records
    # as duplicates of what source A just imported. That's correct dedup
    # behavior, not a bug in the aggregate endpoint.
    assert job["total_new"] == 3
    assert job["total_fetched"] == 9  # every source still fetched its 3 records
    assert {s["source_id"] for s in job["per_source"]} == set(ids)
    assert all(s["status"] == "active" for s in job["per_source"])
    assert sum(s["new"] for s in job["per_source"]) == 3

    # A second run finds nothing new -- proves it actually synced (via the
    # same dedup pipeline as the per-source endpoint), not a no-op stub.
    res2 = client.post("/admin/sync-all-sources")
    job2_id = res2.json()["job_id"]
    job2 = client.get(f"/admin/refresh-jobs/{job2_id}").json()
    assert job2["total_new"] == 0


def test_sync_all_sources_with_no_connected_sources_is_a_harmless_no_op(client):
    res = client.post("/admin/sync-all-sources")
    assert res.status_code == 202
    job = client.get(f"/admin/refresh-jobs/{res.json()['job_id']}").json()
    assert job["status"] == "completed"
    assert job["sources_checked"] == 0
    assert job["total_fetched"] == 0
    assert job["total_new"] == 0
    assert job["total_updated"] == 0
    assert job["notifications_sent"] == 0
    assert job["per_source"] == []


def test_refresh_job_404_for_unknown_id(client):
    res = client.get("/admin/refresh-jobs/does-not-exist")
    assert res.status_code == 404


def test_duplicate_grant_source_name_conflicts(client):
    client.post("/admin/grant-sources", json={"name": "Dup Source", "connector_type": "sample_fixture"})
    res2 = client.post("/admin/grant-sources", json={"name": "Dup Source", "connector_type": "sample_fixture"})
    assert res2.status_code == 409


def test_sync_unknown_source_404(client):
    res = client.post("/admin/sync-source/does-not-exist")
    assert res.status_code == 404


def test_new_grant_triggers_notification_for_matching_subscriber(client):
    # Subscribe with a linked NGO whose focus area matches a sample grant's category ("Health").
    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Health NGO", "registration_type": "trust", "focus_areas": ["health"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]
    client.post("/subscribe", json={"email": "healthsub@example.com", "ngo_id": ngo_id})
    _mark_verified("healthsub@example.com")

    create_res = client.post(
        "/admin/grant-sources",
        json={"name": "Notify Test Source", "connector_type": "sample_fixture"},
    )
    source_id = create_res.json()["source_id"]

    sync_res = client.post(f"/admin/sync-source/{source_id}")
    assert sync_res.json()["notifications_sent"] >= 1

    db = SessionLocal()
    logs = db.query(EmailLog).filter_by(email_type="new_grant_alert").all()
    db.close()
    assert len(logs) >= 1
    assert all(log.status == "sent_mock" for log in logs)


def test_sync_source_uses_ai_provider_when_ai_ngo_id_configured(client, monkeypatch):
    """GrantSource.config["ai_ngo_id"] opts a source into AI-assisted
    extraction (src/ingestion/sync_grants.py) -- this test locks in that
    sync_source() actually resolves that NGO's provider and passes it
    through to the connector, rather than the wiring only existing at the
    connector level. get_provider_for_ngo is monkeypatched (not a real AI
    call) since credential handling/decryption is already covered by
    tests/test_ai.py."""
    from src.discovery.fetcher import FetchResult

    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "AI-Enabled NGO", "registration_type": "trust", "focus_areas": ["health"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]

    class _StubProvider:
        def complete(self, prompt, max_tokens=700):
            from src.ai.providers import AIResult

            return AIResult(ok=True, text='{"title": "AI Grant", "funder": "AI Funder", "deadline": null}')

    import src.ingestion.sync_grants as sync_module
    monkeypatch.setattr(
        sync_module,
        "get_provider_for_ngo",
        lambda db, ngo_id_arg: _StubProvider() if ngo_id_arg == ngo_id else None,
    )

    def fake_fetch(url):
        html = """<html><head><title>Rural Grant</title></head><body>
        <p>Eligibility: nonprofits may apply.</p>
        <p>Grant application deadline: 2027-05-01.</p>
        </body></html>"""
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    db = SessionLocal()
    source = GrantSource(
        name="AI-Enriched Source",
        connector_type="web_discovery",
        config={"seed_urls": ["https://example.org/grant"], "ai_ngo_id": ngo_id},
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    result = sync_source(db, source)
    assert result.status == "active"
    assert result.new == 1

    grant = db.query(ImportedGrant).filter_by(source_id=source.source_id).first()
    assert grant.title == "AI Grant"
    assert grant.extraction_method == "ai_inferred"
    db.close()


def test_sync_source_ignores_unresolvable_ai_ngo_id(client):
    """An `ai_ngo_id` pointing at a profile with no AI config (or that
    doesn't exist) must not fail the sync -- get_provider_for_ngo returns
    None and the source falls back to rule-based extraction, same as if
    `ai_ngo_id` were never set."""
    from src.discovery.fetcher import FetchResult

    def fake_fetch(url):
        html = """<html><head><title>Rural Grant</title></head><body>
        <p>Eligibility: nonprofits may apply.</p>
        <p>Grant application deadline: 2027-05-01.</p>
        </body></html>"""
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.connectors.web_discovery_connector as wdc_module
    wdc_module.fetch_url = fake_fetch

    db = SessionLocal()
    source = GrantSource(
        name="Unresolvable AI Source",
        connector_type="web_discovery",
        config={"seed_urls": ["https://example.org/grant"], "ai_ngo_id": "does-not-exist"},
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    result = sync_source(db, source)
    assert result.status == "active"
    assert result.new == 1

    grant = db.query(ImportedGrant).filter_by(source_id=source.source_id).first()
    assert grant.extraction_method == "rule_based"
    db.close()


def test_resync_detects_deadline_change_and_logs_it(client, monkeypatch):
    """A source that changes an already-imported grant's deadline (or other
    tracked field) on a later sync must UPDATE the existing row and log an
    OpportunityChange -- not just bump last_verified_at and ignore the
    change (the real gap this test locks in a fix for)."""
    from api.db.models import OpportunityChange
    from src.discovery.fetcher import FetchResult

    def make_html(deadline: str) -> str:
        return f"""<html><head><title>Rural Grant</title></head><body>
        <p>Eligibility: nonprofits may apply.</p>
        <p>Grant application deadline: {deadline}.</p>
        </body></html>"""

    calls = {"n": 0}

    def fake_fetch(url):
        calls["n"] += 1
        deadline = "2027-05-01" if calls["n"] == 1 else "2027-06-15"
        return FetchResult(ok=True, url=url, status_code=200, html=make_html(deadline))

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    db = SessionLocal()
    source = GrantSource(
        name="Change Detection Source",
        connector_type="web_discovery",
        config={"seed_urls": ["https://example.org/grant"]},
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    first = sync_source(db, source)
    assert first.new == 1
    assert first.updated == 0

    second = sync_source(db, source)
    assert second.new == 0
    assert second.already_imported == 1
    assert second.updated == 1

    grant = db.query(ImportedGrant).filter_by(source_id=source.source_id).first()
    assert grant.deadline.date().isoformat() == "2027-06-15"

    changes = db.query(OpportunityChange).filter_by(imported_grant_id=grant.imported_grant_id).all()
    # The rule-based description is an extractive summary of page text that
    # happens to include the deadline sentence verbatim, so it legitimately
    # changes too -- this test only asserts on the deadline_changed row.
    deadline_changes = [c for c in changes if c.change_type == "deadline_changed"]
    assert len(deadline_changes) == 1
    assert deadline_changes[0].old_value == "2027-05-01T00:00:00"
    assert deadline_changes[0].new_value == "2027-06-15T00:00:00"
    db.close()


def test_resync_with_identical_content_reports_no_update(client, monkeypatch):
    from src.discovery.fetcher import FetchResult

    def fake_fetch(url):
        html = """<html><head><title>Rural Grant</title></head><body>
        <p>Eligibility: nonprofits may apply.</p>
        <p>Grant application deadline: 2027-05-01.</p>
        </body></html>"""
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    db = SessionLocal()
    source = GrantSource(
        name="No Change Source",
        connector_type="web_discovery",
        config={"seed_urls": ["https://example.org/grant"]},
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    sync_source(db, source)
    second = sync_source(db, source)
    assert second.updated == 0
    db.close()


def test_resync_detects_reopened_status(client, monkeypatch):
    """A grant that was expired (deadline already passed at first sync) but
    is re-listed with a future deadline on a later sync must be marked
    reopened, not left stuck at "expired" forever."""
    from api.db.models import OpportunityChange
    from src.discovery.fetcher import FetchResult

    calls = {"n": 0}

    def fake_fetch(url):
        calls["n"] += 1
        deadline = "2020-01-01" if calls["n"] == 1 else "2027-09-01"
        html = f"""<html><head><title>Rural Grant</title></head><body>
        <p>Eligibility: nonprofits may apply.</p>
        <p>Grant application deadline: {deadline}.</p>
        </body></html>"""
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.connectors.web_discovery_connector as wdc_module
    monkeypatch.setattr(wdc_module, "fetch_url", fake_fetch)

    db = SessionLocal()
    source = GrantSource(
        name="Reopen Source",
        connector_type="web_discovery",
        config={"seed_urls": ["https://example.org/grant"]},
    )
    db.add(source)
    db.commit()
    db.refresh(source)

    sync_source(db, source)
    grant = db.query(ImportedGrant).filter_by(source_id=source.source_id).first()
    assert grant.status == "expired"

    second = sync_source(db, source)
    assert second.updated == 1
    db.refresh(grant)
    assert grant.status == "unknown"  # web_discovery never confirms "open"; just no longer past-deadline

    changes = db.query(OpportunityChange).filter_by(imported_grant_id=grant.imported_grant_id).all()
    assert any(c.change_type == "reopened" for c in changes)
    db.close()


def test_second_sync_does_not_re_notify_same_subscriber(client):
    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Edu NGO", "registration_type": "trust", "focus_areas": ["education"]},
    )
    ngo_id = ngo_res.json()["ngo_id"]
    client.post("/subscribe", json={"email": "edusub@example.com", "ngo_id": ngo_id})
    _mark_verified("edusub@example.com")

    create_res = client.post(
        "/admin/grant-sources",
        json={"name": "No Re-notify Source", "connector_type": "sample_fixture"},
    )
    source_id = create_res.json()["source_id"]

    first_sync = client.post(f"/admin/sync-source/{source_id}")
    first_notified = first_sync.json()["notifications_sent"]
    assert first_notified >= 1

    second_sync = client.post(f"/admin/sync-source/{source_id}")
    assert second_sync.json()["new"] == 0
    assert second_sync.json()["notifications_sent"] == 0
