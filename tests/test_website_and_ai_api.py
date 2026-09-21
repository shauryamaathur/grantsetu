"""API-level tests for: NGO website_url, AI provider configuration (and its
security properties), the website-analysis endpoint, opportunity feedback
(hide/report), and the per-opportunity status/source labels. No real network
or AI calls -- fetch_url and the AI provider factory are monkeypatched."""
import api.routers.website_analysis as website_analysis_router
from src.ai.providers import AIResult
from src.discovery.fetcher import FetchResult


def _create_ngo(client, **overrides):
    payload = {
        "name": "Test NGO",
        "registration_type": "trust",
        "focus_areas": ["health"],
        "website_url": "https://example-ngo.org",
    }
    payload.update(overrides)
    res = client.post("/ngo-profile", json=payload)
    assert res.status_code == 201, res.text
    return res.json()["ngo_id"]


# --- NGO website_url ---------------------------------------------------

def test_create_ngo_profile_with_valid_website_url(client):
    ngo_id = _create_ngo(client)
    res = client.get(f"/ngo-profile/{ngo_id}")
    assert res.json()["website_url"] == "https://example-ngo.org"


def test_create_ngo_profile_rejects_invalid_website_url(client):
    res = client.post(
        "/ngo-profile",
        json={"name": "Bad URL NGO", "focus_areas": ["health"], "website_url": "not-a-url"},
    )
    assert res.status_code == 422


def test_create_ngo_profile_website_url_optional(client):
    res = client.post("/ngo-profile", json={"name": "No Website NGO", "focus_areas": ["health"]})
    assert res.status_code == 201
    assert res.json()["website_url"] is None


def test_update_ngo_profile_can_change_website_url(client):
    ngo_id = _create_ngo(client)
    res = client.put(
        f"/ngo-profile/{ngo_id}",
        json={"name": "Test NGO", "focus_areas": ["health"], "website_url": "https://new-site.org"},
    )
    assert res.status_code == 200
    assert res.json()["website_url"] == "https://new-site.org"


# --- AI provider configuration ------------------------------------------

def test_set_ai_config_never_returns_api_key(client):
    ngo_id = _create_ngo(client)
    res = client.put(
        f"/ngo-profile/{ngo_id}/ai-config",
        json={"provider": "openai", "api_key": "sk-supersecret12345", "model": "gpt-4o-mini"},
    )
    assert res.status_code == 200
    body = res.json()
    assert "api_key" not in body
    assert "api_key_encrypted" not in body
    assert "sk-supersecret12345" not in res.text
    assert body["has_api_key"] is True


def test_ai_config_rejects_unsupported_provider(client):
    ngo_id = _create_ngo(client)
    res = client.put(
        f"/ngo-profile/{ngo_id}/ai-config",
        json={"provider": "not-a-real-provider", "api_key": "sk-12345678"},
    )
    assert res.status_code == 422


def test_ai_config_for_unknown_ngo_404(client):
    res = client.put(
        "/ngo-profile/does-not-exist/ai-config",
        json={"provider": "openai", "api_key": "sk-12345678"},
    )
    assert res.status_code == 404


def test_get_ai_config_before_set_returns_404(client):
    ngo_id = _create_ngo(client)
    res = client.get(f"/ngo-profile/{ngo_id}/ai-config")
    assert res.status_code == 404


def test_update_ai_config_partial(client):
    ngo_id = _create_ngo(client)
    client.put(f"/ngo-profile/{ngo_id}/ai-config", json={"provider": "openai", "api_key": "sk-12345678"})
    res = client.patch(f"/ngo-profile/{ngo_id}/ai-config", json={"is_enabled": False})
    assert res.status_code == 200
    assert res.json()["is_enabled"] is False


def test_delete_ai_config(client):
    ngo_id = _create_ngo(client)
    client.put(f"/ngo-profile/{ngo_id}/ai-config", json={"provider": "openai", "api_key": "sk-12345678"})
    del_res = client.delete(f"/ngo-profile/{ngo_id}/ai-config")
    assert del_res.status_code == 204
    get_res = client.get(f"/ngo-profile/{ngo_id}/ai-config")
    assert get_res.status_code == 404


def test_ai_config_test_connection_reports_real_failure_for_placeholder_key(client, monkeypatch):
    """With a fake/placeholder key and no real network mocking, the real
    OpenAICompatibleProvider.complete() will fail (network error or auth
    error) -- test_connection() must report that honestly as ok=False,
    never fabricate a success."""
    import src.ai.providers as providers_module

    def fake_build_provider(provider_key, api_key, model=None, base_url=None):
        class _AlwaysFails:
            def test_connection(self):
                return AIResult(ok=False, error="invalid_api_key")

        return _AlwaysFails()

    monkeypatch.setattr(providers_module, "build_provider", fake_build_provider)
    import api.routers.ai_config as ai_config_router
    monkeypatch.setattr(ai_config_router, "build_provider", fake_build_provider)

    ngo_id = _create_ngo(client)
    client.put(f"/ngo-profile/{ngo_id}/ai-config", json={"provider": "openai", "api_key": "sk-placeholder"})
    res = client.post(f"/ngo-profile/{ngo_id}/ai-config/test")
    assert res.status_code == 200
    assert res.json()["ok"] is False


def test_ai_config_test_connection_success_path(client, monkeypatch):
    import api.routers.ai_config as ai_config_router

    def fake_build_provider(provider_key, api_key, model=None, base_url=None):
        class _AlwaysOk:
            def test_connection(self):
                return AIResult(ok=True, text="ok")

        return _AlwaysOk()

    monkeypatch.setattr(ai_config_router, "build_provider", fake_build_provider)

    ngo_id = _create_ngo(client)
    client.put(f"/ngo-profile/{ngo_id}/ai-config", json={"provider": "openai", "api_key": "sk-real-would-work"})
    res = client.post(f"/ngo-profile/{ngo_id}/ai-config/test")
    assert res.status_code == 200
    assert res.json()["ok"] is True


# --- Website analysis (AI key missing -> rule-based fallback works) ------

def test_analyze_website_without_ai_config_uses_rule_based(client, monkeypatch):
    html = """<html><head><title>Home</title></head><body>
    <p>We work on education and health for children and women in Maharashtra.</p>
    </body></html>"""

    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.discovery.website_analyzer as analyzer_module
    monkeypatch.setattr(analyzer_module, "fetch_url", fake_fetch)

    ngo_id = _create_ngo(client)
    res = client.post(f"/ngo-profile/{ngo_id}/analyze-website")
    assert res.status_code == 201
    body = res.json()
    assert body["method"] == "rule_based"
    assert "education" in body["detected_focus_areas"] or "health" in body["detected_focus_areas"]
    assert body["status"] == "pending_review"


def test_analyze_website_requires_website_url(client):
    ngo_id = _create_ngo(client, website_url=None)
    res = client.post(f"/ngo-profile/{ngo_id}/analyze-website")
    assert res.status_code == 400


def test_analyze_website_unreachable_site_returns_502(client, monkeypatch):
    def fake_fetch(url):
        return FetchResult(ok=False, url=url, error="DNS resolution failed")

    import src.discovery.website_analyzer as analyzer_module
    monkeypatch.setattr(analyzer_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(website_analysis_router, "analyze_ngo_website", analyzer_module.analyze_ngo_website)

    ngo_id = _create_ngo(client)
    res = client.post(f"/ngo-profile/{ngo_id}/analyze-website")
    assert res.status_code == 502


def test_apply_website_analysis_merges_without_overwriting(client, monkeypatch):
    html = """<html><head><title>Home</title></head><body>
    <p>We work on environment and wildlife conservation for farmers in Kerala.</p>
    </body></html>"""

    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html=html)

    import src.discovery.website_analyzer as analyzer_module
    monkeypatch.setattr(analyzer_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(website_analysis_router, "analyze_ngo_website", analyzer_module.analyze_ngo_website)

    ngo_id = _create_ngo(client)  # focus_areas=["health"]
    analysis_res = client.post(f"/ngo-profile/{ngo_id}/analyze-website")
    analysis_id = analysis_res.json()["analysis_id"]

    apply_res = client.post(
        f"/ngo-profile/{ngo_id}/website-analyses/{analysis_id}/apply",
        json={"apply_focus_areas": True, "apply_locations": True, "apply_beneficiaries": True},
    )
    assert apply_res.status_code == 200
    updated = apply_res.json()
    assert "health" in updated["focus_areas"]  # original value preserved
    assert "environment" in updated["focus_areas"]  # new value merged in


def test_list_website_analyses(client, monkeypatch):
    def fake_fetch(url):
        return FetchResult(ok=True, url=url, status_code=200, html="<html><body>education program</body></html>")

    import src.discovery.website_analyzer as analyzer_module
    monkeypatch.setattr(analyzer_module, "fetch_url", fake_fetch)
    monkeypatch.setattr(website_analysis_router, "analyze_ngo_website", analyzer_module.analyze_ngo_website)

    ngo_id = _create_ngo(client)
    client.post(f"/ngo-profile/{ngo_id}/analyze-website")
    res = client.get(f"/ngo-profile/{ngo_id}/website-analyses")
    assert res.status_code == 200
    assert len(res.json()) == 1


# --- Opportunity feedback (hide / report) --------------------------------

def test_hide_opportunity_and_it_disappears_from_recommended(client):
    """Recommended now ranks LIVE (ImportedGrant) opportunities, so this
    needs a real, non-sample ImportedGrant that matches the NGO's profile
    -- the historical corpus fixture is no longer part of this feed."""
    from api.db.models import GrantSource, ImportedGrant
    from api.db.session import SessionLocal

    db = SessionLocal()
    source = GrantSource(name="Hide Test Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    grant = ImportedGrant(
        source_id=source.source_id, external_id="hide-1", title="Environment Restoration Grant",
        category="Environment", description="Environmental conservation funding.",
        status="open", dedup_hash="hide-test-hash-1", is_sample_data=False,
    )
    db.add(grant)
    db.commit()
    db.refresh(grant)
    grant_id = grant.imported_grant_id
    db.close()

    ngo_id = _create_ngo(client, focus_areas=["environment"])
    before = client.get(f"/opportunities/recommended?ngo_id={ngo_id}").json()
    assert len(before) >= 1
    target_id = before[0]["opportunity"]["imported_grant_id"]
    assert target_id == grant_id

    hide_res = client.post(
        "/opportunity-feedback",
        json={"ngo_id": ngo_id, "opportunity_source": "imported", "opportunity_ref": str(target_id), "feedback_type": "hidden"},
    )
    assert hide_res.status_code == 201

    after = client.get(f"/opportunities/recommended?ngo_id={ngo_id}").json()
    assert all(item["opportunity"]["imported_grant_id"] != target_id for item in after)

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_report_incorrect_is_idempotent(client):
    ngo_id = _create_ngo(client)
    payload = {"ngo_id": ngo_id, "opportunity_source": "historical", "opportunity_ref": "101", "feedback_type": "reported_incorrect", "notes": "wrong deadline"}
    res1 = client.post("/opportunity-feedback", json=payload)
    res2 = client.post("/opportunity-feedback", json=payload)
    assert res1.status_code == 201
    assert res2.status_code == 201
    assert res1.json()["feedback_id"] == res2.json()["feedback_id"]  # same row, not duplicated


def test_feedback_unknown_ngo_404(client):
    res = client.post(
        "/opportunity-feedback",
        json={"ngo_id": "nope", "opportunity_source": "historical", "opportunity_ref": "101", "feedback_type": "hidden"},
    )
    assert res.status_code == 404


def test_feedback_invalid_type_rejected(client):
    ngo_id = _create_ngo(client)
    res = client.post(
        "/opportunity-feedback",
        json={"ngo_id": ngo_id, "opportunity_source": "historical", "opportunity_ref": "101", "feedback_type": "bogus"},
    )
    assert res.status_code == 422


def test_remove_feedback(client):
    ngo_id = _create_ngo(client)
    res = client.post(
        "/opportunity-feedback",
        json={"ngo_id": ngo_id, "opportunity_source": "historical", "opportunity_ref": "102", "feedback_type": "hidden"},
    )
    feedback_id = res.json()["feedback_id"]
    del_res = client.delete(f"/opportunity-feedback/{feedback_id}")
    assert del_res.status_code == 204


# --- Per-opportunity status/source labels --------------------------------

def test_search_results_include_status_and_source_labels(client):
    res = client.get("/opportunities/search?q=environmental")
    assert res.status_code == 200
    for item in res.json():
        assert item["status_label"] in {"Open", "Currently closed", "Deadline unavailable"}
        assert item["source_label"] == "Historical opportunity"


def test_opportunity_detail_includes_eligibility_status_label(client):
    res = client.get("/opportunities/102")
    assert res.status_code == 200
    assert res.json()["eligibility_status_label"] in {"Eligibility requires verification", "Eligibility stated by source"}
