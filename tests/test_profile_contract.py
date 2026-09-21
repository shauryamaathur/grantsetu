"""Regression tests for the NGO-profile lifecycle bug: the frontend was
calling a stale/deleted ngo_id from localStorage (left over from a database
that had since been reset during development), producing repeated 404s
against /ngo-profile/{id} and /ngo-profile/{id}/ai-config with no recovery.

The frontend fix relies on a precise API CONTRACT: every endpoint that
requires an existing NGOProfile must fail with EXACTLY the detail message
"NGO profile '<id>' not found" when it doesn't exist -- the frontend's
`PROFILE_NOT_FOUND_RE` regex in frontend/index.html matches on this exact
string to distinguish "the profile itself is gone" from other 404s (like
"no AI config saved yet", which must NOT trigger the same "clear everything"
recovery path). These tests lock that contract in place across every
profile-scoped router, so a future wording change would break a test here
before it breaks the app silently in the browser.
"""
import re

PROFILE_NOT_FOUND_RE = re.compile(r"^NGO profile '.*' not found$")


def _assert_profile_not_found(detail):
    assert isinstance(detail, str), f"expected a string detail, got {detail!r}"
    assert PROFILE_NOT_FOUND_RE.match(detail), (
        f"detail {detail!r} does not match the exact pattern the frontend's "
        "ProfileNotFoundError classification depends on"
    )


STALE_NGO_ID = "00000000-0000-0000-0000-000000000000"


# --- Profile creation and persistence -----------------------------------

def test_profile_creation_persists_and_is_immediately_retrievable(client):
    create_res = client.post(
        "/ngo-profile",
        json={"name": "Persistence Check NGO", "focus_areas": ["health"]},
    )
    assert create_res.status_code == 201
    ngo_id = create_res.json()["ngo_id"]
    assert ngo_id  # a real, non-empty id was returned

    # Round-trip: the exact id returned by creation must resolve immediately.
    get_res = client.get(f"/ngo-profile/{ngo_id}")
    assert get_res.status_code == 200
    assert get_res.json()["ngo_id"] == ngo_id
    assert get_res.json()["name"] == "Persistence Check NGO"


def test_profile_update_persists(client):
    ngo_id = client.post("/ngo-profile", json={"name": "Original", "focus_areas": ["health"]}).json()["ngo_id"]
    update_res = client.put(f"/ngo-profile/{ngo_id}", json={"name": "Updated", "focus_areas": ["education"]})
    assert update_res.status_code == 200

    get_res = client.get(f"/ngo-profile/{ngo_id}")
    assert get_res.json()["name"] == "Updated"
    assert get_res.json()["focus_areas"] == ["education"]


# --- Stale/missing profile ID handling: exact 404 contract ---------------

def test_get_unknown_profile_matches_not_found_contract(client):
    res = client.get(f"/ngo-profile/{STALE_NGO_ID}")
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_put_unknown_profile_matches_not_found_contract(client):
    res = client.put(f"/ngo-profile/{STALE_NGO_ID}", json={"name": "X", "focus_areas": ["health"]})
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_ai_config_get_unknown_profile_matches_not_found_contract(client):
    res = client.get(f"/ngo-profile/{STALE_NGO_ID}/ai-config")
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_ai_config_put_unknown_profile_matches_not_found_contract(client):
    res = client.put(f"/ngo-profile/{STALE_NGO_ID}/ai-config", json={"provider": "openai", "api_key": "sk-12345678"})
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_ai_config_test_unknown_profile_matches_not_found_contract(client):
    res = client.post(f"/ngo-profile/{STALE_NGO_ID}/ai-config/test")
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_analyze_website_unknown_profile_matches_not_found_contract(client):
    res = client.post(f"/ngo-profile/{STALE_NGO_ID}/analyze-website")
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_saved_opportunity_unknown_profile_matches_not_found_contract(client):
    res = client.post("/saved-opportunities", json={"ngo_id": STALE_NGO_ID, "opportunity_id": 101})
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_opportunity_feedback_unknown_profile_matches_not_found_contract(client):
    res = client.post(
        "/opportunity-feedback",
        json={"ngo_id": STALE_NGO_ID, "opportunity_source": "historical", "opportunity_ref": "101", "feedback_type": "hidden"},
    )
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


def test_subscribe_unknown_profile_matches_not_found_contract(client):
    res = client.post("/subscribe", json={"email": "stale-ngo-check@example.com", "ngo_id": STALE_NGO_ID})
    assert res.status_code == 404
    _assert_profile_not_found(res.json()["detail"])


# --- The critical distinguishing case: a REAL profile with NO AI config --
# This must be a 404 too, but its detail message must NOT match the
# "profile not found" pattern -- otherwise the frontend would incorrectly
# treat "you haven't configured AI yet" as "your whole profile is gone" and
# wipe a perfectly valid ngo_id.

def test_ai_config_missing_on_a_real_profile_is_not_classified_as_profile_not_found(client):
    ngo_id = client.post("/ngo-profile", json={"name": "Real NGO", "focus_areas": ["health"]}).json()["ngo_id"]

    res = client.get(f"/ngo-profile/{ngo_id}/ai-config")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert not PROFILE_NOT_FOUND_RE.match(detail), (
        f"detail {detail!r} incorrectly matches the profile-not-found pattern -- "
        "this would make the frontend wipe a valid ngo_id just because no AI "
        "config has been saved yet"
    )
    assert "AI config" in detail


# --- AI configuration with and without an existing profile ---------------

def test_ai_config_full_lifecycle_requires_a_real_profile(client):
    ngo_id = client.post("/ngo-profile", json={"name": "AI Lifecycle NGO", "focus_areas": ["health"]}).json()["ngo_id"]

    # No config yet -> empty state, not an error the UI should alarm on.
    assert client.get(f"/ngo-profile/{ngo_id}/ai-config").status_code == 404

    # Save -> retrievable, key never echoed back.
    put_res = client.put(f"/ngo-profile/{ngo_id}/ai-config", json={"provider": "openai", "api_key": "sk-1234567890"})
    assert put_res.status_code == 200
    assert "api_key_encrypted" not in put_res.json()
    assert "sk-1234567890" not in put_res.text

    get_res = client.get(f"/ngo-profile/{ngo_id}/ai-config")
    assert get_res.status_code == 200
    assert get_res.json()["has_api_key"] is True

    # Remove -> back to empty state.
    assert client.delete(f"/ngo-profile/{ngo_id}/ai-config").status_code == 204
    assert client.get(f"/ngo-profile/{ngo_id}/ai-config").status_code == 404


# --- Route/API contract consistency (method + path presence) -------------

def test_every_frontend_called_route_exists_with_the_expected_method(client):
    """A direct assertion that the exact (method, path-template) pairs the
    frontend calls (see frontend/index.html's apiFetch/apiPost/apiPut/
    apiPatch/apiDelete call sites) are actually registered on the FastAPI
    app -- catches a route/method typo on either side before it reaches
    the browser as a 404."""
    from api.main import app

    registered = set()
    for route in app.routes:
        if hasattr(route, "methods"):
            for method in route.methods:
                if method != "HEAD":
                    registered.add((method, route.path))

    expected = [
        ("POST", "/ngo-profile"),
        ("GET", "/ngo-profile/{ngo_id}"),
        ("PUT", "/ngo-profile/{ngo_id}"),
        ("GET", "/ngo-profile/{ngo_id}/ai-config"),
        ("PUT", "/ngo-profile/{ngo_id}/ai-config"),
        ("PATCH", "/ngo-profile/{ngo_id}/ai-config"),
        ("DELETE", "/ngo-profile/{ngo_id}/ai-config"),
        ("POST", "/ngo-profile/{ngo_id}/ai-config/test"),
        ("POST", "/ngo-profile/{ngo_id}/analyze-website"),
        ("POST", "/ngo-profile/{ngo_id}/website-analyses/{analysis_id}/apply"),
        ("GET", "/opportunities/search"),
        ("GET", "/opportunities/recommended"),
        ("GET", "/opportunities/live"),
        ("GET", "/opportunities/{opportunity_id}"),
        ("POST", "/opportunity-feedback"),
        ("POST", "/saved-opportunities"),
        ("GET", "/saved-opportunities"),
        ("PATCH", "/saved-opportunities/{saved_id}"),
        ("DELETE", "/saved-opportunities/{saved_id}"),
        ("POST", "/subscribe"),
        ("GET", "/grant-sources"),
        ("POST", "/admin/sync-source/{source_id}"),
    ]
    missing = [pair for pair in expected if pair not in registered]
    assert not missing, f"routes the frontend calls but the backend does not register: {missing}"
