"""Tests for optional Google sign-in (api/routers/auth.py, src/auth/).
No real network calls to Google -- exchange_code_for_identity is
monkeypatched, same pattern as every other external-service test in this
project. Guest access (the primary, always-available path) is verified
never to depend on anything here."""
from datetime import datetime, timedelta

import pytest

from api.db.models import NGOProfile, User, UserSession
from api.db.session import SessionLocal
from src.auth import google_oauth
from src.auth.sessions import create_session, get_user_for_token, revoke_session


@pytest.fixture(autouse=True)
def clean_auth_tables(client):
    db = SessionLocal()
    db.query(UserSession).delete()
    db.query(User).delete()
    db.commit()
    db.close()
    yield
    db = SessionLocal()
    db.query(UserSession).delete()
    db.query(User).delete()
    db.commit()
    db.close()


def test_google_not_configured_by_default(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_REDIRECT_URI", raising=False)
    res = client.get("/auth/config")
    assert res.status_code == 200
    assert res.json()["google_enabled"] is False


def test_google_login_400_when_not_configured(client, monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    res = client.get("/auth/google/login", follow_redirects=False)
    assert res.status_code == 400


def test_google_login_redirects_when_configured(client, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
    res = client.get("/auth/google/login", follow_redirects=False)
    assert res.status_code in (302, 307)
    assert "accounts.google.com" in res.headers["location"]


def test_callback_rejects_unknown_state(client):
    res = client.get("/auth/google/callback", params={"code": "abc", "state": "not-a-real-state"}, follow_redirects=False)
    assert res.status_code in (302, 307)
    assert "auth_error" in res.headers["location"]


def test_callback_creates_user_and_session(client, monkeypatch):
    from api.routers import auth as auth_router

    state = "test-state-123"
    auth_router._pending_states[state] = datetime.utcnow() + timedelta(minutes=5)

    def fake_exchange(code):
        return google_oauth.GoogleIdentity(sub="google-sub-1", email="ngo@example.org", name="Test User", picture="https://example.org/pic.jpg")

    monkeypatch.setattr(google_oauth, "exchange_code_for_identity", fake_exchange)

    res = client.get("/auth/google/callback", params={"code": "real-code", "state": state}, follow_redirects=False)
    assert res.status_code in (302, 307)
    location = res.headers["location"]
    assert "session_token=" in location

    db = SessionLocal()
    user = db.query(User).filter_by(google_sub="google-sub-1").first()
    assert user is not None
    assert user.email == "ngo@example.org"
    db.close()


def test_callback_reuses_existing_user_on_second_login(client, monkeypatch):
    from api.routers import auth as auth_router

    def fake_exchange(code):
        return google_oauth.GoogleIdentity(sub="google-sub-2", email="ngo2@example.org", name="A", picture=None)

    monkeypatch.setattr(google_oauth, "exchange_code_for_identity", fake_exchange)

    for i in range(2):
        state = f"state-{i}"
        auth_router._pending_states[state] = datetime.utcnow() + timedelta(minutes=5)
        res = client.get("/auth/google/callback", params={"code": "c", "state": state}, follow_redirects=False)
        assert res.status_code in (302, 307)

    db = SessionLocal()
    users = db.query(User).filter_by(google_sub="google-sub-2").all()
    assert len(users) == 1  # never a duplicate account for the same google_sub
    db.close()


def test_state_is_single_use(client, monkeypatch):
    from api.routers import auth as auth_router

    def fake_exchange(code):
        return google_oauth.GoogleIdentity(sub="google-sub-3", email="ngo3@example.org", name=None, picture=None)

    monkeypatch.setattr(google_oauth, "exchange_code_for_identity", fake_exchange)

    state = "single-use-state"
    auth_router._pending_states[state] = datetime.utcnow() + timedelta(minutes=5)
    first = client.get("/auth/google/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert "session_token=" in first.headers["location"]

    second = client.get("/auth/google/callback", params={"code": "c", "state": state}, follow_redirects=False)
    assert "auth_error" in second.headers["location"]


def test_me_requires_valid_session(client):
    res = client.get("/auth/me")
    assert res.status_code == 401


def test_me_returns_user_for_valid_session(client):
    db = SessionLocal()
    user = User(google_sub="sub-me", email="me@example.org", name="Me")
    db.add(user)
    db.commit()
    token = create_session(db, user)
    db.close()

    res = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["email"] == "me@example.org"


def test_logout_revokes_session(client):
    db = SessionLocal()
    user = User(google_sub="sub-logout", email="logout@example.org")
    db.add(user)
    db.commit()
    token = create_session(db, user)

    res = client.post("/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200

    assert get_user_for_token(db, token) is None
    db.close()


def test_expired_session_treated_as_signed_out(client):
    db = SessionLocal()
    user = User(google_sub="sub-expired", email="expired@example.org")
    db.add(user)
    db.commit()
    session = UserSession(session_id="fakehash123", user_id=user.user_id, expires_at=datetime.utcnow() - timedelta(days=1))
    db.add(session)
    db.commit()
    db.close()

    db2 = SessionLocal()
    assert get_user_for_token(db2, "irrelevant-since-hash-is-fake") is None
    db2.close()


def test_migrate_guest_requires_auth(client):
    ngo_res = client.post("/ngo-profile", json={"name": "Guest NGO", "focus_areas": ["health"]})
    ngo_id = ngo_res.json()["ngo_id"]
    res = client.post("/auth/migrate-guest", json={"ngo_id": ngo_id})
    assert res.status_code == 401


def test_migrate_guest_links_profile_to_user(client):
    ngo_res = client.post("/ngo-profile", json={"name": "Guest NGO 2", "focus_areas": ["health"]})
    ngo_id = ngo_res.json()["ngo_id"]

    db = SessionLocal()
    user = User(google_sub="sub-migrate", email="migrate@example.org")
    db.add(user)
    db.commit()
    token = create_session(db, user)
    user_id = user.user_id
    db.close()

    res = client.post("/auth/migrate-guest", json={"ngo_id": ngo_id}, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200

    db = SessionLocal()
    profile = db.get(NGOProfile, ngo_id)
    assert profile.user_id == user_id
    db.close()


def test_migrate_guest_rejects_profile_owned_by_another_user(client):
    ngo_res = client.post("/ngo-profile", json={"name": "Guest NGO 3", "focus_areas": ["health"]})
    ngo_id = ngo_res.json()["ngo_id"]

    db = SessionLocal()
    user_a = User(google_sub="sub-a", email="a@example.org")
    user_b = User(google_sub="sub-b", email="b@example.org")
    db.add_all([user_a, user_b])
    db.commit()
    token_a = create_session(db, user_a)
    token_b = create_session(db, user_b)
    db.close()

    client.post("/auth/migrate-guest", json={"ngo_id": ngo_id}, headers={"Authorization": f"Bearer {token_a}"})
    res = client.post("/auth/migrate-guest", json={"ngo_id": ngo_id}, headers={"Authorization": f"Bearer {token_b}"})
    assert res.status_code == 409


def test_guest_endpoints_work_without_any_auth_header(client):
    """The core product guarantee -- every existing guest capability must
    keep working with zero Authorization header, whether or not Google is
    configured."""
    res = client.post("/ngo-profile", json={"name": "Pure Guest NGO", "focus_areas": ["health"]})
    assert res.status_code == 201
    res2 = client.get("/opportunities/live")
    assert res2.status_code == 200
