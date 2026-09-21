"""Tests for the email subscription / verification / unsubscribe API
(src/email/*, api/routers/subscribers.py). Uses the `capture_emails` fixture
(tests/conftest.py) to get the REAL raw token out of the email body exactly
as a real user would (by reading the link), since the token is deliberately
NOT recoverable from the database once hashed at rest (src/email/tokens.py)."""
from datetime import datetime, timedelta

from api.db.models import EmailLog, GrantSource, ImportedGrant, NGOProfile, Subscriber
from api.db.session import SessionLocal
from src.email.notify import notify_subscribers_of_new_grants
from tests.conftest import extract_token


def test_subscribe_creates_pending_subscriber_and_sends_verification_email(client, capture_emails):
    res = client.post("/subscribe", json={"email": "Lead@GreenEarth.org"})
    assert res.status_code == 201
    data = res.json()
    assert data["email"] == "lead@greenearth.org"  # normalized to lowercase
    assert data["status"] == "pending"
    assert data["is_verified"] is False
    assert data["is_unsubscribed"] is False
    assert data["notify_new_grants"] is True

    assert len(capture_emails.sent) == 1
    assert capture_emails.sent[0]["to"] == "lead@greenearth.org"
    assert "Confirm" in capture_emails.sent[0]["subject"]


def test_subscribe_rejects_invalid_email(client):
    res = client.post("/subscribe", json={"email": "not-an-email"})
    assert res.status_code == 422


def test_subscribe_with_unknown_ngo_id_returns_404(client):
    res = client.post("/subscribe", json={"email": "a@b.com", "ngo_id": "does-not-exist"})
    assert res.status_code == 404


def test_resubscribing_reactivates_rather_than_duplicating(client):
    res1 = client.post("/subscribe", json={"email": "repeat@example.com"})
    subscriber_id = res1.json()["subscriber_id"]

    res2 = client.post("/subscribe", json={"email": "repeat@example.com"})
    assert res2.status_code == 201
    assert res2.json()["subscriber_id"] == subscriber_id  # same row, not a new one


def test_verification_email_is_sent_exactly_once_for_an_unchanged_pending_subscriber(client, capture_emails):
    res = client.post("/subscribe", json={"email": "onlyone@example.com"})
    subscriber_id = res.json()["subscriber_id"]
    client.post("/subscribe", json={"email": "onlyone@example.com"})  # re-subscribe while still pending

    # Re-subscribing while already pending (not unsubscribed) does not spam
    # a second verification email -- the first one is still valid.
    assert len(capture_emails.sent) == 1

    db = SessionLocal()
    logs = db.query(EmailLog).filter_by(email_type="verification", subscriber_id=subscriber_id).all()
    db.close()
    assert len(logs) == 1
    assert logs[0].status == "sent_mock"


def test_resubscribing_after_unsubscribe_sends_a_fresh_verification_email(client, capture_emails):
    res = client.post("/subscribe", json={"email": "backagain@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    db = SessionLocal()
    sub = db.get(Subscriber, subscriber_id)
    sub.status = Subscriber.STATUS_UNSUBSCRIBED
    db.commit()
    db.close()

    client.post("/subscribe", json={"email": "backagain@example.com"})
    fetched = client.get(f"/subscribers/{subscriber_id}").json()
    assert fetched["status"] == "pending"  # NOT silently restored to verified
    assert len(capture_emails.sent) == 2  # original + the fresh one on re-subscribe


def test_unsubscribe_flow(client, capture_emails):
    res = client.post("/subscribe", json={"email": "leaving@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    unsubscribe_token = extract_token(capture_emails.sent[0]["text"], "unsubscribe_token")
    unsub_res = client.get(f"/subscribers/unsubscribe?token={unsubscribe_token}")
    assert unsub_res.status_code == 200
    assert unsub_res.json()["status"] == "unsubscribed"

    fetched = client.get(f"/subscribers/{subscriber_id}").json()
    assert fetched["status"] == "unsubscribed"
    assert fetched["is_unsubscribed"] is True


def test_unsubscribe_invalid_token_404(client):
    res = client.get("/subscribers/unsubscribe?token=not-a-real-token")
    assert res.status_code == 404


def test_verify_email_flow(client, capture_emails):
    res = client.post("/subscribe", json={"email": "verifyme@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    raw_token = extract_token(capture_emails.sent[0]["text"], "verify_token")
    verify_res = client.get(f"/subscribers/verify?token={raw_token}")
    assert verify_res.status_code == 200

    fetched = client.get(f"/subscribers/{subscriber_id}").json()
    assert fetched["status"] == "verified"
    assert fetched["is_verified"] is True


def test_verify_token_is_single_use(client, capture_emails):
    client.post("/subscribe", json={"email": "onceonly@example.com"})
    raw_token = extract_token(capture_emails.sent[0]["text"], "verify_token")

    first = client.get(f"/subscribers/verify?token={raw_token}")
    assert first.status_code == 200

    second = client.get(f"/subscribers/verify?token={raw_token}")
    assert second.status_code == 410
    assert "already been used" in second.json()["detail"]


def test_verify_token_expiry_is_enforced(client, capture_emails):
    res = client.post("/subscribe", json={"email": "expired@example.com"})
    subscriber_id = res.json()["subscriber_id"]
    raw_token = extract_token(capture_emails.sent[0]["text"], "verify_token")

    # Force the stored expiry into the past -- simulates a link older than 24h.
    db = SessionLocal()
    sub = db.get(Subscriber, subscriber_id)
    sub.verify_token_expires_at = datetime.utcnow() - timedelta(hours=1)
    db.commit()
    db.close()

    res = client.get(f"/subscribers/verify?token={raw_token}")
    assert res.status_code == 410
    assert "expired" in res.json()["detail"]


def test_verify_unknown_token_404(client):
    res = client.get("/subscribers/verify?token=totally-made-up")
    assert res.status_code == 404


def test_resend_verification_sends_a_new_email(client, capture_emails):
    res = client.post("/subscribe", json={"email": "resend@example.com"})
    subscriber_id = res.json()["subscriber_id"]
    assert len(capture_emails.sent) == 1

    # Force the rate-limit window to have already elapsed so the resend isn't blocked.
    db = SessionLocal()
    sub = db.get(Subscriber, subscriber_id)
    sub.last_verification_email_sent_at = datetime.utcnow() - timedelta(minutes=5)
    db.commit()
    db.close()

    resend_res = client.post(f"/subscribers/{subscriber_id}/resend-verification")
    assert resend_res.status_code == 202
    assert len(capture_emails.sent) == 2

    # The OLD token must no longer work -- resending issues a fresh one.
    old_token = extract_token(capture_emails.sent[0]["text"], "verify_token")
    new_token = extract_token(capture_emails.sent[1]["text"], "verify_token")
    assert old_token != new_token
    assert client.get(f"/subscribers/verify?token={old_token}").status_code == 404
    assert client.get(f"/subscribers/verify?token={new_token}").status_code == 200


def test_resend_verification_is_rate_limited(client, capture_emails):
    res = client.post("/subscribe", json={"email": "ratelimited@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    # Immediately resending (within MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS
    # of the original send) must be refused, not silently spam a second email.
    resend_res = client.post(f"/subscribers/{subscriber_id}/resend-verification")
    assert resend_res.status_code == 429
    assert len(capture_emails.sent) == 1  # still just the original


def test_resend_verification_on_already_verified_subscriber_is_a_no_op(client, capture_emails):
    res = client.post("/subscribe", json={"email": "alreadyverified@example.com"})
    subscriber_id = res.json()["subscriber_id"]
    raw_token = extract_token(capture_emails.sent[0]["text"], "verify_token")
    client.get(f"/subscribers/verify?token={raw_token}")

    resend_res = client.post(f"/subscribers/{subscriber_id}/resend-verification")
    assert resend_res.status_code == 200
    assert resend_res.json()["status"] == "already_verified"
    assert len(capture_emails.sent) == 1  # no new email sent


def test_resend_verification_unknown_subscriber_404(client):
    res = client.post("/subscribers/does-not-exist/resend-verification")
    assert res.status_code == 404


def test_get_unknown_subscriber_404(client):
    res = client.get("/subscribers/does-not-exist")
    assert res.status_code == 404


def test_update_preferences(client):
    res = client.post("/subscribe", json={"email": "prefs@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    update_res = client.patch(
        f"/subscribers/{subscriber_id}/preferences",
        json={"notify_new_grants": False, "notify_digest": True},
    )
    assert update_res.status_code == 200
    data = update_res.json()
    assert data["notify_new_grants"] is False
    assert data["notify_digest"] is True


def test_update_preferences_unknown_ngo_id_404(client):
    res = client.post("/subscribe", json={"email": "prefs2@example.com"})
    subscriber_id = res.json()["subscriber_id"]

    update_res = client.patch(
        f"/subscribers/{subscriber_id}/preferences",
        json={"ngo_id": "does-not-exist"},
    )
    assert update_res.status_code == 404


def test_lookup_subscriber_by_email(client):
    client.post("/subscribe", json={"email": "findme@example.com"})
    res = client.get("/subscribers/by-email/lookup?email=findme@example.com")
    assert res.status_code == 200
    assert res.json()["email"] == "findme@example.com"


def test_lookup_subscriber_by_unknown_email_404(client):
    res = client.get("/subscribers/by-email/lookup?email=nobody@example.com")
    assert res.status_code == 404


def test_verify_token_is_never_exposed_in_any_api_response(client):
    """Security regression test: the raw AND hashed verify_token must never
    appear in any subscriber-facing JSON response."""
    res = client.post("/subscribe", json={"email": "notoken@example.com"})
    assert "verify_token" not in res.text
    subscriber_id = res.json()["subscriber_id"]
    get_res = client.get(f"/subscribers/{subscriber_id}")
    assert "verify_token" not in get_res.text


def test_verification_email_html_is_a_real_branded_document(client, capture_emails):
    """The redesigned template (src/email/templates.py) must actually be a
    full, styled HTML document -- not a bare handful of <p> tags -- and
    must never leak raw None/null/undefined placeholders for missing data."""
    client.post("/subscribe", json={"email": "template-check@example.com"})
    html_body = capture_emails.sent[0]["html"]
    assert html_body is not None
    assert "<!DOCTYPE html>" in html_body
    assert "GrantSetu" in html_body
    assert "Unsubscribe" in html_body
    for bad in ["None", "null", "undefined", "NaN"]:
        assert bad not in html_body


def _seed_ngo_and_verified_subscriber(client, email: str, focus_areas: list[str]) -> tuple[str, str]:
    ngo_res = client.post(
        "/ngo-profile",
        json={"name": "Digest Test NGO", "registration_type": "trust", "focus_areas": focus_areas},
    )
    ngo_id = ngo_res.json()["ngo_id"]
    sub_res = client.post("/subscribe", json={"email": email, "ngo_id": ngo_id})
    subscriber_id = sub_res.json()["subscriber_id"]
    db = SessionLocal()
    sub = db.get(Subscriber, subscriber_id)
    sub.status = Subscriber.STATUS_VERIFIED
    db.commit()
    db.close()
    return ngo_id, subscriber_id


def test_multiple_matching_new_grants_produce_one_digest_email_not_several(client, capture_emails):
    """The core fix this batching change guards against: a sync that
    discovers several new grants matching the same subscriber must send
    ONE email listing all of them (a digest), never one email per grant --
    that used to mean a subscriber's inbox got hit with a burst of
    near-identical emails from a single sync run."""
    _, subscriber_id = _seed_ngo_and_verified_subscriber(client, "digest-batch@example.com", ["health"])
    capture_emails.sent.clear()  # drop the verification email the subscribe call above just sent

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    source = GrantSource(name="Digest Batch Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)

    grants = [
        ImportedGrant(
            source_id=source.source_id, external_id=f"digest-{i}", title=f"Health Grant {i}",
            category="Health", status="open", dedup_hash=f"digest-hash-{i}", is_sample_data=False,
        )
        for i in range(3)
    ]
    db.add_all(grants)
    db.commit()

    logs = notify_subscribers_of_new_grants(db, grants)
    db.close()

    # Other subscribers seeded by other tests sharing this session's SQLite
    # file (see tests/conftest.py) may also legitimately match these
    # grants -- scope every assertion to just this test's own subscriber.
    matching_sent = [e for e in capture_emails.sent if e["to"] == "digest-batch@example.com"]
    assert len(matching_sent) == 1  # one digest email, not three
    matching_logs = [log for log in logs if log.subscriber_id == subscriber_id]
    assert len(matching_logs) == 3  # dedup/idempotency bookkeeping stays per-grant

    html_body = matching_sent[0]["html"]
    for i in range(3):
        assert f"Health Grant {i}" in html_body
    assert "3 new funding opportunities" in matching_sent[0]["subject"]

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()


def test_digest_email_omits_missing_fields_instead_of_showing_placeholders(client, capture_emails):
    """A grant with no funder/deadline/description must not render 'N/A',
    'None', or an empty label in the email -- the field is simply absent."""
    _seed_ngo_and_verified_subscriber(client, "digest-missing@example.com", ["education"])
    capture_emails.sent.clear()  # drop the verification email the subscribe call above just sent

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    source = GrantSource(name="Digest Missing-Fields Source", connector_type="sample_fixture")
    db.add(source)
    db.commit()
    db.refresh(source)
    grant = ImportedGrant(
        source_id=source.source_id, external_id="digest-missing-1", title="Bare Education Grant",
        category="Education", status="open", dedup_hash="digest-missing-hash", is_sample_data=False,
        # funder, deadline, description, eligibility_text, country all left unset/default.
    )
    db.add(grant)
    db.commit()

    notify_subscribers_of_new_grants(db, [grant])
    db.close()

    sent = [e for e in capture_emails.sent if e["to"] == "digest-missing@example.com"]
    assert len(sent) == 1
    html_body = sent[0]["html"]
    assert "Bare Education Grant" in html_body
    for bad in ["None", "null", "undefined", "N/A", "Funder: <", "Deadline <"]:
        assert bad not in html_body

    db = SessionLocal()
    db.query(ImportedGrant).delete()
    db.query(GrantSource).delete()
    db.commit()
    db.close()
