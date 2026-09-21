"""Sends emails through the configured provider AND records every attempt in
EmailLog. `dedup_key` guarantees a subscriber is never sent the same LOGICAL
grant-alert twice (e.g. two syncs discovering the same "new" grant must not
double-notify someone) -- verification emails are handled differently (see
send_verification_email) since resending one is a legitimate, expected
action, not a duplicate to suppress.
"""
import logging
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from api.db.models import EmailLog, ImportedGrant, NGOProfile, Subscriber
from src.email.service import get_email_provider
from src.email.templates import _frontend_base_url, new_grant_digest_email, verification_email
from src.email.tokens import MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS, generate_verification_token

logger = logging.getLogger("grantsetu.notify")


class VerificationRateLimited(Exception):
    """Raised when a verification email was requested too soon after the
    last one for this subscriber. Callers should turn this into an HTTP 429."""

    def __init__(self, retry_after_seconds: int):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"Retry after {retry_after_seconds}s")


def send_verification_email(db: Session, subscriber: Subscriber, *, force: bool = False) -> EmailLog:
    """Generates a fresh, single-use, expiring verification token (see
    src/email/tokens.py), stores only its HASH on the subscriber, and sends
    the email containing the raw token. Safe to call multiple times (e.g. a
    "resend verification" action) -- each call is rate-limited to at most
    once per MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS unless `force=True`
    (used for the very first send at subscribe time, which has nothing to
    rate-limit against yet)."""
    now = datetime.utcnow()
    if not force and subscriber.last_verification_email_sent_at is not None:
        elapsed = (now - subscriber.last_verification_email_sent_at).total_seconds()
        if elapsed < MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS:
            raise VerificationRateLimited(retry_after_seconds=int(MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS - elapsed))

    raw_token, token_hash, expires_at = generate_verification_token()
    subscriber.verify_token = token_hash
    subscriber.verify_token_expires_at = expires_at
    subscriber.verify_token_used_at = None
    subscriber.last_verification_email_sent_at = now

    subject, text_body, html_body = verification_email(subscriber.email, raw_token, subscriber.unsubscribe_token)
    provider = get_email_provider()
    result = provider.send(subscriber.email, subject, text_body, html_body)

    log = EmailLog(
        subscriber_id=subscriber.subscriber_id,
        email_type="verification",
        subject=subject,
        # Each verification send is a distinct, legitimate event (resends
        # are expected), so the dedup key is unique per attempt rather than
        # per-subscriber -- it exists here for a consistent audit trail
        # (Phase 3's "event records for ... email sent, and email failed"),
        # not to block resends the way new_grant_alert's dedup_key does.
        dedup_key=f"{subscriber.subscriber_id}:verification:{uuid.uuid4()}",
        status=result.status,
        provider=result.provider,
        error=result.error,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def _matched_focus_areas(grant: ImportedGrant, focus_areas: list[str]) -> list[str]:
    """Which of the NGO's own focus-area terms actually appear in this
    grant's text -- real, checkable evidence for the digest email's "why
    this matches" line, never an invented reason."""
    haystack = f"{grant.title} {grant.category or ''}".lower()
    return [area for area in focus_areas if area.lower() in haystack]


def _grant_to_card_dict(grant: ImportedGrant, matched_focus_areas: list[str]) -> dict:
    return {
        "title": grant.title,
        "funder": grant.funder,
        "deadline": grant.deadline,
        "country": grant.country,
        "description": grant.description,
        "eligibility_text": grant.eligibility_text,
        "url": grant.url,
        "detail_link": f"{_frontend_base_url()}/index.html?view=detail&src=imported&ref={grant.imported_grant_id}",
        "is_sample_data": grant.is_sample_data,
        "matched_focus_areas": matched_focus_areas,
    }


def notify_subscribers_of_new_grants(db: Session, new_grants: list[ImportedGrant]) -> list[EmailLog]:
    """Notifies every VERIFIED, non-unsubscribed subscriber whose linked NGO
    profile's focus_areas plausibly match any newly-imported grant -- or,
    if a subscriber has no linked NGO profile, about everything (they
    explicitly asked for new-grant alerts with no profile to filter
    against). Skips anyone not yet verified (Phase 3, requirement 8: "Only
    verified subscriptions receive grant alerts"), anyone unsubscribed/
    bounced/disabled, anyone with alerts turned off, and any (subscriber,
    grant) pair already logged.

    Sends AT MOST ONE EMAIL PER SUBSCRIBER per call, covering every one of
    their new matches as opportunity cards in a single digest -- not one
    email per grant. A sync that discovers 5 new grants matching the same
    subscriber used to mean 5 separate emails landing at once, which reads
    as spam rather than a polished product notification; batching by
    subscriber is what new_grant_digest_email (src/email/templates.py) is
    built for. EmailLog rows stay per-(subscriber, grant) for the existing
    dedup/idempotency guarantee -- only the actual SEND is batched."""
    if not new_grants:
        return []

    logs: list[EmailLog] = []
    subscribers = (
        db.query(Subscriber)
        .filter(Subscriber.status == Subscriber.STATUS_VERIFIED, Subscriber.notify_new_grants.is_(True))
        .all()
    )
    provider = get_email_provider()

    for subscriber in subscribers:
        ngo = db.get(NGOProfile, subscriber.ngo_id) if subscriber.ngo_id else None
        focus_areas = ngo.focus_areas if ngo else []

        to_notify: list[tuple[ImportedGrant, list[str]]] = []
        for grant in new_grants:
            matched = _matched_focus_areas(grant, focus_areas) if focus_areas else []
            if focus_areas and not matched:
                continue  # doesn't match this subscriber's NGO focus areas

            dedup_key = f"{subscriber.subscriber_id}:new_grant_alert:{grant.imported_grant_id}"
            if db.query(EmailLog).filter_by(dedup_key=dedup_key).first():
                continue  # already notified this subscriber about this grant -- idempotency

            to_notify.append((grant, matched))

        if not to_notify:
            continue

        # Soonest deadline first -- the most actionable opportunities lead
        # the digest; grants with no deadline sort last, not first.
        to_notify.sort(key=lambda pair: pair[0].deadline or datetime.max)

        subject, text_body, html_body = new_grant_digest_email(
            subscriber.email,
            [_grant_to_card_dict(grant, matched) for grant, matched in to_notify],
            subscriber.unsubscribe_token,
        )
        result = provider.send(subscriber.email, subject, text_body, html_body)

        for grant, _matched in to_notify:
            log = EmailLog(
                subscriber_id=subscriber.subscriber_id,
                email_type="new_grant_alert",
                subject=subject,
                dedup_key=f"{subscriber.subscriber_id}:new_grant_alert:{grant.imported_grant_id}",
                status=result.status,
                provider=result.provider,
                error=result.error,
            )
            db.add(log)
            logs.append(log)

    db.commit()
    for log in logs:
        db.refresh(log)
    return logs
