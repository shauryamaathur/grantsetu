from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from api.db.models import NGOProfile, Subscriber
from api.db.session import SessionLocal, get_db
from api.models.schemas import (
    SubscribeRequest,
    SubscriberPreferencesUpdate,
    SubscriberResponse,
)
from src.email.notify import VerificationRateLimited, send_verification_email
from src.email.tokens import hash_token

router = APIRouter(tags=["subscribers"])


def _send_verification_email_task(subscriber_id: str, force: bool):
    """Wrapper used as the BackgroundTasks target. Deliberately opens its
    OWN database session rather than reusing the request's `Depends(get_db)`
    session -- a real bug, found via testing, not assumed: passing the
    request-scoped session into a background task meant its mutations
    (storing the verify_token hash/expiry) were silently lost, because that
    session is torn down (`get_db()`'s `finally: db.close()`) around the
    same time the background task runs, and a commit against an already-
    closed/returned connection does not durably persist. Every background-
    task email sender in this module follows this same fresh-session
    pattern for that reason."""
    db = SessionLocal()
    try:
        subscriber = db.get(Subscriber, subscriber_id)
        if subscriber is not None:
            send_verification_email(db, subscriber, force=force)
    finally:
        db.close()


@router.post("/subscribe", response_model=SubscriberResponse, status_code=201)
def subscribe(payload: SubscribeRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Step 1-5 of the verification flow (Phase 3): validate/normalize the
    email (done by SubscribeRequest's Pydantic validator), store a PENDING
    subscription, and send a verification email as a non-blocking
    background task -- the HTTP response returns immediately regardless of
    whether the send succeeds; the real result is recorded in EmailLog and
    visible via GET /subscribers/{id}."""
    if payload.ngo_id and db.get(NGOProfile, payload.ngo_id) is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")

    existing = db.query(Subscriber).filter_by(email=payload.email).first()
    if existing:
        if payload.ngo_id:
            existing.ngo_id = payload.ngo_id
        if existing.status == Subscriber.STATUS_UNSUBSCRIBED:
            # Re-subscribing after unsubscribing is treated as a NEW consent
            # event, not silently restored -- back to pending, and a fresh
            # verification email is required, exactly like a new signup.
            existing.status = Subscriber.STATUS_PENDING
            db.commit()
            background_tasks.add_task(_send_verification_email_task, existing.subscriber_id, True)
        else:
            db.commit()
        db.refresh(existing)
        return existing

    subscriber = Subscriber(email=payload.email, ngo_id=payload.ngo_id, status=Subscriber.STATUS_PENDING)
    db.add(subscriber)
    db.commit()
    db.refresh(subscriber)

    background_tasks.add_task(_send_verification_email_task, subscriber.subscriber_id, True)

    return subscriber


@router.post("/subscribers/{subscriber_id}/resend-verification")
def resend_verification(subscriber_id: str, response: Response, db: Session = Depends(get_db)):
    subscriber = db.get(Subscriber, subscriber_id)
    if subscriber is None:
        raise HTTPException(status_code=404, detail=f"Subscriber '{subscriber_id}' not found")
    if subscriber.status == Subscriber.STATUS_VERIFIED:
        response.status_code = 200  # nothing was queued -- distinct from 202 "an email is being sent"
        return {"status": "already_verified"}

    try:
        send_verification_email(db, subscriber)
    except VerificationRateLimited as exc:
        raise HTTPException(
            status_code=429,
            detail=f"A verification email was already sent recently. Try again in {exc.retry_after_seconds} seconds.",
        )
    response.status_code = 202
    return {"status": "verification_email_sent"}


@router.get("/subscribers/verify", tags=["subscribers"])
def verify_email(token: str, db: Session = Depends(get_db)):
    """Looks up the subscriber by the HASH of the provided raw token (the
    raw token is never stored -- see src/email/tokens.py). Enforces expiry
    and single-use explicitly, with a distinct, honest message for each
    failure mode rather than one generic "invalid token" that would make
    "your link expired" indistinguishable from "you already used this"."""
    token_hash = hash_token(token)
    subscriber = db.query(Subscriber).filter_by(verify_token=token_hash).first()
    if subscriber is None:
        raise HTTPException(status_code=404, detail="Invalid verification link.")
    if subscriber.verify_token_used_at is not None:
        raise HTTPException(status_code=410, detail="This verification link has already been used.")
    if subscriber.verify_token_expires_at is None or subscriber.verify_token_expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="This verification link has expired. Request a new one.")

    subscriber.status = Subscriber.STATUS_VERIFIED
    subscriber.verify_token_used_at = datetime.utcnow()
    db.commit()
    return {"status": "verified", "email": subscriber.email}


@router.get("/subscribers/unsubscribe", tags=["subscribers"])
@router.post("/subscribers/unsubscribe", tags=["subscribers"])
def unsubscribe(token: str, db: Session = Depends(get_db)):
    subscriber = db.query(Subscriber).filter_by(unsubscribe_token=token).first()
    if subscriber is None:
        raise HTTPException(status_code=404, detail="Invalid unsubscribe link.")
    subscriber.status = Subscriber.STATUS_UNSUBSCRIBED
    db.commit()
    return {"status": "unsubscribed", "email": subscriber.email}


@router.get("/subscribers/{subscriber_id}", response_model=SubscriberResponse)
def get_subscriber(subscriber_id: str, db: Session = Depends(get_db)):
    subscriber = db.get(Subscriber, subscriber_id)
    if subscriber is None:
        raise HTTPException(status_code=404, detail=f"Subscriber '{subscriber_id}' not found")
    return subscriber


@router.patch("/subscribers/{subscriber_id}/preferences", response_model=SubscriberResponse)
def update_preferences(subscriber_id: str, payload: SubscriberPreferencesUpdate, db: Session = Depends(get_db)):
    subscriber = db.get(Subscriber, subscriber_id)
    if subscriber is None:
        raise HTTPException(status_code=404, detail=f"Subscriber '{subscriber_id}' not found")

    if payload.ngo_id is not None:
        if db.get(NGOProfile, payload.ngo_id) is None:
            raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")
        subscriber.ngo_id = payload.ngo_id
    if payload.notify_new_grants is not None:
        subscriber.notify_new_grants = payload.notify_new_grants
    if payload.notify_digest is not None:
        subscriber.notify_digest = payload.notify_digest

    db.commit()
    db.refresh(subscriber)
    return subscriber


@router.get("/subscribers/by-email/lookup", response_model=SubscriberResponse)
def lookup_by_email(email: str = Query(...), db: Session = Depends(get_db)):
    subscriber = db.query(Subscriber).filter_by(email=email.strip().lower()).first()
    if subscriber is None:
        raise HTTPException(status_code=404, detail="No subscriber found for that email")
    return subscriber
