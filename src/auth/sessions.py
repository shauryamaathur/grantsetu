"""Session token issuance/validation for signed-in users.

Same hashed-token-at-rest pattern as src/email/tokens.py's email
verification tokens: the raw session token is high-entropy
(secrets.token_urlsafe) and returned to the browser exactly once (at
login), which stores it (localStorage) and sends it back on every request
via a header. Only its SHA-256 hash is ever persisted (UserSession.
session_id) -- a database read alone can never be used to impersonate a
signed-in user.
"""
import hashlib
import secrets
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from api.db.models import User, UserSession

SESSION_TOKEN_BYTES = 32  # 256 bits of entropy
SESSION_TTL = timedelta(days=30)


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def create_session(db: Session, user: User) -> str:
    """Returns the RAW token -- the only time it ever exists outside the
    caller's response. Never logged, never stored as-is."""
    raw = secrets.token_urlsafe(SESSION_TOKEN_BYTES)
    session = UserSession(
        session_id=_hash_token(raw),
        user_id=user.user_id,
        expires_at=datetime.utcnow() + SESSION_TTL,
    )
    db.add(session)
    db.commit()
    return raw


def get_user_for_token(db: Session, raw_token: str | None) -> User | None:
    """Never raises -- an absent/invalid/expired token simply means "not
    signed in" (the caller falls back to guest behavior), not an error."""
    if not raw_token:
        return None
    session = db.get(UserSession, _hash_token(raw_token))
    if session is None:
        return None
    if session.expires_at < datetime.utcnow():
        db.delete(session)
        db.commit()
        return None
    return db.get(User, session.user_id)


def revoke_session(db: Session, raw_token: str | None) -> None:
    if not raw_token:
        return
    session = db.get(UserSession, _hash_token(raw_token))
    if session is not None:
        db.delete(session)
        db.commit()
