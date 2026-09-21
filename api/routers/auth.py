"""Optional Google sign-in. Guest usage of GrantSetu never depends on
anything in this file working -- see interview.txt Section 16-17: signing
in provides PERSISTENCE (an NGO profile/saved opportunities tied to a
durable account instead of one browser's localStorage), never permission.
`GET /auth/config` lets the frontend hide the "Sign in with Google" button
entirely when GOOGLE_CLIENT_ID/SECRET/REDIRECT_URI aren't set, rather than
showing a button that would 400.
"""
import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.db.models import NGOProfile, User
from api.db.session import get_db
from api.models.schemas import AuthConfigResponse, MigrateGuestRequest, UserResponse
from src.auth import google_oauth
from src.auth.sessions import create_session, get_user_for_token, revoke_session

logger = logging.getLogger("grantsetu.auth")

router = APIRouter(prefix="/auth", tags=["auth"])

# In-process CSRF-state store for the OAuth redirect round-trip: short-lived
# (10 minutes), single-use, cleared on verification. A module-level dict is
# a deliberate, documented trade-off appropriate to this project's
# single-process SQLite deployment model (see api/db/session.py) -- a
# multi-worker production deployment would need a shared store (e.g. the
# database or Redis) instead.
_pending_states: dict[str, datetime] = {}
_STATE_TTL = timedelta(minutes=10)


def _cleanup_states():
    now = datetime.utcnow()
    expired = [s for s, exp in _pending_states.items() if exp < now]
    for s in expired:
        del _pending_states[s]


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """Returns the signed-in User, or None for a guest -- NEVER raises for a
    missing/invalid token, since almost every endpoint in this app must work
    identically for guests. Endpoints that genuinely require sign-in
    (POST /auth/migrate-guest) check for None themselves and 401."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    raw_token = authorization[7:].strip()
    return get_user_for_token(db, raw_token)


@router.get("/config", response_model=AuthConfigResponse)
def auth_config():
    return AuthConfigResponse(google_enabled=google_oauth.is_configured())


@router.get("/google/login")
def google_login():
    if not google_oauth.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Google sign-in is not configured on this server. GrantSetu works fully as a guest without it.",
        )
    _cleanup_states()
    state = google_oauth.generate_state()
    _pending_states[state] = datetime.utcnow() + _STATE_TTL
    return RedirectResponse(google_oauth.build_authorization_url(state))


@router.get("/google/callback")
def google_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    import os

    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:5500").rstrip("/")

    if error:
        return RedirectResponse(f"{frontend_url}/?auth_error={error}")
    if not code or not state:
        return RedirectResponse(f"{frontend_url}/?auth_error=missing_code_or_state")

    _cleanup_states()
    if state not in _pending_states:
        return RedirectResponse(f"{frontend_url}/?auth_error=invalid_or_expired_state")
    del _pending_states[state]  # single-use

    try:
        identity = google_oauth.exchange_code_for_identity(code)
    except google_oauth.GoogleAuthError as exc:
        logger.warning("Google sign-in failed: %s", exc)
        return RedirectResponse(f"{frontend_url}/?auth_error=verification_failed")

    user = db.query(User).filter_by(google_sub=identity.sub).first()
    if user is None:
        user = User(google_sub=identity.sub, email=identity.email, name=identity.name, picture_url=identity.picture)
        db.add(user)
    else:
        user.email = identity.email
        user.name = identity.name
        user.picture_url = identity.picture
        user.last_login_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    token = create_session(db, user)
    # The token travels in the URL FRAGMENT (#...), not a query string --
    # fragments are never sent to the server on subsequent requests and
    # never appear in server access logs or the Referer header, unlike a
    # query parameter would.
    return RedirectResponse(f"{frontend_url}/#session_token={token}")


@router.post("/logout")
def logout(authorization: str | None = Header(default=None), db: Session = Depends(get_db)):
    if authorization and authorization.lower().startswith("bearer "):
        revoke_session(db, authorization[7:].strip())
    return {"status": "ok"}


@router.get("/me", response_model=UserResponse)
def me(user: User | None = Depends(get_current_user)):
    if user is None:
        raise HTTPException(status_code=401, detail="Not signed in")
    return user


@router.post("/migrate-guest", response_model=UserResponse)
def migrate_guest(
    payload: MigrateGuestRequest,
    user: User | None = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Associates a guest-created NGOProfile with the signed-in user --
    ONLY on this explicit, user-confirmed call (see interview.txt Section
    18: "Allow the user to explicitly confirm... Do not silently upload or
    associate personal information."). Never triggered automatically on
    sign-in."""
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in with Google first")

    profile = db.get(NGOProfile, payload.ngo_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{payload.ngo_id}' not found")
    if profile.user_id is not None and profile.user_id != user.user_id:
        raise HTTPException(status_code=409, detail="This NGO profile is already linked to a different account")

    profile.user_id = user.user_id
    db.commit()
    return user
