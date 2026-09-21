"""Google OAuth 2.0 / OpenID Connect -- the authorization-code flow, used
ONLY to let a user optionally persist their GrantSetu data (see interview.txt
Section 16-17: "SIGN-IN MUST NOT BE REQUIRED"; every feature works for a
guest with no Google account at all).

Reads GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET / GOOGLE_REDIRECT_URI from the
environment -- never hard-coded, never invented. `is_configured()` returns
False (not a fake success) when they're unset, so the rest of the app can
degrade to "Google sign-in unavailable" instead of crashing.

The ID token's signature is verified against Google's own published JWKS
(fetched/cached by PyJWT's PyJWKClient) -- never decoded-but-unverified, and
never trusted merely because it round-tripped through Google's token
endpoint over HTTPS. `sub` (Google's stable per-account identifier) is the
key used to find/create a User row -- never email, which a Google account
holder can change.
"""
import logging
import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import httpx
import jwt
from jwt import PyJWKClient

logger = logging.getLogger("grantsetu.auth.google")

AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
JWKS_URL = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
REQUEST_TIMEOUT = 10.0

_jwk_client: PyJWKClient | None = None


def _get_jwk_client() -> PyJWKClient:
    global _jwk_client
    if _jwk_client is None:
        _jwk_client = PyJWKClient(JWKS_URL)
    return _jwk_client


def _client_id() -> str:
    return os.environ.get("GOOGLE_CLIENT_ID", "").strip()


def _client_secret() -> str:
    return os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()


def _redirect_uri() -> str:
    return os.environ.get("GOOGLE_REDIRECT_URI", "").strip()


def is_configured() -> bool:
    return bool(_client_id() and _client_secret() and _redirect_uri())


def generate_state() -> str:
    """CSRF protection for the OAuth redirect -- the caller persists this
    (hashed, single-use, short-lived -- see api/routers/auth.py) and
    verifies it matches on callback before exchanging the code."""
    return secrets.token_urlsafe(24)


def build_authorization_url(state: str) -> str:
    params = {
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}"


@dataclass
class GoogleIdentity:
    sub: str
    email: str
    name: str | None
    picture: str | None


class GoogleAuthError(RuntimeError):
    """Raised for any real failure in the exchange/verify flow -- always
    caught at the router boundary and turned into an honest error response,
    never silently treated as success."""


def exchange_code_for_identity(code: str) -> GoogleIdentity:
    """Exchanges an authorization `code` for tokens, then verifies the
    returned ID token's signature/issuer/audience/expiry against Google's
    real JWKS before trusting ANY claim in it. Raises GoogleAuthError on any
    failure -- never returns a partially-trusted identity."""
    try:
        response = httpx.post(
            TOKEN_ENDPOINT,
            data={
                "code": code,
                "client_id": _client_id(),
                "client_secret": _client_secret(),
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        token_data = response.json()
    except httpx.HTTPError as exc:
        raise GoogleAuthError(f"Google token exchange failed: {exc}") from exc

    id_token = token_data.get("id_token")
    if not id_token:
        raise GoogleAuthError("Google did not return an id_token")

    try:
        signing_key = _get_jwk_client().get_signing_key_from_jwt(id_token)
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=_client_id(),
            options={"require": ["exp", "iat", "sub", "email"]},
        )
    except jwt.PyJWTError as exc:
        raise GoogleAuthError(f"Google ID token verification failed: {exc}") from exc

    if claims.get("iss") not in GOOGLE_ISSUERS:
        raise GoogleAuthError(f"Unexpected ID token issuer: {claims.get('iss')!r}")
    if not claims.get("email_verified", True):
        # Google sets this False only for non-Google-verified emails on
        # some legacy account types -- refuse rather than trust an
        # unverified address as this user's identity.
        raise GoogleAuthError("Google account email is not verified")

    return GoogleIdentity(
        sub=claims["sub"],
        email=claims["email"],
        name=claims.get("name"),
        picture=claims.get("picture"),
    )
