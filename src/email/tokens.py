"""Secure token generation and hashing for email verification.

Verification tokens are single-use, short-lived proof that a specific
person controls a specific inbox -- the classic case for hashing at rest:
if the database were ever read (backup leak, SQL injection, etc.), a raw
token would let an attacker complete verification for a victim's email.
Hashing means the database alone is useless for that -- the attacker would
also need to have intercepted the actual email.

Unsubscribe tokens are a DIFFERENT kind of secret (see Subscriber's
docstring in api/db/models.py): a long-lived capability embedded in every
future alert email, not a one-time proof, so they are generated with high
entropy but NOT hashed here -- the raw value must remain usable to build
links in emails sent months from now.
"""
import hashlib
import secrets
from datetime import datetime, timedelta

VERIFICATION_TOKEN_BYTES = 32  # 256 bits of entropy
VERIFICATION_TOKEN_TTL = timedelta(hours=24)

MIN_SECONDS_BETWEEN_VERIFICATION_EMAILS = 60  # basic resend rate limit


def generate_verification_token() -> tuple[str, str, datetime]:
    """Returns (raw_token, token_hash, expires_at). The raw token is what
    goes in the email link; only the hash is ever persisted."""
    raw = secrets.token_urlsafe(VERIFICATION_TOKEN_BYTES)
    token_hash = hash_token(raw)
    expires_at = datetime.utcnow() + VERIFICATION_TOKEN_TTL
    return raw, token_hash, expires_at


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_unsubscribe_token() -> str:
    return secrets.token_urlsafe(32)
