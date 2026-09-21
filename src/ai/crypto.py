"""Encryption for AI provider API keys at rest.

Uses Fernet (AES-128-CBC + HMAC, from the `cryptography` package) with a key
derived from the `GRANTSETU_ENCRYPTION_KEY` environment variable. This is
"encrypted at rest in the database" -- meaningfully better than plaintext --
but it is NOT a full secrets-management solution: the encryption key itself
lives in an environment variable on the same machine as the database, so
anyone with access to both the process environment and the DB file can still
decrypt. A real production deployment should use a managed secrets store
(e.g. AWS KMS/Secrets Manager, HashiCorp Vault) instead -- documented
honestly here and in README/interview.txt rather than overstating what this
provides.

If `GRANTSETU_ENCRYPTION_KEY` is not set, a key is generated at process
startup and used for that run only (logged as a warning) -- this keeps local
development working out of the box, but means keys encrypted in one process
run CANNOT be decrypted after a restart unless the same key is reused. Set
`GRANTSETU_ENCRYPTION_KEY` explicitly for anything beyond a single local
session.
"""
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger("grantsetu.crypto")

_fernet: Fernet | None = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is not None:
        return _fernet

    key = os.environ.get("GRANTSETU_ENCRYPTION_KEY", "").strip()
    if not key:
        key = Fernet.generate_key().decode()
        logger.warning(
            "GRANTSETU_ENCRYPTION_KEY is not set -- generated a temporary encryption key "
            "for this process only. AI provider API keys encrypted now will NOT be "
            "decryptable after a restart. Set GRANTSETU_ENCRYPTION_KEY in .env for "
            "anything beyond a single local session."
        )
    try:
        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
    except (ValueError, TypeError) as exc:
        # A set-but-invalid key (e.g. the .env.example placeholder
        # "YOUR_FERNET_ENCRYPTION_KEY_HERE" left in place) reaches here as a
        # cryptic "Fernet key must be 32 url-safe base64-encoded bytes."
        # from the cryptography library. Re-raise with the exact env var
        # name and the exact command to fix it, so this fails with an
        # actionable message instead of an opaque 500.
        raise ValueError(
            "GRANTSETU_ENCRYPTION_KEY is set but is not a valid Fernet key "
            f"({exc}). It must be exactly 32 url-safe base64-encoded bytes. "
            "Generate a real one with:\n"
            '  python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"\n'
            "then set GRANTSETU_ENCRYPTION_KEY to that generated value in your .env file."
        ) from exc
    return _fernet


def encrypt_api_key(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_api_key(ciphertext: str) -> str | None:
    """Returns None (never raises) if decryption fails -- e.g. the key was
    encrypted with a different GRANTSETU_ENCRYPTION_KEY (such as after a
    restart with no key configured). Callers must treat None as
    "provider unusable, fall back"."""
    try:
        return _get_fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except (InvalidToken, ValueError):
        logger.warning("Failed to decrypt a stored API key (wrong/rotated encryption key?)")
        return None
