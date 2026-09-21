"""Resolves the configured AIProvider for one NGO, or None if unavailable --
callers MUST handle None by falling back to rule-based logic (src/ai/research.py
does this consistently; nothing else in the app should call src/ai/providers.py
directly)."""
import logging

from sqlalchemy.orm import Session

from api.db.models import AIProviderConfig
from src.ai.crypto import decrypt_api_key
from src.ai.providers import AIProvider, build_provider

logger = logging.getLogger("grantsetu.ai.factory")


def get_provider_for_ngo(db: Session, ngo_id: str) -> AIProvider | None:
    config = db.get(AIProviderConfig, ngo_id)
    if config is None or not config.is_enabled:
        return None
    if not config.api_key_encrypted:
        return None

    api_key = decrypt_api_key(config.api_key_encrypted)
    if api_key is None:
        logger.warning("AI config for ngo_id=%s could not be decrypted; falling back", ngo_id)
        return None

    try:
        return build_provider(config.provider, api_key, model=config.model, base_url=config.base_url)
    except ValueError as exc:
        logger.warning("Could not build AI provider for ngo_id=%s: %s", ngo_id, exc)
        return None
