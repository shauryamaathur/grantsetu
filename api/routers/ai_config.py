from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db.models import AIProviderConfig, NGOProfile
from api.db.session import get_db
from api.models.schemas import (
    AIProviderConfigCreate,
    AIProviderConfigResponse,
    AIProviderConfigUpdate,
    AIProviderTestResult,
)
from src.ai.crypto import decrypt_api_key, encrypt_api_key
from src.ai.providers import build_provider

router = APIRouter(prefix="/ngo-profile/{ngo_id}/ai-config", tags=["ai-config"])


def _to_response(config: AIProviderConfig) -> AIProviderConfigResponse:
    return AIProviderConfigResponse(
        ngo_id=config.ngo_id,
        provider=config.provider,
        model=config.model,
        base_url=config.base_url,
        is_enabled=config.is_enabled,
        has_api_key=bool(config.api_key_encrypted),
        last_test_status=config.last_test_status,
        last_test_message=config.last_test_message,
        last_tested_at=config.last_tested_at,
    )


def _require_ngo(db: Session, ngo_id: str) -> NGOProfile:
    ngo = db.get(NGOProfile, ngo_id)
    if ngo is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")
    return ngo


@router.put("", response_model=AIProviderConfigResponse)
def set_ai_config(ngo_id: str, payload: AIProviderConfigCreate, db: Session = Depends(get_db)):
    """Create or replace the AI provider config for this NGO. The API key
    is encrypted (src/ai/crypto.py) before being stored -- it is NEVER
    written back in this or any other response."""
    _require_ngo(db, ngo_id)

    config = db.get(AIProviderConfig, ngo_id)
    if config is None:
        config = AIProviderConfig(ngo_id=ngo_id)
        db.add(config)

    config.provider = payload.provider
    config.api_key_encrypted = encrypt_api_key(payload.api_key)
    config.model = payload.model
    config.base_url = payload.base_url
    config.is_enabled = payload.is_enabled
    config.last_test_status = "untested"
    config.last_test_message = None
    config.last_tested_at = None

    db.commit()
    db.refresh(config)
    return _to_response(config)


@router.patch("", response_model=AIProviderConfigResponse)
def update_ai_config(ngo_id: str, payload: AIProviderConfigUpdate, db: Session = Depends(get_db)):
    _require_ngo(db, ngo_id)
    config = db.get(AIProviderConfig, ngo_id)
    if config is None:
        raise HTTPException(status_code=404, detail="No AI config exists for this NGO yet -- use PUT to create one")

    if payload.api_key is not None:
        config.api_key_encrypted = encrypt_api_key(payload.api_key)
        config.last_test_status = "untested"
    if payload.model is not None:
        config.model = payload.model
    if payload.base_url is not None:
        config.base_url = payload.base_url
    if payload.is_enabled is not None:
        config.is_enabled = payload.is_enabled

    db.commit()
    db.refresh(config)
    return _to_response(config)


@router.get("", response_model=AIProviderConfigResponse)
def get_ai_config(ngo_id: str, db: Session = Depends(get_db)):
    _require_ngo(db, ngo_id)
    config = db.get(AIProviderConfig, ngo_id)
    if config is None:
        raise HTTPException(status_code=404, detail="No AI config exists for this NGO yet")
    return _to_response(config)


@router.delete("", status_code=204)
def delete_ai_config(ngo_id: str, db: Session = Depends(get_db)):
    _require_ngo(db, ngo_id)
    config = db.get(AIProviderConfig, ngo_id)
    if config is None:
        raise HTTPException(status_code=404, detail="No AI config exists for this NGO yet")
    db.delete(config)
    db.commit()


@router.post("/test", response_model=AIProviderTestResult)
def test_ai_config(ngo_id: str, db: Session = Depends(get_db)):
    """Makes a REAL, minimal call to the configured provider and honestly
    reports whether it succeeded. This is the only endpoint that actually
    contacts an external AI API in this project -- everywhere else, AI is
    used via src/ai/research.py with an automatic rule-based fallback."""
    _require_ngo(db, ngo_id)
    config = db.get(AIProviderConfig, ngo_id)
    if config is None or not config.api_key_encrypted:
        raise HTTPException(status_code=404, detail="No AI config with an API key exists for this NGO yet")

    api_key = decrypt_api_key(config.api_key_encrypted)
    if api_key is None:
        config.last_test_status = "failed"
        config.last_test_message = "Stored API key could not be decrypted (encryption key may have changed)"
        db.commit()
        return AIProviderTestResult(ok=False, message=config.last_test_message)

    try:
        provider = build_provider(config.provider, api_key, model=config.model, base_url=config.base_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = provider.test_connection()

    config.last_test_status = "ok" if result.ok else "failed"
    config.last_test_message = result.text if result.ok else result.error
    config.last_tested_at = datetime.utcnow()
    db.commit()

    return AIProviderTestResult(ok=result.ok, message=(result.text or "Connected successfully") if result.ok else (result.error or "Connection failed"))
