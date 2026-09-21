from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.db.models import NGOProfile
from api.db.session import get_db
from api.models.schemas import NGOProfileCreate, NGOProfileResponse

router = APIRouter(prefix="/ngo-profile", tags=["ngo-profile"])


@router.post("", response_model=NGOProfileResponse, status_code=201)
def create_ngo_profile(payload: NGOProfileCreate, db: Session = Depends(get_db)):
    profile = NGOProfile(
        name=payload.name,
        registration_type=payload.registration_type,
        website_url=payload.website_url,
        focus_areas=payload.focus_areas,
        locations=payload.locations,
        beneficiaries=payload.beneficiaries,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


@router.put("/{ngo_id}", response_model=NGOProfileResponse)
def update_ngo_profile(ngo_id: str, payload: NGOProfileCreate, db: Session = Depends(get_db)):
    profile = db.get(NGOProfile, ngo_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")
    profile.name = payload.name
    profile.registration_type = payload.registration_type
    profile.website_url = payload.website_url
    profile.focus_areas = payload.focus_areas
    profile.locations = payload.locations
    profile.beneficiaries = payload.beneficiaries
    db.commit()
    db.refresh(profile)
    return profile


@router.get("/{ngo_id}", response_model=NGOProfileResponse)
def get_ngo_profile(ngo_id: str, db: Session = Depends(get_db)):
    profile = db.get(NGOProfile, ngo_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"NGO profile '{ngo_id}' not found")
    return profile
