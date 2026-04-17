"""Candidate profile endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from reskillio.api.routes.extract import _get_profile_store
from reskillio.models.profile import CandidateProfile
from reskillio.storage.profile_store import CandidateProfileStore

router = APIRouter(prefix="/candidate", tags=["candidate"])


@router.get("/{candidate_id}/profile", response_model=CandidateProfile)
def get_candidate_profile(
    candidate_id: str,
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> CandidateProfile:
    """
    Fetch the aggregated skill profile for a candidate.

    Returns skills ranked by frequency, with first_seen / last_seen
    timestamps and average confidence across all extractions.
    """
    if profile_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured — profile store unavailable.",
        )

    logger.info(f"Profile request for candidate='{candidate_id}'")
    profile = profile_store.get_profile(candidate_id)

    if not profile.skills:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for candidate '{candidate_id}'. Submit an extraction first.",
        )

    return profile


@router.post("/{candidate_id}/profile/refresh", response_model=dict)
def refresh_candidate_profile(
    candidate_id: str,
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> dict:
    """
    Force a profile recompute from skill_extractions for a candidate.

    Useful if extractions were written directly to BigQuery outside the API.
    """
    if profile_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured — profile store unavailable.",
        )

    logger.info(f"Profile refresh triggered for candidate='{candidate_id}'")
    affected = profile_store.upsert_profile(candidate_id)
    return {"candidate_id": candidate_id, "rows_affected": affected}