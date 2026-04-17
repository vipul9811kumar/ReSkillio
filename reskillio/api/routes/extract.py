"""POST /extract — skill extraction endpoint."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from reskillio.api.schemas import ExtractRequest, ExtractResponse
from reskillio.pipelines.skill_pipeline import run_skill_extraction
from reskillio.storage.bigquery_store import BigQuerySkillStore
from reskillio.storage.profile_store import CandidateProfileStore
from config.settings import settings

router = APIRouter(prefix="/extract", tags=["extraction"])


def _get_store() -> BigQuerySkillStore | None:
    """Return a BigQuerySkillStore if GCP is configured, else None."""
    if not settings.gcp_project_id:
        return None
    return BigQuerySkillStore(project_id=settings.gcp_project_id)


def _get_profile_store() -> CandidateProfileStore | None:
    if not settings.gcp_project_id:
        return None
    return CandidateProfileStore(project_id=settings.gcp_project_id)


@router.post("", response_model=ExtractResponse, status_code=status.HTTP_200_OK)
def extract_skills(
    body: ExtractRequest,
    store: BigQuerySkillStore | None = Depends(_get_store),
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> ExtractResponse:
    """
    Extract skills from resume or job-description text.

    - Runs the spaCy skill extraction pipeline.
    - Persists results to BigQuery when GCP is configured.
    - Updates candidate profile after each extraction.
    - Returns structured skills with category and confidence.
    """
    logger.info(f"Extract request for candidate_id='{body.candidate_id}' ({len(body.text)} chars)")

    try:
        result = run_skill_extraction(body.text, model_name=settings.spacy_model)
    except Exception as exc:
        logger.error(f"Extraction failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Skill extraction failed.",
        ) from exc

    extraction_id = str(uuid.uuid4())
    stored = False

    if store is not None:
        try:
            store.store_extraction(result, candidate_id=body.candidate_id)
            stored = True
        except Exception as exc:
            logger.warning(f"BigQuery write failed (non-fatal): {exc}")

    if stored and profile_store is not None:
        try:
            profile_store.upsert_profile(body.candidate_id)
        except Exception as exc:
            logger.warning(f"Profile upsert failed (non-fatal): {exc}")

    return ExtractResponse(
        candidate_id=body.candidate_id,
        extraction_id=extraction_id,
        skill_count=result.skill_count,
        skills=result.skills,
        model_used=result.model_used,
        stored=stored,
    )