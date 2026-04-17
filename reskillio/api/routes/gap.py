"""POST /gap — gap analysis endpoint."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from reskillio.api.routes.extract import _get_profile_store
from reskillio.models.gap import GapAnalysisResult
from reskillio.models.jd import Industry
from reskillio.pipelines.gap_pipeline import run_gap_analysis
from reskillio.storage.profile_store import CandidateProfileStore
from config.settings import settings

router = APIRouter(prefix="/gap", tags=["gap-analysis"])


class GapRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate to analyse")

    # Option A — stored JD
    jd_id: Optional[str] = Field(default=None, description="ID of a stored JD")

    # Option B — inline JD text (used when jd_id is not provided)
    jd_text: Optional[str] = Field(
        default=None, min_length=30, description="Raw JD text (not stored)"
    )
    jd_title:   Optional[str] = Field(default=None)
    jd_company: Optional[str] = Field(default=None)
    industry:   Optional[Industry] = Field(default=None)

    similarity_threshold: float = Field(
        default=0.75, ge=0.5, le=0.99,
        description="Cosine similarity threshold for transferable skills",
    )


@router.post("", response_model=GapAnalysisResult, status_code=status.HTTP_200_OK)
def analyse_gap(
    body: GapRequest,
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> GapAnalysisResult:
    """
    Compare a candidate's skill profile against a job description.

    Supply either `jd_id` (stored JD) or `jd_text` (inline, not persisted).

    Returns
    -------
    GapAnalysisResult
        - matched_skills     — exact name matches
        - transferable_skills — semantic matches above the threshold
        - missing_skills     — required skills not found
        - gap_score          — 0–100 fit score
        - recommendation     — plain-English assessment
    """
    if profile_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    if not body.jd_id and not body.jd_text:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide either jd_id or jd_text.",
        )

    logger.info(
        f"Gap analysis request: candidate='{body.candidate_id}' "
        f"jd_id={body.jd_id or 'inline'}"
    )

    try:
        result = run_gap_analysis(
            candidate_id=body.candidate_id,
            project_id=settings.gcp_project_id,
            jd_id=body.jd_id,
            jd_text=body.jd_text,
            jd_title=body.jd_title,
            jd_company=body.jd_company,
            industry=body.industry,
            similarity_threshold=body.similarity_threshold,
            region=settings.gcp_region,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Gap analysis failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gap analysis failed.",
        ) from exc

    return result