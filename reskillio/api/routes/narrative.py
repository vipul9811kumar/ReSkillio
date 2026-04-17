"""POST /narrative — RAG-grounded career narrative endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from reskillio.models.narrative import NarrativeRequest, NarrativeResult
from reskillio.pipelines.narrative_pipeline import run_narrative_pipeline
from config.settings import settings

router = APIRouter(prefix="/narrative", tags=["narrative"])


@router.post("", response_model=NarrativeResult, status_code=status.HTTP_200_OK)
def generate_narrative(body: NarrativeRequest) -> NarrativeResult:
    """
    Generate a 3-sentence RAG-grounded career rebound narrative.

    RAG sources (all from BigQuery — no hallucination):
    - Candidate top skills from `candidate_profiles`
    - Industry demand data from `industry_profiles`
    - Sample JD titles from `jd_profiles`

    If `industry` is omitted, the top match from the industry scoring
    pipeline is used automatically.

    Returns
    -------
    NarrativeResult
        - narrative      — the 3-sentence Gemini-generated narrative
        - grounding      — exact BQ facts passed to the model
        - industry_label — resolved industry name
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    logger.info(
        f"Narrative request: candidate='{body.candidate_id}' "
        f"role='{body.target_role}' industry={body.industry}"
    )

    try:
        result = run_narrative_pipeline(
            candidate_id=body.candidate_id,
            target_role=body.target_role,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            industry=body.industry,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Narrative generation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Narrative generation failed.",
        ) from exc

    return result