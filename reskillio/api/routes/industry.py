"""GET /industry/match/{candidate_id} — industry match score endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from reskillio.models.industry import IndustryMatchResult
from reskillio.pipelines.industry_match_pipeline import run_industry_match
from config.settings import settings

router = APIRouter(prefix="/industry", tags=["industry-match"])


@router.get(
    "/match/{candidate_id}",
    response_model=IndustryMatchResult,
    status_code=status.HTTP_200_OK,
)
def industry_match(candidate_id: str) -> IndustryMatchResult:
    """
    Score a candidate's skill profile against all 8 target industries.

    Uses BQML ML.DISTANCE (COSINE) against pre-built industry centroid vectors.

    Returns
    -------
    IndustryMatchResult
        - top_industry / top_industry_label — best-fit industry
        - scores — all 8 industries ranked by match_score (0–100)
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    logger.info(f"Industry match request: candidate='{candidate_id}'")

    try:
        result = run_industry_match(
            candidate_id=candidate_id,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
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
        logger.error(f"Industry match failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Industry match failed.",
        ) from exc

    return result