"""POST /market/analyze — MarketAnalystAgent endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from reskillio.agents.market_analyst_agent import run_market_analyst_agent
from reskillio.models.market import MarketAnalysisResult
from config.settings import settings

router = APIRouter(prefix="/market", tags=["market-analyst"])

_DEFAULT_TOP_N = 8


class MarketAnalyzeRequest(BaseModel):
    skills:       Optional[list[str]] = Field(
        default=None,
        description="Explicit skill list to analyse. If omitted, fetches from candidate profile.",
    )
    candidate_id: Optional[str] = Field(
        default=None,
        description="Candidate ID — used to fetch top skills when skills list is not provided.",
    )
    top_n: int = Field(
        default=_DEFAULT_TOP_N, ge=1, le=10,
        description="Number of top skills to pull from profile (ignored if skills provided).",
    )


@router.post("", response_model=MarketAnalysisResult, status_code=status.HTTP_200_OK)
def analyze_market(body: MarketAnalyzeRequest) -> MarketAnalysisResult:
    """
    Run the MarketAnalystAgent on a list of skills.

    The agent uses real-time DuckDuckGo web search to assess current job
    market demand, then returns demand_score (0–100) and trend direction
    (growing / stable / declining) per skill.

    Supply either:
    - `skills`       — explicit list of skill names
    - `candidate_id` — fetches the candidate's top skills from BigQuery
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    # Resolve skill list
    skills = body.skills
    if not skills:
        if not body.candidate_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Provide either 'skills' or 'candidate_id'.",
            )
        try:
            from reskillio.storage.profile_store import CandidateProfileStore
            store   = CandidateProfileStore(project_id=settings.gcp_project_id)
            profile = store.get_profile(body.candidate_id)
            if not profile.skills:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"No profile found for candidate '{body.candidate_id}'.",
                )
            skills = [s.skill_name for s in profile.skills[: body.top_n]]
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Profile fetch failed: {exc}",
            ) from exc

    logger.info(f"Market analysis request: skills={skills}")

    try:
        result = run_market_analyst_agent(
            skills=skills,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Market analyst agent failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Market analysis failed.",
        ) from exc

    return result