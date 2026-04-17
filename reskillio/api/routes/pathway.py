"""POST /pathway/plan — PathwayPlannerAgent endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from reskillio.agents.pathway_planner_agent import run_pathway_planner_agent
from reskillio.models.pathway import PathwayRoadmap
from config.settings import settings

router = APIRouter(prefix="/pathway", tags=["pathway-planner"])


class PathwayPlanRequest(BaseModel):
    candidate_id:        str             = Field(..., description="Candidate identifier")
    target_role:         str             = Field(..., description="Job title being targeted")
    missing_skills:      list[str]       = Field(default_factory=list, description="Skills required but absent from profile")
    transferable_skills: list[str]       = Field(default_factory=list, description="Skills partially covered — worth deepening")
    gap_score:           float           = Field(..., ge=0.0, le=100.0, description="Current JD fit score 0–100")
    market_scores:       dict[str, float] = Field(
        default_factory=dict,
        description="demand_score per skill from MarketAnalystAgent (0–100). Skills not listed default to 50.",
    )


@router.post("/plan", response_model=PathwayRoadmap, status_code=status.HTTP_200_OK)
def plan_pathway(body: PathwayPlanRequest) -> PathwayRoadmap:
    """
    Generate a 90-day reskilling roadmap for a displaced professional.

    The PathwayPlannerAgent runs a two-agent CrewAI crew:
    1. **Researcher** — searches Coursera and Udemy (via DuckDuckGo) for each
       priority skill gap and returns real course links.
    2. **Planner** — synthesises the research into a structured 3-phase roadmap
       with milestones, weekly study hours, and success metrics.

    Provide either `missing_skills` or `transferable_skills` (or both).
    Optionally supply `market_scores` from the `/market` endpoint to prioritise
    the highest-demand skills first.
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    if not body.missing_skills and not body.transferable_skills:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide at least one of 'missing_skills' or 'transferable_skills'.",
        )

    logger.info(
        f"Pathway plan request: candidate='{body.candidate_id}' "
        f"role='{body.target_role}' missing={len(body.missing_skills)} "
        f"transferable={len(body.transferable_skills)} gap={body.gap_score:.0f}"
    )

    try:
        roadmap = run_pathway_planner_agent(
            candidate_id=body.candidate_id,
            target_role=body.target_role,
            missing_skills=body.missing_skills,
            transferable_skills=body.transferable_skills,
            gap_score=body.gap_score,
            market_scores=body.market_scores,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Pathway planner agent failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Pathway planning failed.",
        ) from exc

    return roadmap