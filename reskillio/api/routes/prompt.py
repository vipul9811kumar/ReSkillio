"""
Step 8 — POST /prompt

Fires after /analyze returns. Generates one pointed question for the candidate
based on their skill profile and archetype. The answer is sent back to /enrich
as enrichment_context to sharpen market pulse and company radar results.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger
from pydantic import BaseModel

router = APIRouter(tags=["prompt"])


class PromptResult(BaseModel):
    candidate_id: str
    question:     str


def _require_gcp():
    from config.settings import settings
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured — set GCP_PROJECT_ID in .env",
        )
    return settings


@router.post(
    "/prompt",
    response_model=PromptResult,
    status_code=status.HTTP_200_OK,
    summary="Generate a personalised follow-up question for the candidate",
    description=(
        "Step 8. Called after /analyze returns. Generates one pointed question "
        "based on the candidate's archetype and skill profile. The user's answer "
        "should be passed to /enrich as enrichment_context."
    ),
)
async def generate_prompt(
    candidate_id:       Annotated[str,           Form(description="Candidate ID from /analyze")] = ...,
    archetype:          Annotated[Optional[str], Form(description="Archetype from trait_profile")] = None,
    identity_statement: Annotated[Optional[str], Form(description="Identity statement from trait_profile")] = None,
    industry:           Annotated[Optional[str], Form(description="Top industry label from /analyze")] = None,
) -> PromptResult:
    settings = _require_gcp()

    arch  = (archetype or "Operator").strip()
    ident = (identity_statement or "").strip()
    ind   = (industry or "your field").strip()

    logger.info(
        f"[prompt] Generating question — candidate={candidate_id} "
        f"archetype={arch} industry='{ind}'"
    )

    from reskillio.pipelines.prompt_pipeline import run_prompt_pipeline

    try:
        question = run_prompt_pipeline(
            candidate_id=candidate_id,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            archetype=arch,
            identity_statement=ident,
            industry=ind,
        )
    except Exception as exc:
        logger.exception(f"[prompt] Pipeline error: {exc}")
        raise HTTPException(500, f"Prompt generation failed: {exc}") from exc

    return PromptResult(candidate_id=candidate_id, question=question)
