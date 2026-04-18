"""
Option B — POST /enrich

Called by the frontend immediately after /analyze returns, in the background.
Accepts candidate_id (no resume re-upload needed) and runs the slow
enrichment agents: MarketPulse, AutoGap, SalaryIntel, CompanyRadar.

Returns EnrichmentResult — all Optional fields are None until their
respective build step is implemented (Steps 3–6).
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from reskillio.models.enrich import EnrichmentResult

router = APIRouter(tags=["enrich"])


def _require_gcp():
    from config.settings import settings
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured — set GCP_PROJECT_ID in .env",
        )
    return settings


@router.post(
    "/enrich",
    response_model=EnrichmentResult,
    status_code=status.HTTP_200_OK,
    summary="Background enrichment — market pulse, salary, company radar",
    description=(
        "Option B slow path. Call after /analyze returns. "
        "Accepts candidate_id (no resume re-upload). "
        "Runs MarketPulseAgent, auto-gap, salary intelligence, and CompanyRadarAgent. "
        "All stages are fail-safe — a slow or failing stage never blocks earlier results."
    ),
)
async def enrich(
    candidate_id: Annotated[str,           Form(description="Candidate ID from the /analyze response")] = ...,
    target_role:  Annotated[Optional[str], Form(description="Target role from the /analyze call")] = None,
    context_text: Annotated[Optional[str], Form(description="Free-text context from the original upload")] = None,
    ideal_stage:  Annotated[Optional[str], Form(description="Ideal company stage from trait profile (Startup/Growth-stage/Enterprise/Turnaround)")] = None,
) -> EnrichmentResult:
    """
    Background enrichment for a candidate already processed by /analyze.

    The candidate's skill profile is read from BigQuery — no resume re-upload needed.
    Runs MarketPulse, AutoGap, SalaryIntel, and CompanyRadar stages.
    All stages are fail-safe — a slow or failing stage never blocks earlier results.
    """
    settings = _require_gcp()

    role_str  = (target_role or "").strip() or "Career Transition"
    stage_str = (ideal_stage or "").strip() or "Enterprise"

    logger.info(
        f"[enrich] Starting enrichment — "
        f"candidate={candidate_id} role='{role_str}' stage='{stage_str}' "
        f"context={'yes' if context_text else 'no'}"
    )

    from reskillio.pipelines.enrich_pipeline import run_enrichment

    try:
        result = run_enrichment(
            candidate_id=candidate_id,
            target_role=role_str,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            context_text=context_text or None,
            ideal_stage=stage_str,
        )
    except Exception as exc:
        logger.exception(f"[enrich] Pipeline error: {exc}")
        raise HTTPException(500, f"Enrichment failed: {exc}") from exc

    return result
