"""
Option B — Enrichment pipeline (slow path, ~60 s when fully built).

Called by POST /enrich after the fast /analyze has already returned.
Reads the candidate's stored skill profile from BigQuery so the client
only needs to send candidate_id — no re-upload of resume needed.

Stages added here as Steps 3-6 are built:
  Stage E1 — MarketPulseAgent    (Step 3)
  Stage E2 — Auto-gap            (Step 4)
  Stage E3 — Salary Intelligence (Step 5)
  Stage E4 — CompanyRadarAgent   (Step 6)
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.models.analyze import StageResult
from reskillio.models.enrich import (
    AutoGapResult,
    CompanyRadarResult,
    EnrichmentResult,
    MarketPulseResult,
    SalaryIntelResult,
)


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def run_enrichment(
    candidate_id: str,
    target_role:  str,
    project_id:   str,
    region:       str = "us-central1",
    context_text: Optional[str] = None,
) -> EnrichmentResult:
    """
    Run the enrichment pipeline for a candidate whose skills are already
    stored in BigQuery from a prior /analyze call.

    Parameters
    ----------
    candidate_id : Unique ID matching the /analyze call.
    target_role  : Role string from the /analyze call (may be "Career Transition").
    project_id   : GCP project ID.
    region       : Vertex AI region.
    context_text : Optional free-text context passed through from the frontend.

    Returns
    -------
    EnrichmentResult — all Optional fields are None until their step is built.
    """
    wall_start = time.perf_counter()
    stages: dict[str, StageResult] = {}

    # ── Resolve candidate skill profile from BigQuery ─────────────────────
    skill_names:     list[str] = []
    industry_label:  str       = "General Business"

    try:
        from reskillio.storage.profile_store import CandidateProfileStore
        from reskillio.pipelines.industry_match_pipeline import run_industry_match
        from reskillio.models.industry import _INDUSTRY_LABELS

        profile_store = CandidateProfileStore(project_id=project_id)
        profile = profile_store.get_profile(candidate_id)
        skill_names = [s.skill_name for s in profile.skills[:20]]

        if skill_names:
            industry_result = run_industry_match(
                candidate_id=candidate_id,
                project_id=project_id,
                region=region,
            )
            if industry_result.scores:
                top_key   = industry_result.scores[0].industry
                industry_label = _INDUSTRY_LABELS.get(top_key, top_key)

        logger.info(
            f"[enrich] Profile loaded — {len(skill_names)} skills, "
            f"industry='{industry_label}' for candidate='{candidate_id}'"
        )
    except Exception as exc:
        logger.warning(f"[enrich] Profile lookup failed: {exc} — proceeding with empty profile")

    # ====================================================================
    # Stage E1 — Market Pulse  (Step 3)
    # ====================================================================
    market_pulse: Optional[MarketPulseResult] = None
    t = time.perf_counter()
    try:
        from reskillio.agents.market_pulse_agent import run_market_pulse_agent
        market_pulse = run_market_pulse_agent(
            skill_names=skill_names,
            industry=industry_label,
            target_role=target_role,
            project_id=project_id,
            region=region,
        )
        stages["market_pulse"] = StageResult(success=True, duration_ms=_ms(t))
        logger.info(
            f"[enrich] Market pulse done — {len(market_pulse.top_roles)} roles, "
            f"demand={market_pulse.overall_demand}"
        )
    except Exception as exc:
        stages["market_pulse"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
        logger.warning(f"[enrich] Market pulse failed: {exc}")

    # ====================================================================
    # Stage E2 — Auto-gap  (Step 4)
    # ====================================================================
    auto_gap: Optional[AutoGapResult] = None
    t = time.perf_counter()
    if market_pulse and market_pulse.top_roles:
        try:
            from reskillio.pipelines.auto_gap_pipeline import run_auto_gap
            auto_gap = run_auto_gap(
                top_roles=market_pulse.top_roles,
                candidate_id=candidate_id,
                project_id=project_id,
                region=region,
            )
            stages["auto_gap"] = StageResult(success=True, duration_ms=_ms(t))
            logger.info(
                f"[enrich] Auto-gap done — readiness={auto_gap.overall_readiness:.1f}, "
                f"top_missing={auto_gap.top_skills_to_add[:3]}"
            )
        except Exception as exc:
            stages["auto_gap"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
            logger.warning(f"[enrich] Auto-gap failed: {exc}")
    else:
        stages["auto_gap"] = StageResult(
            success=True, duration_ms=0, error="skipped — no market pulse roles"
        )

    # ====================================================================
    # Stage E3 — Salary Intelligence  (Step 5 — placeholder)
    # ====================================================================
    salary_intel: Optional[SalaryIntelResult] = None
    # TODO Step 5: from reskillio.pipelines.salary_intel_pipeline import run_salary_intel
    stages["salary_intel"] = StageResult(
        success=True, duration_ms=0, error="pending — Step 5 not yet built"
    )

    # ====================================================================
    # Stage E4 — Company Radar  (Step 6 — placeholder)
    # ====================================================================
    company_radar: Optional[CompanyRadarResult] = None
    # TODO Step 6: from reskillio.agents.company_radar_agent import run_company_radar_agent
    stages["company_radar"] = StageResult(
        success=True, duration_ms=0, error="pending — Step 6 not yet built"
    )

    total_ms = _ms(wall_start)
    logger.info(f"[enrich] Complete: {total_ms} ms for candidate='{candidate_id}'")

    return EnrichmentResult(
        candidate_id=candidate_id,
        target_role=target_role,
        enriched_at=datetime.now(timezone.utc),
        market_pulse=market_pulse,
        auto_gap=auto_gap,
        salary_intel=salary_intel,
        company_radar=company_radar,
        stages=stages,
        total_duration_ms=total_ms,
    )
