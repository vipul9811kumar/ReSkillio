"""
Option B — Enrichment pipeline (parallel stages, ~20-30 s).

Stages run in two parallel rounds:
  Round 1 (parallel): E1 MarketPulse  +  E3 SalaryIntel
  Round 2 (parallel): E2 AutoGap      +  E4 CompanyRadar
  (E2 and E4 both need MarketPulse output, so they wait for Round 1)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ideal_stage:  str = "Enterprise",
) -> EnrichmentResult:
    """
    Run enrichment pipeline for a candidate already processed by /analyze.
    Parallel execution cuts total time from ~60s sequential → ~20-30s.
    """
    wall_start = time.perf_counter()
    stages: dict[str, StageResult] = {}

    # ── Resolve candidate skill profile from BigQuery ─────────────────────
    skill_names:    list[str] = []
    industry_label: str       = "General Business"

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
                top_key        = industry_result.scores[0].industry
                industry_label = _INDUSTRY_LABELS.get(top_key, top_key)

        logger.info(
            f"[enrich] Profile — {len(skill_names)} skills, "
            f"industry='{industry_label}', candidate='{candidate_id}'"
        )
    except Exception as exc:
        logger.warning(f"[enrich] Profile lookup failed: {exc}")

    # ====================================================================
    # Round 1 — E1 MarketPulse  +  E3 SalaryIntel  (parallel)
    # ====================================================================
    market_pulse: Optional[MarketPulseResult] = None
    salary_intel: Optional[SalaryIntelResult] = None

    def _run_market_pulse():
        from reskillio.agents.market_pulse_agent import run_market_pulse_agent
        return run_market_pulse_agent(
            skill_names=skill_names,
            industry=industry_label,
            target_role=target_role,
            project_id=project_id,
            region=region,
        )

    def _run_salary_intel():
        from reskillio.pipelines.salary_intel_pipeline import run_salary_intel
        return run_salary_intel(
            skill_names=skill_names,
            industry=industry_label,
            target_role=target_role,
            project_id=project_id,
            region=region,
        )

    t_r1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_mp = pool.submit(_run_market_pulse)
        f_si = pool.submit(_run_salary_intel)

        t_mp = time.perf_counter()
        try:
            market_pulse = f_mp.result(timeout=60)
            stages["market_pulse"] = StageResult(success=True, duration_ms=_ms(t_mp))
            logger.info(f"[enrich] MarketPulse done — {len(market_pulse.top_roles)} roles, demand={market_pulse.overall_demand}")
        except Exception as exc:
            stages["market_pulse"] = StageResult(success=False, duration_ms=_ms(t_mp), error=str(exc))
            logger.warning(f"[enrich] MarketPulse failed: {exc}")

        t_si = time.perf_counter()
        try:
            salary_intel = f_si.result(timeout=60)
            stages["salary_intel"] = StageResult(success=True, duration_ms=_ms(t_si))
            logger.info(f"[enrich] SalaryIntel done — ${salary_intel.floor_usd:,}/${salary_intel.median_usd:,}/${salary_intel.ceiling_usd:,}")
        except Exception as exc:
            stages["salary_intel"] = StageResult(success=False, duration_ms=_ms(t_si), error=str(exc))
            logger.warning(f"[enrich] SalaryIntel failed: {exc}")

    logger.info(f"[enrich] Round 1 done in {_ms(t_r1)} ms")

    # ====================================================================
    # Round 2 — E2 AutoGap  +  E4 CompanyRadar  (parallel, need E1 output)
    # ====================================================================
    auto_gap:      Optional[AutoGapResult]      = None
    company_radar: Optional[CompanyRadarResult] = None

    top_role_titles = (
        [r.title for r in market_pulse.top_roles]
        if market_pulse and market_pulse.top_roles
        else [target_role]
    )

    def _run_auto_gap():
        if not (market_pulse and market_pulse.top_roles):
            raise ValueError("skipped — no market pulse roles")
        from reskillio.pipelines.auto_gap_pipeline import run_auto_gap
        return run_auto_gap(
            top_roles=market_pulse.top_roles,
            candidate_id=candidate_id,
            project_id=project_id,
            region=region,
        )

    def _run_company_radar():
        from reskillio.agents.company_radar_agent import run_company_radar_agent
        return run_company_radar_agent(
            skill_names=skill_names,
            industry=industry_label,
            top_roles=top_role_titles,
            ideal_stage=ideal_stage,
            project_id=project_id,
            region=region,
        )

    t_r2 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_ag = pool.submit(_run_auto_gap)
        f_cr = pool.submit(_run_company_radar)

        t_ag = time.perf_counter()
        try:
            auto_gap = f_ag.result(timeout=60)
            stages["auto_gap"] = StageResult(success=True, duration_ms=_ms(t_ag))
            logger.info(f"[enrich] AutoGap done — readiness={auto_gap.overall_readiness:.1f}, top_missing={auto_gap.top_skills_to_add[:3]}")
        except Exception as exc:
            stages["auto_gap"] = StageResult(success=False, duration_ms=_ms(t_ag), error=str(exc))
            logger.warning(f"[enrich] AutoGap failed: {exc}")

        t_cr = time.perf_counter()
        try:
            company_radar = f_cr.result(timeout=60)
            stages["company_radar"] = StageResult(success=True, duration_ms=_ms(t_cr))
            logger.info(f"[enrich] CompanyRadar done — {len(company_radar.companies)} employers")
        except Exception as exc:
            stages["company_radar"] = StageResult(success=False, duration_ms=_ms(t_cr), error=str(exc))
            logger.warning(f"[enrich] CompanyRadar failed: {exc}")

    logger.info(f"[enrich] Round 2 done in {_ms(t_r2)} ms")

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
