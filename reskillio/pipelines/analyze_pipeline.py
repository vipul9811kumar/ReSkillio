"""
F16 — Full career-rebound orchestration pipeline.

Runs all five stages in sequence.  Each stage is wrapped in a try/except so
a failure in one stage (e.g. no industry vectors seeded yet) never blocks the
results already produced by earlier stages.

Stages
------
1. extract   — spaCy skill extraction + BigQuery storage + profile upsert
2. gap        — gap analysis vs the target JD / role description
3. industry   — BQML cosine scoring against 8 industry vectors
4. narrative  — Gemini RAG-grounded career narrative
5. pathway    — CrewAI 90-day reskilling plan (opt-in, slow ~45 s)

The caller should set include_pathway=False for quick demos.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.models.analyze import AnalysisResult, SkillSummary, StageResult
from reskillio.models.gap import GapAnalysisResult
from reskillio.models.industry import IndustryMatchResult
from reskillio.models.jd import Industry
from reskillio.models.narrative import NarrativeGrounding, NarrativeResult
from reskillio.models.pathway import PathwayRoadmap


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def run_full_analysis(
    resume_text:     str,
    candidate_id:    str,
    target_role:     str,
    project_id:      str,
    region:          str = "us-central1",
    jd_text:         Optional[str] = None,
    jd_title:        Optional[str] = None,
    industry:        Optional[str] = None,
    include_pathway: bool = False,
    spacy_model:     str = "en_core_web_lg",
) -> AnalysisResult:
    """
    Orchestrate the full career-rebound pipeline for one candidate.

    Parameters
    ----------
    resume_text:
        Raw text from the resume (PDF already converted to text by the caller).
    candidate_id:
        Unique identifier — used as the BigQuery primary key.
    target_role:
        Job title the candidate is targeting, e.g. "Senior Data Engineer".
    project_id:
        GCP project ID.
    region:
        Vertex AI region for embeddings + Gemini.
    jd_text:
        Optional job description text for gap analysis.
        If omitted, gap analysis is skipped.
    jd_title:
        Optional JD title (used in gap meta and narrative).
    industry:
        Optional industry name (auto-detected from industry_match if omitted).
    include_pathway:
        If True, run the CrewAI PathwayPlannerAgent (~45 s).
        Defaults to False for fast demo responses.
    spacy_model:
        spaCy model used for extraction.

    Returns
    -------
    AnalysisResult
        Aggregated output from all completed stages + per-stage metadata.
    """
    wall_start = time.perf_counter()
    stages: dict[str, StageResult] = {}

    # ── Resolve industry enum ────────────────────────────────────────────
    industry_enum: Optional[Industry] = None
    if industry:
        try:
            industry_enum = Industry(industry)
        except ValueError:
            logger.warning(f"[analyze] Unknown industry '{industry}' — will auto-detect")

    # ====================================================================
    # Stage 1 — Skill extraction + BigQuery storage + profile upsert
    # ====================================================================
    t = time.perf_counter()
    top_skills: list[SkillSummary] = []
    skill_count = 0

    try:
        from reskillio.pipelines.skill_pipeline import run_skill_extraction
        from reskillio.storage.bigquery_store import BigQuerySkillStore
        from reskillio.storage.profile_store import CandidateProfileStore

        result = run_skill_extraction(
            text=resume_text,
            model_name=spacy_model,
            candidate_id=candidate_id,
        )
        skill_count = len(result.skills)

        # Persist to BigQuery
        bq_store = BigQuerySkillStore(project_id=project_id)
        bq_store.store_extraction(result, candidate_id)

        # Upsert profile (needed by all downstream stages)
        profile_store = CandidateProfileStore(project_id=project_id)
        profile_store.upsert_profile(candidate_id)

        top_skills = [
            SkillSummary(
                name=s.name,
                category=s.category.value,
                confidence=round(s.confidence, 3),
            )
            for s in sorted(result.skills, key=lambda x: x.confidence, reverse=True)[:20]
        ]

        stages["extract"] = StageResult(success=True, duration_ms=_ms(t))
        logger.info(f"[analyze] Stage 1 extract: {skill_count} skills in {_ms(t)} ms")

    except Exception as exc:
        stages["extract"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
        logger.error(f"[analyze] Stage 1 extract FAILED: {exc}")
        # Cannot proceed without skills
        return AnalysisResult(
            candidate_id=candidate_id,
            target_role=target_role,
            analyzed_at=datetime.now(timezone.utc),
            skill_count=0,
            top_skills=[],
            stages=stages,
            total_duration_ms=_ms(wall_start),
        )

    # ====================================================================
    # Stage 2 — Gap analysis (only when JD text provided)
    # ====================================================================
    t = time.perf_counter()
    gap: Optional[GapAnalysisResult] = None

    if jd_text:
        try:
            from reskillio.pipelines.gap_pipeline import run_gap_analysis

            gap = run_gap_analysis(
                candidate_id=candidate_id,
                project_id=project_id,
                jd_text=jd_text,
                jd_title=jd_title or target_role,
                industry=industry_enum,
                region=region,
            )
            stages["gap"] = StageResult(success=True, duration_ms=_ms(t))
            logger.info(
                f"[analyze] Stage 2 gap: score={gap.gap_score:.1f} "
                f"matched={len(gap.matched_skills)} missing={len(gap.missing_skills)} "
                f"in {_ms(t)} ms"
            )
        except Exception as exc:
            stages["gap"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
            logger.warning(f"[analyze] Stage 2 gap FAILED: {exc}")
    else:
        stages["gap"] = StageResult(success=True, duration_ms=0, error="skipped — no jd_text")
        logger.info("[analyze] Stage 2 gap: skipped (no jd_text provided)")

    # ====================================================================
    # Stage 3 — Industry match
    # ====================================================================
    t = time.perf_counter()
    industry_match: Optional[IndustryMatchResult] = None

    try:
        from reskillio.pipelines.industry_match_pipeline import run_industry_match

        industry_match = run_industry_match(
            candidate_id=candidate_id,
            project_id=project_id,
            region=region,
        )
        stages["industry"] = StageResult(success=True, duration_ms=_ms(t))
        best = industry_match.scores[0] if industry_match.scores else None
        logger.info(
            f"[analyze] Stage 3 industry: best='{best.industry if best else '?'}' "
            f"score={best.match_score if best else 0:.1f} in {_ms(t)} ms"
        )
    except Exception as exc:
        stages["industry"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
        logger.warning(f"[analyze] Stage 3 industry FAILED: {exc}")

    # ====================================================================
    # Stage 4 — Gemini career narrative
    # ====================================================================
    t = time.perf_counter()
    narrative_text: Optional[str] = None
    narrative_grounding: Optional[NarrativeGrounding] = None

    try:
        from reskillio.pipelines.narrative_pipeline import run_narrative_pipeline

        narrative_result: NarrativeResult = run_narrative_pipeline(
            candidate_id=candidate_id,
            target_role=target_role,
            project_id=project_id,
            region=region,
            industry=industry_enum,
        )
        narrative_text      = narrative_result.narrative
        narrative_grounding = narrative_result.grounding
        stages["narrative"] = StageResult(success=True, duration_ms=_ms(t))
        logger.info(f"[analyze] Stage 4 narrative: {len(narrative_text)} chars in {_ms(t)} ms")

    except Exception as exc:
        stages["narrative"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
        logger.warning(f"[analyze] Stage 4 narrative FAILED: {exc}")

    # ====================================================================
    # Stage 5 — 90-day reskilling pathway (opt-in)
    # ====================================================================
    t = time.perf_counter()
    pathway: Optional[PathwayRoadmap] = None

    if include_pathway:
        try:
            from reskillio.agents.pathway_planner_agent import run_pathway_planner_agent

            missing  = gap.missing_skills if gap else []
            transfer = [t.candidate_skill for t in gap.transferable_skills] if gap else []
            gap_score = gap.gap_score if gap else 50.0

            # Pull market scores from industry match if available
            market_scores: dict[str, float] = {}
            if industry_match and industry_match.scores:
                market_scores = {
                    s.industry: s.match_score
                    for s in industry_match.scores
                }

            pathway = run_pathway_planner_agent(
                candidate_id=candidate_id,
                target_role=target_role,
                missing_skills=missing,
                transferable_skills=transfer,
                gap_score=gap_score,
                market_scores=market_scores,
                project_id=project_id,
                region=region,
            )
            stages["pathway"] = StageResult(success=True, duration_ms=_ms(t))
            logger.info(f"[analyze] Stage 5 pathway: done in {_ms(t)} ms")

        except Exception as exc:
            stages["pathway"] = StageResult(success=False, duration_ms=_ms(t), error=str(exc))
            logger.warning(f"[analyze] Stage 5 pathway FAILED: {exc}")
    else:
        stages["pathway"] = StageResult(
            success=True, duration_ms=0, error="skipped — include_pathway=false"
        )

    total_ms = _ms(wall_start)
    logger.info(f"[analyze] Complete: {total_ms} ms total for candidate={candidate_id}")

    return AnalysisResult(
        candidate_id=candidate_id,
        target_role=target_role,
        analyzed_at=datetime.now(timezone.utc),
        skill_count=skill_count,
        top_skills=top_skills,
        gap=gap,
        industry_match=industry_match,
        narrative=narrative_text,
        narrative_grounding=narrative_grounding,
        pathway=pathway,
        stages=stages,
        total_duration_ms=total_ms,
    )
