"""
F8 — Career narrative pipeline (RAG-grounded).

RAG retrieval order
-------------------
1. Candidate top skills  → candidate_profiles (BQ)
2. Industry demand data  → industry_profiles (BQ)
3. Sample JD titles      → jd_profiles (BQ)
4. Overlap computation   → Python set intersection

Gemini call
-----------
gemini-1.5-flash with temperature=0.2 and a tightly scoped system
instruction — model is constrained to only reference facts from
the retrieved context, eliminating hallucination.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from google.cloud import bigquery
from loguru import logger

from reskillio.models.industry import _INDUSTRY_LABELS
from reskillio.models.jd import Industry
from reskillio.models.narrative import NarrativeGrounding, NarrativeResult
from reskillio.storage.profile_store import CandidateProfileStore

NARRATIVE_MODEL_VERTEX = "gemini-2.5-flash"
NARRATIVE_MODEL_STUDIO = "gemini-2.0-flash"
_TOP_CANDIDATE_SKILLS = 8
_TOP_INDUSTRY_SKILLS = 10
_MAX_JD_TITLES = 5

_SYSTEM_INSTRUCTION = (
    "You are a career transition advisor writing concise, factual career narratives "
    "for professionals re-entering the workforce after displacement. "
    "Use ONLY the skills, roles, and market facts provided in the user message. "
    "Do not invent certifications, companies, projects, or skills not listed. "
    "Write in second person (\"You bring...\", \"Your background...\")."
)


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


def _fetch_industry_skills(
    client: bigquery.Client,
    project_id: str,
    industry: str,
    limit: int = _TOP_INDUSTRY_SKILLS,
) -> list[str]:
    """Top demanded skill names for an industry, ordered by demand_weight."""
    query = f"""
        SELECT skill_name
        FROM `{project_id}.reskillio.industry_profiles`
        WHERE industry = @industry
        ORDER BY demand_weight DESC
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("industry", "STRING", industry),
            bigquery.ScalarQueryParameter("limit",    "INT64",  limit),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return [row["skill_name"] for row in rows]


def _fetch_jd_titles(
    client: bigquery.Client,
    project_id: str,
    industry: str,
    limit: int = _MAX_JD_TITLES,
) -> list[str]:
    """Sample JD titles from the curated dataset for this industry."""
    query = f"""
        SELECT DISTINCT title
        FROM `{project_id}.reskillio.jd_profiles`
        WHERE industry = @industry
          AND title IS NOT NULL
        ORDER BY title
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("industry", "STRING", industry),
            bigquery.ScalarQueryParameter("limit",    "INT64",  limit),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    return [row["title"] for row in rows]


def _auto_detect_industry(candidate_id: str, project_id: str, region: str) -> str:
    """Run industry match and return the top industry key."""
    from reskillio.pipelines.industry_match_pipeline import run_industry_match
    result = run_industry_match(
        candidate_id=candidate_id,
        project_id=project_id,
        region=region,
    )
    return result.top_industry


def _build_prompt(
    target_role: str,
    industry_label: str,
    grounding: NarrativeGrounding,
) -> str:
    candidate_skills_str  = ", ".join(grounding.candidate_top_skills)
    industry_skills_str   = ", ".join(grounding.industry_top_skills)
    overlap_str           = ", ".join(grounding.skill_overlap) or "none directly matched"
    jd_titles_str         = "; ".join(grounding.sample_jd_titles) or "various roles"

    return f"""Write a 3-sentence career rebound narrative for a professional re-entering the {target_role} field.

CANDIDATE FACTS
• Top skills (by demonstrated frequency): {candidate_skills_str}
• Skills that match {industry_label} market demand ({grounding.overlap_count} of {grounding.total_industry_skills} top demanded): {overlap_str}

MARKET CONTEXT
• Target industry: {industry_label}
• Most in-demand skills right now: {industry_skills_str}
• Sample roles actively hiring: {jd_titles_str}

NARRATIVE RULES
Sentence 1 — Candidate strengths: Summarise their demonstrated technical strengths from the skills list.
Sentence 2 — Market fit: Connect those strengths to the current demand in the {industry_label} market.
Sentence 3 — Rebound pathway: State a clear, specific pathway into {target_role} roles, referencing their overlap skills.

Write exactly 3 sentences. Reference only the facts above. Do not add skills, companies, or credentials not listed."""


def _call_gemini(prompt: str, project_id: str, region: str) -> tuple[str, str]:
    from reskillio.utils.gemini import call_gemini, _STUDIO_MODEL, _VERTEX_MODEL, _prefer_studio
    text = call_gemini(prompt, _SYSTEM_INSTRUCTION, project_id, region, temperature=0.2, max_tokens=512)
    model = _STUDIO_MODEL if _prefer_studio() else _VERTEX_MODEL
    return text, model


def run_narrative_pipeline(
    candidate_id: str,
    target_role: str,
    project_id: str,
    region: str = "us-central1",
    industry: Optional[Industry] = None,
) -> NarrativeResult:
    """
    RAG-grounded career narrative generation.

    Parameters
    ----------
    candidate_id:   Candidate whose profile to narrate.
    target_role:    Target job title, e.g. "Senior Data Engineer".
    project_id:     GCP project ID.
    region:         Vertex AI region.
    industry:       Target industry — auto-detected if None.

    Returns
    -------
    NarrativeResult with narrative + full grounding context.
    """
    _apply_credentials()

    # ── 1. Resolve industry ───────────────────────────────────────────────
    if industry is None:
        logger.info(f"Auto-detecting industry for candidate='{candidate_id}'")
        industry_key = _auto_detect_industry(candidate_id, project_id, region)
    else:
        industry_key = industry.value

    industry_label = _INDUSTRY_LABELS.get(industry_key, industry_key)
    logger.info(f"Narrative pipeline: candidate='{candidate_id}' role='{target_role}' industry='{industry_key}'")

    # ── 2. Retrieve candidate top skills from BQ ──────────────────────────
    profile_store = CandidateProfileStore(project_id=project_id)
    profile = profile_store.get_profile(candidate_id)
    candidate_top_skills = [s.skill_name for s in profile.skills[:_TOP_CANDIDATE_SKILLS]]

    # candidate_profiles may be empty (DML MERGE blocked in BQ Sandbox).
    # Fall back to skill_extractions which is written via batch load job.
    if not candidate_top_skills:
        try:
            from reskillio.storage.bigquery_store import BigQuerySkillStore
            rows = BigQuerySkillStore(project_id=project_id).get_skills_for_candidate(candidate_id)
            candidate_top_skills = [r["skill_name"] for r in rows[:_TOP_CANDIDATE_SKILLS]]
            logger.info(f"[narrative] Using {len(candidate_top_skills)} skills from skill_extractions fallback")
        except Exception as exc:
            logger.warning(f"[narrative] skill_extractions fallback failed: {exc}")

    if not candidate_top_skills:
        raise ValueError(f"No skill profile found for candidate '{candidate_id}'.")

    logger.info(f"Candidate top {len(candidate_top_skills)} skills retrieved")

    # ── 3. Retrieve industry demand data from BQ ──────────────────────────
    bq_client = profile_store.client
    industry_top_skills = _fetch_industry_skills(bq_client, project_id, industry_key)

    # No industry seed data yet — generate a skills-only narrative without RAG context.
    # This handles the case where industry_profiles table is empty or not yet seeded.
    if not industry_top_skills:
        logger.warning(
            f"[narrative] No industry_profiles data for '{industry_key}' — "
            "generating skills-only narrative (run seed_market_data.py to enable RAG)"
        )
        skills_str = ", ".join(candidate_top_skills)
        simple_prompt = (
            f"Write a 3-sentence career narrative for a professional targeting {target_role} roles.\n"
            f"Their top skills: {skills_str}.\n"
            "Sentence 1: Summarise their technical strengths in specific terms.\n"
            f"Sentence 2: Explain how those skills directly apply to {target_role} roles.\n"
            "Sentence 3: State a concrete, specific path to get hired.\n"
            "Write in second person. Reference only the listed skills. Be specific, not generic."
        )
        narrative, model_used = _call_gemini(simple_prompt, project_id, region)
        grounding = NarrativeGrounding(
            candidate_top_skills=candidate_top_skills,
            industry_top_skills=[],
            skill_overlap=[],
            overlap_count=0,
            total_industry_skills=0,
            sample_jd_titles=[],
        )
        return NarrativeResult(
            candidate_id=candidate_id,
            target_role=target_role,
            industry=industry_key,
            industry_label=industry_label,
            narrative=narrative,
            grounding=grounding,
            model=model_used,
            generated_at=datetime.now(timezone.utc),
        )

    # ── 4. Retrieve sample JD titles from BQ ─────────────────────────────
    sample_jd_titles = _fetch_jd_titles(bq_client, project_id, industry_key)

    # ── 5. Compute skill overlap ──────────────────────────────────────────
    candidate_lower   = {s.lower() for s in candidate_top_skills}
    industry_lower    = {s.lower() for s in industry_top_skills}
    overlap_lower     = candidate_lower & industry_lower

    # Preserve original casing from industry list for readability
    skill_overlap = [s for s in industry_top_skills if s.lower() in overlap_lower]

    grounding = NarrativeGrounding(
        candidate_top_skills=candidate_top_skills,
        industry_top_skills=industry_top_skills,
        skill_overlap=skill_overlap,
        overlap_count=len(skill_overlap),
        total_industry_skills=len(industry_top_skills),
        sample_jd_titles=sample_jd_titles,
    )

    logger.info(
        f"Grounding built — overlap={len(skill_overlap)}/{len(industry_top_skills)} skills, "
        f"{len(sample_jd_titles)} JD titles"
    )

    # ── 6. Call Gemini Flash ──────────────────────────────────────────────
    prompt = _build_prompt(target_role, industry_label, grounding)
    logger.debug(f"Prompt:\n{prompt}")

    narrative, model_used = _call_gemini(prompt, project_id, region)
    logger.info(f"Narrative generated via {model_used} ({len(narrative)} chars)")

    return NarrativeResult(
        candidate_id=candidate_id,
        target_role=target_role,
        industry=industry_key,
        industry_label=industry_label,
        narrative=narrative,
        grounding=grounding,
        model=model_used,
        generated_at=datetime.now(timezone.utc),
    )