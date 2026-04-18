"""
Step 4 — Auto-gap pipeline (JD-less).

Takes the top roles from MarketPulseResult as synthetic JDs.
A single Gemini call expands each role title into ~10 required skills;
the candidate's stored skill profile is then matched against each role
to produce readiness scores and a ranked list of skills to add.

No JD paste required — this runs automatically in the background.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Optional

from loguru import logger

from reskillio.models.enrich import AutoGapResult, RoleGap, RoleSignal

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_SYSTEM_INSTRUCTION = (
    "You are a senior technical recruiter and job market specialist. "
    "You know exactly which skills are required for each professional role. "
    "You respond only with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
For each job role listed below, provide the 10 most commonly required skills
that appear in real job postings. Focus on skills a hiring manager would
actually filter candidates on — not generic traits like "communication."

Roles to expand:
{roles_list}

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "roles": [
    {{
      "title": "<exact role title from input>",
      "required_skills": ["<skill1>", "<skill2>", ..., "<skill10>"]
    }}
  ]
}}

Constraints:
- 10 skills per role, no more, no less.
- Skills should be specific and searchable (e.g. "SQL", "Stakeholder Management", "Python", "Excel").
- Include both technical and domain-specific soft skills relevant to the role.
- required_skills must be an array of 10 strings.
"""


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                settings.google_application_credentials
            )
    except Exception:
        pass


def _call_gemini(prompt: str, project_id: str, region: str) -> str:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(vertexai=True, project=project_id, location=region)
        response = client.models.generate_content(
            model=_MODEL_VERTEX,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.1,
                max_output_tokens=1200,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[auto-gap] Vertex AI failed: {e} — trying AI Studio key")

    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for auto-gap — enable Vertex AI or set GEMINI_API_KEY"
        )

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL_STUDIO,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=1200,
        ),
    )
    return response.text.strip()


def _expand_roles(
    roles: list[RoleSignal],
    project_id: str,
    region: str,
) -> dict[str, list[str]]:
    """
    Returns {role_title: [required_skill, ...]} via a single Gemini call.
    Falls back to using each role's top_skills_needed if Gemini fails.
    """
    roles_list = "\n".join(f"- {r.title}" for r in roles)
    prompt = _PROMPT_TEMPLATE.format(roles_list=roles_list)

    try:
        raw = _call_gemini(prompt, project_id, region)
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(clean)
        result: dict[str, list[str]] = {}
        for entry in data.get("roles", []):
            title = entry.get("title", "")
            skills = entry.get("required_skills", [])[:10]
            if title and skills:
                result[title] = skills
        if result:
            return result
    except Exception as exc:
        logger.warning(f"[auto-gap] Role expansion failed: {exc} — using top_skills_needed")

    # Fallback: use the top_skills_needed from market pulse (3 per role)
    return {r.title: list(r.top_skills_needed) for r in roles}


def _compute_gap(
    candidate_skills_lower: set[str],
    required_skills: list[str],
    role_title: str,
) -> RoleGap:
    matched: list[str] = []
    missing: list[str] = []

    for skill in required_skills:
        if skill.lower() in candidate_skills_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    total = len(required_skills)
    if total == 0:
        gap_score = 50.0
    else:
        gap_score = round(min(100.0, (len(matched) / total) * 100), 1)

    return RoleGap(
        role_title=role_title,
        gap_score=gap_score,
        matched_skills=matched,
        missing_skills=missing,
    )


def run_auto_gap(
    top_roles:    list[RoleSignal],
    candidate_id: str,
    project_id:   str,
    region:       str = "us-central1",
) -> AutoGapResult:
    """
    JD-less gap analysis using MarketPulse top roles as synthetic JDs.

    Parameters
    ----------
    top_roles:    RoleSignal list from MarketPulseResult.top_roles.
    candidate_id: Candidate whose profile to read from BigQuery.
    project_id:   GCP project ID.
    region:       Vertex AI region.

    Returns
    -------
    AutoGapResult with per-role readiness scores and top skills to add.
    """
    if not top_roles:
        raise ValueError("top_roles is empty — run MarketPulse first")

    _apply_credentials()

    logger.info(
        f"[auto-gap] Starting — {len(top_roles)} roles for candidate='{candidate_id}'"
    )

    # Read candidate skill set from BigQuery
    try:
        from reskillio.storage.profile_store import CandidateProfileStore
        profile_store = CandidateProfileStore(project_id=project_id)
        profile = profile_store.get_profile(candidate_id)
        candidate_skills_lower = {s.skill_name.lower() for s in profile.skills}
    except Exception as exc:
        logger.warning(f"[auto-gap] Could not load candidate profile: {exc} — using empty set")
        candidate_skills_lower = set()

    logger.info(f"[auto-gap] Candidate has {len(candidate_skills_lower)} skills in profile")

    # Expand roles → required skills via single Gemini call
    role_requirements = _expand_roles(top_roles, project_id, region)

    # Compute gap per role
    role_gaps: list[RoleGap] = []
    missing_counter: Counter = Counter()

    for role in top_roles:
        required = role_requirements.get(role.title, list(role.top_skills_needed))
        if not required:
            continue
        rg = _compute_gap(candidate_skills_lower, required, role.title)
        role_gaps.append(rg)
        for skill in rg.missing_skills:
            missing_counter[skill] += 1

    if not role_gaps:
        raise ValueError("No role gaps could be computed")

    overall_readiness = round(
        sum(rg.gap_score for rg in role_gaps) / len(role_gaps), 1
    )
    # Top skills to add: ranked by how many roles they're missing from
    top_skills_to_add = [skill for skill, _ in missing_counter.most_common(8)]

    logger.info(
        f"[auto-gap] Done — {len(role_gaps)} roles, "
        f"overall_readiness={overall_readiness:.1f}, "
        f"top_missing={top_skills_to_add[:3]}"
    )

    return AutoGapResult(
        role_gaps=role_gaps,
        overall_readiness=overall_readiness,
        top_skills_to_add=top_skills_to_add,
    )
