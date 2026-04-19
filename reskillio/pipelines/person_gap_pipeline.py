"""
Person Gap pipeline.

Generates a personalised gap analysis driven by intake context (what they
loved, what they want next) rather than a generic JD match.

Two modes:
  - intake mode: intake profile found → gap is tied to stated want_next + loved_aspects
  - trait mode:  no intake → falls back to archetype + top industry from /analyze

Single Gemini call returns structured JSON.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from loguru import logger

from reskillio.models.person_gap import AlignedSkill, GrowthSkill, PersonGapResult

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_SYSTEM = (
    "You are a precise career coach who gives honest, specific, actionable gap analysis. "
    "You ground every insight in the person's actual stated goals and real skill profile. "
    "Never be generic. Never hallucinate skills they didn't have. "
    "Respond ONLY with valid JSON — no markdown, no commentary."
)

_PROMPT_TEMPLATE = """\
Candidate skill profile:
{skills_block}

Their professional archetype: {archetype}
Their identity statement: {identity_statement}

Goal context ({context_mode}):
{goal_context}

Generate a personalised gap analysis. Return JSON with exactly these keys:
{{
  "narrative": "2–3 sentence personalised context tying their background to their stated goal",
  "aligned_skills": [
    {{"skill": "...", "relevance": "one sentence — why this matters for their goal"}}
    // 4–6 skills from their profile that are genuinely relevant
  ],
  "growth_skills": [
    {{
      "skill": "...",
      "priority": "high|medium|low",
      "why_needed": "one sentence",
      "how_to_build": "specific resource, course, or approach — not generic advice"
    }}
    // 4–6 skills to develop, ordered by priority
  ],
  "surprise_transfers": [
    "one-sentence description of a transferable strength they may undervalue"
    // 2–3 items
  ],
  "readiness_score": 0.0,  // float 0–1: how ready are they right now for their stated goal
  "recommended_actions": [
    "Specific next step"
    // 4–5 concrete, time-boxed actions (e.g. "Complete Coursera ML Specialisation — 8 weeks")
  ]
}}
"""

_GOAL_INTAKE = """\
What they loved about their career: {loved_aspects}
What they want next: {want_next}
Open to fractional/consulting: {open_to_fractional}
Engagement format preference: {engagement_format}
"""

_GOAL_TRAIT = """\
Top industry match: {industry}
Ideal company stage: {ideal_stage}
Work values: {work_values}
"""


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


def _call_gemini(prompt: str, project_id: str, region: str) -> dict:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

    from google import genai
    from google.genai import types as gt

    try:
        client = genai.Client(vertexai=True, project=project_id, location=region)
        response = client.models.generate_content(
            model=_MODEL_VERTEX,
            contents=prompt,
            config=gt.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.3,
                max_output_tokens=1200,
                thinking_config=gt.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip()
    except Exception as e:
        logger.warning(f"[person_gap] Vertex failed: {e}, trying AI Studio")
        try:
            from config.settings import settings
            api_key = settings.gemini_api_key
        except Exception:
            api_key = ""
        if not api_key:
            raise RuntimeError("Gemini unavailable") from e
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_MODEL_STUDIO,
            contents=prompt,
            config=gt.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.3,
                max_output_tokens=1200,
            ),
        )
        text = response.text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


def run_person_gap(
    candidate_id:       str,
    project_id:         str,
    region:             str   = "us-central1",
    archetype:          str   = "Operator",
    identity_statement: str   = "",
    industry:           str   = "your field",
    ideal_stage:        str   = "Growth-stage",
    work_values:        list  = None,
    # intake fields (optional)
    loved_aspects:      str   = "",
    want_next:          str   = "",
    open_to_fractional: bool  = False,
    engagement_format:  str   = "",
    fallback_skills:    list  = None,
) -> PersonGapResult:
    _apply_credentials()

    # Read skill profile from BigQuery; use fallback_skills if BQ is unavailable or empty
    skill_names: list[str] = []
    skill_block_lines: list[str] = []
    try:
        from reskillio.storage.profile_store import CandidateProfileStore
        store = CandidateProfileStore(project_id=project_id)
        profile = store.get_profile(candidate_id)
        skills = profile.skills[:20]
        skill_names = [s.skill_name for s in skills]
        skill_block_lines = [
            f"  - {s.skill_name} (category: {s.category}, confidence: {s.confidence_score:.2f})"
            for s in skills
        ]
    except Exception as exc:
        logger.warning(f"[person_gap] Profile read failed: {exc}")

    # If BQ gave nothing, use the skills passed directly from the frontend
    if not skill_block_lines and fallback_skills:
        skill_names = fallback_skills
        skill_block_lines = [f"  - {s}" for s in fallback_skills]
        logger.info(f"[person_gap] Using {len(fallback_skills)} fallback skills from request")

    skills_block = "\n".join(skill_block_lines) or "  (no skill profile available)"

    # Choose goal context mode
    use_intake = bool(loved_aspects or want_next)
    if use_intake:
        goal_context = _GOAL_INTAKE.format(
            loved_aspects=loved_aspects or "not specified",
            want_next=want_next or "not specified",
            open_to_fractional="yes" if open_to_fractional else "no",
            engagement_format=engagement_format or "not specified",
        )
        context_mode = "from intake conversation"
        context_used = "intake"
    else:
        goal_context = _GOAL_TRAIT.format(
            industry=industry,
            ideal_stage=ideal_stage,
            work_values=", ".join(work_values or ["Growth", "Impact"]),
        )
        context_mode = "inferred from trait profile"
        context_used = "trait_only"

    logger.info(
        f"[person_gap] candidate={candidate_id} archetype={archetype} "
        f"mode={context_used} skills={len(skill_names)}"
    )

    prompt = _PROMPT_TEMPLATE.format(
        skills_block=skills_block,
        archetype=archetype,
        identity_statement=identity_statement or "A seasoned professional navigating a career transition.",
        context_mode=context_mode,
        goal_context=goal_context,
    )

    try:
        data = _call_gemini(prompt, project_id, region)
    except Exception as exc:
        logger.error(f"[person_gap] Gemini failed: {exc}")
        return PersonGapResult(
            candidate_id=candidate_id,
            narrative=f"As a {archetype}, your profile shows strong transferable skills relevant to {industry}.",
            aligned_skills=[],
            growth_skills=[],
            surprise_transfers=[],
            readiness_score=0.5,
            recommended_actions=["Complete your intake to get a personalised plan."],
            context_used=context_used,
        )

    def _aligned(raw: list) -> list[AlignedSkill]:
        out = []
        for item in raw[:6]:
            try:
                out.append(AlignedSkill(skill=item["skill"], relevance=item.get("relevance", "")))
            except Exception:
                pass
        return out

    def _growth(raw: list) -> list[GrowthSkill]:
        out = []
        for item in raw[:6]:
            try:
                out.append(GrowthSkill(
                    skill=item["skill"],
                    priority=item.get("priority", "medium"),
                    why_needed=item.get("why_needed", ""),
                    how_to_build=item.get("how_to_build", ""),
                ))
            except Exception:
                pass
        return out

    return PersonGapResult(
        candidate_id=candidate_id,
        narrative=data.get("narrative", ""),
        aligned_skills=_aligned(data.get("aligned_skills", [])),
        growth_skills=_growth(data.get("growth_skills", [])),
        surprise_transfers=data.get("surprise_transfers", [])[:3],
        readiness_score=min(1.0, max(0.0, float(data.get("readiness_score", 0.5)))),
        recommended_actions=data.get("recommended_actions", [])[:5],
        context_used=context_used,
    )
