"""
Step 8 — Conversational prompt pipeline.

After /analyze returns and the user lands on the skills screen, this generates
a single, pointed question specific to their archetype and skill profile.

The question is designed to surface what the resume cannot:
  - What energises them now (not what they've always done)
  - What kind of environment they want to land in
  - What problem they want to solve next

Their answer feeds back into /enrich as enrichment_context, sharpening
the market pulse queries and company radar searches.
"""

from __future__ import annotations

import os
import re
from typing import Optional

from loguru import logger

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_SYSTEM_INSTRUCTION = (
    "You are a sharp, empathetic career coach helping displaced professionals "
    "rediscover what they want — not just what they've done. "
    "You ask one precise, open question that gets to the heart of what this "
    "person needs right now. The question must feel personal, never generic. "
    "Never ask 'What are your strengths?' or 'Where do you see yourself in 5 years?' "
    "You respond with only the question — no preamble, no explanation, no quotes."
)

_ARCHETYPE_QUESTIONS = {
    "Builder": "You've built things from scratch before — what scale or type of problem are you most energised to tackle in your next chapter?",
    "Operator": "Your profile shows you run complex systems well — are you looking to go deeper in {industry}, or is there a different domain calling you?",
    "Fixer": "You've turned broken situations around — what kind of 'broken' would you find most meaningful to walk into right now?",
    "Advisor": "You shape decisions for others — whose decisions do you most want to be in the room for next?",
    "Connector": "You bring people together — what kind of collaboration or community do you want to build around in your next role?",
    "Innovator": "You challenge the status quo — what assumption in your field do you most want to help dismantle or rebuild?",
}

_PROMPT_TEMPLATE = """\
A displaced professional just got their skill profile analysed. Help them reflect.

Their professional archetype: {archetype}
Their identity statement: {identity_statement}
Top skills: {skills_csv}
Industry: {industry}

Generate ONE pointed, specific question that will help them articulate what they
want next — in a way that a job search engine can actually use to find better matches.

The question should:
- Be specific to THEIR archetype and background (not generic)
- Surface direction, energy, or environment preference — things a resume hides
- Feel like it comes from someone who read their profile, not a chatbot script
- Be answerable in 2–4 sentences

Respond with only the question. No quotes. No preamble.
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
                temperature=0.7,
                max_output_tokens=200,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[prompt] Vertex AI failed: {e} — trying AI Studio key")

    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for prompt generation — enable Vertex AI or set GEMINI_API_KEY"
        )

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL_STUDIO,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.7,
            max_output_tokens=200,
        ),
    )
    return response.text.strip()


def _fallback_question(archetype: str, industry: str) -> str:
    template = _ARCHETYPE_QUESTIONS.get(archetype, _ARCHETYPE_QUESTIONS["Operator"])
    return template.format(industry=industry)


def run_prompt_pipeline(
    candidate_id: str,
    project_id:   str,
    region:       str = "us-central1",
    archetype:    str = "Operator",
    identity_statement: str = "",
    industry:     str = "your field",
) -> str:
    """
    Generate one pointed question for the candidate based on their skill profile.

    Parameters
    ----------
    candidate_id:       Used to read top skills from BigQuery.
    project_id:         GCP project ID.
    region:             Vertex AI region.
    archetype:          From TraitProfile (Builder/Operator/Fixer/etc.).
    identity_statement: From TraitProfile identity_statement.
    industry:           Human-readable industry label from /analyze.

    Returns
    -------
    str — a single question, no punctuation wrapping.
    """
    _apply_credentials()

    # Read top skills from candidate profile
    skill_names: list[str] = []
    try:
        from reskillio.storage.profile_store import CandidateProfileStore
        store = CandidateProfileStore(project_id=project_id)
        profile = store.get_profile(candidate_id)
        skill_names = [s.skill_name for s in profile.skills[:10]]
    except Exception as exc:
        logger.warning(f"[prompt] Profile read failed: {exc} — using fallback question")
        return _fallback_question(archetype, industry)

    logger.info(
        f"[prompt] Generating question — archetype={archetype} "
        f"industry='{industry}' skills={len(skill_names)}"
    )

    prompt = _PROMPT_TEMPLATE.format(
        archetype=archetype,
        identity_statement=identity_statement or "A seasoned professional re-entering the workforce.",
        skills_csv=", ".join(skill_names) or "general professional skills",
        industry=industry,
    )

    try:
        question = _call_gemini(prompt, project_id, region)
        # Strip surrounding quotes if Gemini added them
        question = re.sub(r'^["\']|["\']$', '', question).strip()
        logger.info(f"[prompt] Generated: {question[:80]}...")
        return question
    except Exception as exc:
        logger.error(f"[prompt] Gemini call failed: {exc} — using fallback")
        return _fallback_question(archetype, industry)
