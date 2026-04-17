"""
Trait inference pipeline — Stage 1.5 of the analyze orchestration.

Analyzes resume text + optional free-text context using Gemini to produce
a TraitProfile: professional archetype, work values, decision style,
ideal company stage, identity statement, and hidden strengths.

The free-text context (candidate's own words) is weighted heavily when
present — it surfaces signals the resume suppresses.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from loguru import logger

from reskillio.models.trait import TraitProfile

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_SYSTEM_INSTRUCTION = (
    "You are an expert career psychologist and talent analyst specialising in "
    "displaced professionals re-entering the workforce. You read between the lines "
    "of career histories to identify who someone truly is beyond their job title. "
    "You are empathetic, precise, and never generic. "
    "You respond only with valid JSON — never markdown, never explanation."
)

_ARCHETYPE_DEFINITIONS = """\
ARCHETYPE DEFINITIONS — choose the single most dominant one:
  Builder    — Creates teams, systems, or functions from scratch. Scales things. Thrives in ambiguity.
  Operator   — Runs complex systems reliably at scale. Optimises and maintains. Thrives in complexity.
  Fixer      — Diagnoses broken situations and turns them around. The person parachuted in when things go wrong.
  Advisor    — Synthesises complexity and shapes decisions for others. Trusted counsel. Thrives through influence.
  Connector  — Works across silos. Relationship capital is their primary tool. Thrives in collaboration.
  Innovator  — Challenges existing approaches and introduces new methods. Questions the status quo. Thrives at the frontier.\
"""

_PROMPT_TEMPLATE = """\
Analyse this professional's career history to build their trait profile.

RESUME:
{resume_text}
{context_block}
{archetype_defs}

WORK VALUES — pick the top 2–3 that best fit this person:
  Stability · Growth · Impact · Autonomy · Craft

DECISION STYLE — pick the single most dominant one:
  Data-driven · Intuitive · Collaborative · Directive

IDEAL COMPANY STAGE — where would this person genuinely thrive:
  Startup · Growth-stage · Enterprise · Turnaround

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "archetype":          "<one of the 6 archetypes>",
  "archetype_reason":   "<one sentence citing specific evidence from their career>",
  "work_values":        ["<value1>", "<value2>"],
  "decision_style":     "<one of the 4 styles>",
  "ideal_stage":        "<one of the 4 stages>",
  "identity_statement": "<one sentence starting with 'You are a ...' — specific, never generic>",
  "hidden_strengths":   ["<strength the resume buries>", "<strength2>", "<strength3>"]
}}

Constraints:
- identity_statement: must be specific to this person's actual history, never a generic phrase like "results-driven professional"
- hidden_strengths: surface what the resume downplays or omits entirely (e.g. crisis leadership, institutional knowledge, culture-building)
- archetype_reason: cite a specific role, achievement, or recurring pattern from the resume
- If additional context text is provided, weight it heavily — the candidate is telling you what the resume cannot
"""


def _build_context_block(context_text: Optional[str]) -> str:
    if not context_text or not context_text.strip():
        return ""
    return (
        "\nADDITIONAL CONTEXT (candidate's own words — treat as high-signal input):\n"
        + context_text.strip()
        + "\n"
    )


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

    # Path 1 — Vertex AI via google-genai SDK
    try:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(vertexai=True, project=project_id, location=region)
        response = client.models.generate_content(
            model=_MODEL_VERTEX,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.2,
                max_output_tokens=1024,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[trait] Vertex AI failed: {e} — trying AI Studio key")

    # Path 2 — AI Studio API key fallback
    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for trait inference — enable Vertex AI or set GEMINI_API_KEY"
        )

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL_STUDIO,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
            max_output_tokens=1024,
            candidate_count=1,
        ),
    )
    return response.text.strip()


def _parse_response(raw: str) -> Optional[dict]:
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        return json.loads(clean)
    except Exception as exc:
        logger.warning(f"[trait] JSON parse failed: {exc} — raw snippet: {raw[:300]}")
        return None


_VALID_ARCHETYPES  = {"Builder", "Operator", "Fixer", "Advisor", "Connector", "Innovator"}
_VALID_VALUES      = {"Stability", "Growth", "Impact", "Autonomy", "Craft"}
_VALID_STYLES      = {"Data-driven", "Intuitive", "Collaborative", "Directive"}
_VALID_STAGES      = {"Startup", "Growth-stage", "Enterprise", "Turnaround"}


def _safe_fallback() -> TraitProfile:
    return TraitProfile(
        archetype="Operator",
        archetype_reason="Could not infer from available data.",
        work_values=["Growth", "Impact"],
        decision_style="Collaborative",
        ideal_stage="Enterprise",
        identity_statement="You are a seasoned professional with strong transferable skills.",
        hidden_strengths=["Cross-functional collaboration", "Problem solving under pressure"],
    )


def run_trait_inference(
    resume_text:  str,
    project_id:   str,
    region:       str = "us-central1",
    context_text: Optional[str] = None,
) -> TraitProfile:
    """
    Infer professional trait profile from resume text + optional free-text context.

    Parameters
    ----------
    resume_text:   Raw text extracted from the resume.
    project_id:    GCP project ID (Vertex AI).
    region:        Vertex AI region.
    context_text:  Candidate's own words — "what your resume doesn't show."
                   When provided, weighted heavily in archetype and value inference.

    Returns
    -------
    TraitProfile with archetype, work values, decision style, identity statement,
    and hidden strengths.
    """
    _apply_credentials()

    prompt = _PROMPT_TEMPLATE.format(
        resume_text=resume_text[:4000],      # cap for token budget
        context_block=_build_context_block(context_text),
        archetype_defs=_ARCHETYPE_DEFINITIONS,
    )

    logger.info(
        f"[trait] Running inference — "
        f"resume={len(resume_text)} chars  context={'yes' if context_text else 'no'}"
    )

    try:
        raw = _call_gemini(prompt, project_id, region)
    except Exception as exc:
        logger.error(f"[trait] Gemini call failed: {exc}")
        return _safe_fallback()

    logger.debug(f"[trait] Raw response: {raw[:400]}")

    data = _parse_response(raw)
    if not data:
        return _safe_fallback()

    archetype = data.get("archetype", "Operator")
    if archetype not in _VALID_ARCHETYPES:
        archetype = "Operator"

    work_values = [v for v in data.get("work_values", []) if v in _VALID_VALUES][:3]
    if not work_values:
        work_values = ["Growth", "Impact"]

    decision_style = data.get("decision_style", "Collaborative")
    if decision_style not in _VALID_STYLES:
        decision_style = "Collaborative"

    ideal_stage = data.get("ideal_stage", "Enterprise")
    if ideal_stage not in _VALID_STAGES:
        ideal_stage = "Enterprise"

    result = TraitProfile(
        archetype=archetype,
        archetype_reason=data.get("archetype_reason", ""),
        work_values=work_values,
        decision_style=decision_style,
        ideal_stage=ideal_stage,
        identity_statement=data.get("identity_statement", ""),
        hidden_strengths=(data.get("hidden_strengths") or [])[:3],
    )

    logger.info(
        f"[trait] Done — archetype={result.archetype}  "
        f"values={result.work_values}  style={result.decision_style}  "
        f"stage={result.ideal_stage}"
    )
    return result
