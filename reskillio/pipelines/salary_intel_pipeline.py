"""
Step 5 — Salary Intelligence pipeline.

DDG search for current salary data + single Gemini synthesis.
Produces a skill-attributed salary band (floor/median/ceiling) and explains
which specific skills push the number up or down for this candidate.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from ddgs import DDGS
from loguru import logger

from reskillio.models.enrich import SalaryDriver, SalaryIntelResult

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"
_DDG_RESULTS  = 4

_SYSTEM_INSTRUCTION = (
    "You are a compensation specialist with deep knowledge of US tech and professional "
    "services salary markets. You combine live salary data with skill-level analysis to "
    "give displaced professionals an honest, specific picture of their earning potential. "
    "You always cite which skills drive the number up or down. "
    "You respond only with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
A displaced professional is re-entering the workforce.
Target role: {target_role}
Industry: {industry}
Top skills: {skills_csv}

Fresh salary search results:
{search_snippets}

Using the search data above and your knowledge of current US compensation benchmarks,
produce a realistic salary band for someone with this EXACT profile.

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "floor_usd":   <integer — 10th percentile, entry-level or under-qualified for role>,
  "median_usd":  <integer — realistic mid-point for this specific skill set>,
  "ceiling_usd": <integer — top of range for someone who can demonstrate all listed skills>,
  "drivers": [
    {{
      "label":     "<specific skill or trait from their profile>",
      "direction": "<up|down>",
      "delta_usd": <integer — rough $ impact on median, positive value>,
      "reason":    "<one sentence: why this skill moves the number>"
    }}
  ],
  "note": "<one sentence: context e.g. 'US national median, mid-career {industry} professional'>"
}}

Constraints:
- floor/median/ceiling must be realistic annual USD figures (full integers, no decimals).
- ceiling must be > median > floor.
- Provide 4–6 drivers: mix of up-movers and down-movers. Be specific — name actual skills.
- At least 2 drivers should be "up" (premium skills) and at least 1 "down" (gap or risk).
- delta_usd: realistic $ delta, typically $3,000–$25,000 per driver.
- note: must mention the industry and seniority level implied by the skill set.
"""


def _ddg_search(query: str, max_results: int = _DDG_RESULTS) -> list[str]:
    snippets: list[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in (results or []):
                snippets.append(f"[{r.get('title','')}] {r.get('body','')[:250]}")
    except Exception as exc:
        logger.warning(f"[salary] DDG search failed for '{query}': {exc}")
    return snippets


def _gather_snippets(target_role: str, industry: str, skills: list[str]) -> str:
    top_skills = ", ".join(skills[:4])
    queries = [
        f"{target_role} salary 2025 US",
        f"{industry} professional salary range 2025",
        f"{top_skills} skills salary premium 2025",
    ]
    all_snippets: list[str] = []
    for q in queries:
        found = _ddg_search(q)
        all_snippets.extend(found)
    if not all_snippets:
        return "No live salary data found — use general US compensation benchmarks."
    return "\n".join(all_snippets[:18])


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
                temperature=0.2,
                max_output_tokens=1000,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[salary] Vertex AI failed: {e} — trying AI Studio key")

    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for salary intel — enable Vertex AI or set GEMINI_API_KEY"
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
            max_output_tokens=1000,
        ),
    )
    return response.text.strip()


def _safe_fallback(target_role: str, industry: str) -> SalaryIntelResult:
    return SalaryIntelResult(
        floor_usd=65000,
        median_usd=95000,
        ceiling_usd=130000,
        drivers=[
            SalaryDriver(
                label="Domain expertise",
                direction="up",
                delta_usd=10000,
                reason="Sector-specific knowledge commands a premium over generalists.",
            ),
            SalaryDriver(
                label="Career gap",
                direction="down",
                delta_usd=8000,
                reason="Re-entry candidates often start at a slight discount until they rebuild recent project history.",
            ),
        ],
        note=f"US national estimate for {target_role} in {industry}. Live data unavailable.",
    )


def run_salary_intel(
    skill_names:  list[str],
    industry:     str,
    target_role:  str,
    project_id:   str,
    region:       str = "us-central1",
) -> SalaryIntelResult:
    """
    Produce a skill-attributed salary band for the candidate's profile.

    DDG searches gather live salary signals; a single Gemini call synthesises
    them with skill-level analysis into floor/median/ceiling + driver explanations.

    Parameters
    ----------
    skill_names:  Top skills from the candidate's profile.
    industry:     Human-readable industry label.
    target_role:  Target role from /analyze (or best-fit role from MarketPulse).
    project_id:   GCP project ID.
    region:       Vertex AI region.

    Returns
    -------
    SalaryIntelResult with floor/median/ceiling USD and per-skill drivers.
    """
    _apply_credentials()
    skills = skill_names[:12]

    logger.info(
        f"[salary] Starting — role='{target_role}' industry='{industry}' "
        f"skills={len(skills)}"
    )

    snippets = _gather_snippets(target_role, industry, skills)

    prompt = _PROMPT_TEMPLATE.format(
        target_role=target_role,
        industry=industry,
        skills_csv=", ".join(skills),
        search_snippets=snippets,
    )

    try:
        raw = _call_gemini(prompt, project_id, region)
    except Exception as exc:
        logger.error(f"[salary] Gemini call failed: {exc} — using fallback")
        return _safe_fallback(target_role, industry)

    logger.debug(f"[salary] Raw response: {raw[:400]}")

    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(clean)
    except Exception as exc:
        logger.warning(f"[salary] JSON parse failed: {exc}")
        return _safe_fallback(target_role, industry)

    floor   = int(data.get("floor_usd",  65000))
    median  = int(data.get("median_usd", 95000))
    ceiling = int(data.get("ceiling_usd", 130000))

    # Sanity-check ordering
    if not (floor < median < ceiling):
        logger.warning(f"[salary] Band ordering invalid ({floor}/{median}/{ceiling}) — using fallback")
        return _safe_fallback(target_role, industry)

    drivers: list[SalaryDriver] = []
    for d in (data.get("drivers") or [])[:6]:
        direction = d.get("direction", "up")
        if direction not in ("up", "down"):
            direction = "up"
        drivers.append(SalaryDriver(
            label=d.get("label", "Skill"),
            direction=direction,
            delta_usd=abs(int(d.get("delta_usd", 5000))),
            reason=d.get("reason", ""),
        ))

    result = SalaryIntelResult(
        floor_usd=floor,
        median_usd=median,
        ceiling_usd=ceiling,
        drivers=drivers,
        note=data.get("note", f"US national, {industry}"),
    )

    logger.info(
        f"[salary] Done — ${floor:,} / ${median:,} / ${ceiling:,}  "
        f"drivers={len(drivers)}"
    )
    return result
