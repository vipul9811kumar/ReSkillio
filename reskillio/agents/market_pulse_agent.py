"""
Step 3 — MarketPulseAgent.

DDG search + single Gemini synthesis (faster than CrewAI — no agent loop overhead).

Given a candidate's top skills and industry label, searches DuckDuckGo for
currently-hiring roles and synthesises a MarketPulseResult: top 5 roles actively
hiring for this profile, overall demand signal, and a plain-language market summary.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Optional

from ddgs import DDGS
from loguru import logger

from reskillio.models.enrich import MarketPulseResult, RoleSignal

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"
_DDG_RESULTS  = 4   # snippets per query

_SYSTEM_INSTRUCTION = (
    "You are a senior labour-market analyst specialising in career transitions for "
    "displaced professionals. You synthesise real web search data into honest, "
    "specific hiring-market intelligence. You never fabricate job titles or companies. "
    "You respond only with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
A displaced professional is looking to re-enter the workforce.
Their skill profile: {skills_csv}
Their closest industry: {industry}
Their stated target: {target_role}

Below are fresh DuckDuckGo search results about the current job market for this profile:

{search_snippets}

Based ONLY on the search evidence above (plus your general knowledge of the current \
hiring market), identify the top 5 job roles that are actively hiring for someone \
with this exact skill profile.

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "top_roles": [
    {{
      "title":             "<job title>",
      "hiring_activity":   "<high|medium|low>",
      "top_skills_needed": ["<skill1>", "<skill2>", "<skill3>"],
      "evidence":          "<one sentence citing search evidence>"
    }}
  ],
  "overall_demand": "<high|medium|low>",
  "market_summary": "<2–3 sentences: what the market looks like for this person right now>",
  "data_sources":   ["DuckDuckGo web search (real-time)"]
}}

Constraints:
- Return exactly 5 roles in top_roles, ranked by hiring activity (highest first).
- top_skills_needed: list the 3 skills from their profile that most drive hiring for each role.
- evidence: must reference something concrete from the search snippets (company, trend, or stat).
- market_summary: be honest. If the market is tough, say so — and name one concrete opportunity.
- hiring_activity: high = many active postings; medium = moderate; low = specialist/niche demand.
"""


def _ddg_search(query: str, max_results: int = _DDG_RESULTS) -> list[str]:
    snippets: list[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in (results or []):
                title = r.get("title", "")
                body  = r.get("body", "")[:250]
                snippets.append(f"[{title}] {body}")
    except Exception as exc:
        logger.warning(f"[market-pulse] DDG search failed for '{query}': {exc}")
    return snippets


def _gather_snippets(skills: list[str], industry: str, target_role: str) -> str:
    """Run DDG queries in parallel and return concatenated snippets."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    top_skills = ", ".join(skills[:4])
    queries = [
        f"{industry} jobs hiring 2025 {top_skills}",
        f"{target_role} hiring demand 2025",
    ]
    all_snippets: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_ddg_search, q, _DDG_RESULTS): q for q in queries}
        for future in as_completed(futures, timeout=10):
            q = futures[future]
            try:
                found = future.result(timeout=3)
                all_snippets.extend(found)
                logger.debug(f"[market-pulse] Query '{q}' → {len(found)} snippets")
            except Exception as exc:
                logger.warning(f"[market-pulse] Query '{q}' failed: {exc}")

    if not all_snippets:
        return "No live search results available — synthesise from general market knowledge."

    return "\n".join(all_snippets[:16])  # cap to avoid token overflow


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
                temperature=0.3,
                max_output_tokens=1500,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[market-pulse] Vertex AI failed: {e} — trying AI Studio key")

    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for market pulse — enable Vertex AI or set GEMINI_API_KEY"
        )

    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=_MODEL_STUDIO,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=1500,
        ),
    )
    return response.text.strip()


def _parse_response(raw: str) -> Optional[dict]:
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        return json.loads(clean)
    except Exception as exc:
        logger.warning(f"[market-pulse] JSON parse failed: {exc} — snippet: {raw[:300]}")
        return None


_VALID_ACTIVITY = {"high", "medium", "low"}


def _safe_fallback(skills: list[str], industry: str) -> MarketPulseResult:
    return MarketPulseResult(
        top_roles=[
            RoleSignal(
                title="Operations Manager",
                hiring_activity="medium",
                top_skills_needed=skills[:3] or ["Operations", "Leadership", "Communication"],
                evidence="Broad demand across industries for operations leadership.",
            ),
            RoleSignal(
                title="Project Manager",
                hiring_activity="medium",
                top_skills_needed=skills[:3] or ["Project Management", "Stakeholder Management", "Planning"],
                evidence="Consistent demand for project management expertise.",
            ),
            RoleSignal(
                title="Business Analyst",
                hiring_activity="medium",
                top_skills_needed=skills[:3] or ["Analysis", "Data", "Communication"],
                evidence="Growing demand for analytical roles across sectors.",
            ),
            RoleSignal(
                title="Strategy Consultant",
                hiring_activity="low",
                top_skills_needed=skills[:3] or ["Strategy", "Analysis", "Problem Solving"],
                evidence="Specialist demand in consulting firms.",
            ),
            RoleSignal(
                title=f"{industry} Specialist",
                hiring_activity="low",
                top_skills_needed=skills[:3] or ["Domain Expertise", "Communication", "Leadership"],
                evidence=f"Domain expertise in {industry} valued by specialist employers.",
            ),
        ],
        overall_demand="medium",
        market_summary=(
            f"The market for professionals with your {industry} background shows steady demand "
            "across operations, project management, and analytical roles. "
            "Focus on roles that leverage your specific domain expertise and transferable skills."
        ),
        data_sources=["Fallback — live search unavailable"],
        analyzed_at=datetime.now(timezone.utc),
    )


def run_market_pulse_agent(
    skill_names:   list[str],
    industry:      str,
    target_role:   str,
    project_id:    str,
    region:        str = "us-central1",
) -> MarketPulseResult:
    """
    Find top 5 roles actively hiring for a candidate's skill/industry profile.

    DDG searches gather live hiring signals; a single Gemini call synthesises
    them into structured MarketPulseResult. Faster than CrewAI — no agent loop.

    Parameters
    ----------
    skill_names:  Top skills from the candidate's profile (first 10 used).
    industry:     Human-readable industry label (e.g. "Technology & Software").
    target_role:  Target role string from /analyze (may be "Career Transition").
    project_id:   GCP project ID (Vertex AI).
    region:       Vertex AI region.

    Returns
    -------
    MarketPulseResult with top 5 roles, demand signal, and market summary.
    """
    _apply_credentials()
    skills = skill_names[:10]

    logger.info(
        f"[market-pulse] Starting — {len(skills)} skills, "
        f"industry='{industry}', role='{target_role}'"
    )

    snippets = _gather_snippets(skills, industry, target_role)

    prompt = _PROMPT_TEMPLATE.format(
        skills_csv=", ".join(skills),
        industry=industry,
        target_role=target_role,
        search_snippets=snippets,
    )

    try:
        raw = _call_gemini(prompt, project_id, region)
    except Exception as exc:
        logger.error(f"[market-pulse] Gemini call failed: {exc} — using fallback")
        return _safe_fallback(skills, industry)

    logger.debug(f"[market-pulse] Raw response: {raw[:400]}")

    data = _parse_response(raw)
    if not data:
        return _safe_fallback(skills, industry)

    roles_raw = data.get("top_roles") or []
    top_roles: list[RoleSignal] = []
    for r in roles_raw[:5]:
        activity = r.get("hiring_activity", "medium")
        if activity not in _VALID_ACTIVITY:
            activity = "medium"
        top_roles.append(RoleSignal(
            title=r.get("title", "Unknown Role"),
            hiring_activity=activity,
            top_skills_needed=(r.get("top_skills_needed") or [])[:3],
            evidence=r.get("evidence", ""),
        ))

    if not top_roles:
        return _safe_fallback(skills, industry)

    overall = data.get("overall_demand", "medium")
    if overall not in _VALID_ACTIVITY:
        overall = "medium"

    result = MarketPulseResult(
        top_roles=top_roles,
        overall_demand=overall,
        market_summary=data.get("market_summary", ""),
        data_sources=data.get("data_sources") or ["DuckDuckGo web search (real-time)"],
        analyzed_at=datetime.now(timezone.utc),
    )

    logger.info(
        f"[market-pulse] Done — {len(top_roles)} roles, "
        f"demand={result.overall_demand}, "
        f"top='{top_roles[0].title}'"
    )
    return result
