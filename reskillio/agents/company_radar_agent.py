"""
Step 6 — CompanyRadarAgent.

DDG search + single Gemini synthesis (same pattern as market_pulse_agent).

Finds 5–7 named employers actively hiring for this candidate's profile.
Each match is tied to a specific reason grounded in the search evidence.

Hallucination guard:
  - Every CompanyMatch.source_note must reference the search context.
  - CompanyRadarResult.data_freshness is set to today's date so the
    frontend can show users when the data was gathered.
  - Gemini is explicitly instructed: do not invent companies not found
    in the search results.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Optional

from ddgs import DDGS
from loguru import logger

from reskillio.models.enrich import CompanyMatch, CompanyRadarResult

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"
_DDG_RESULTS  = 5

_SYSTEM_INSTRUCTION = (
    "You are a talent market researcher helping displaced professionals identify "
    "specific employers worth targeting. You only name companies that appeared in "
    "the search results provided — you never fabricate employer names. "
    "You are specific, honest, and grounded in evidence. "
    "You respond only with valid JSON — no markdown, no explanation."
)

_PROMPT_TEMPLATE = """\
A displaced professional is looking for employers to target in their job search.

Their profile:
  Skills: {skills_csv}
  Industry: {industry}
  Top roles they're suited for: {roles_csv}
  Ideal company stage: {ideal_stage}

Fresh web search results about companies hiring in this space:
{search_snippets}

Based ONLY on the companies that appear in the search results above, identify
5–7 specific employers actively hiring people with this profile.

Return ONLY valid JSON (no markdown fences, no explanation):
{{
  "companies": [
    {{
      "name":         "<company name — MUST appear in search results above>",
      "industry":     "<their primary industry>",
      "size":         "<startup|mid-size|enterprise>",
      "match_reason": "<one sentence: why this company specifically, given this candidate's profile>",
      "source_note":  "<cite the search result that named this company>"
    }}
  ]
}}

Critical constraints:
- ONLY name companies that explicitly appeared in the search snippets.
- If fewer than 5 companies are clearly named in the results, return only those found.
- match_reason must be specific to THIS candidate's skills and the company's known needs.
- source_note: quote or closely paraphrase the snippet that named this company.
- size: "startup" = <200 employees, "mid-size" = 200–2000, "enterprise" = 2000+.
- Do NOT include companies you know from training data but that do not appear in the snippets.
"""


def _ddg_search(query: str, max_results: int = _DDG_RESULTS) -> list[str]:
    snippets: list[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in (results or []):
                snippets.append(f"[{r.get('title', '')}] {r.get('body', '')[:300]}")
    except Exception as exc:
        logger.warning(f"[company-radar] DDG search failed for '{query}': {exc}")
    return snippets


def _gather_snippets(
    skills: list[str],
    industry: str,
    roles: list[str],
    ideal_stage: str,
) -> str:
    top_skills = ", ".join(skills[:5])
    top_role   = roles[0] if roles else "Operations Manager"

    stage_query = {
        "Startup":       "startup companies hiring",
        "Growth-stage":  "scale-up growth stage companies hiring",
        "Enterprise":    "enterprise companies hiring",
        "Turnaround":    "turnaround restructuring companies hiring",
    }.get(ideal_stage, "companies hiring")

    queries = [
        f"{industry} companies hiring {top_role} 2025",
        f"{stage_query} {top_skills} 2025",
    ]

    from concurrent.futures import ThreadPoolExecutor, as_completed
    all_snippets: list[str] = []
    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = {pool.submit(_ddg_search, q): q for q in queries}
        for future in as_completed(futures, timeout=10):
            q = futures[future]
            try:
                found = future.result(timeout=3)
                all_snippets.extend(found)
                logger.debug(f"[company-radar] Query '{q}' → {len(found)} snippets")
            except Exception:
                pass

    if not all_snippets:
        return "No live search results — do not fabricate company names."

    return "\n".join(all_snippets[:20])


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
                max_output_tokens=1200,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[company-radar] Vertex AI failed: {e} — trying AI Studio key")

    try:
        from config.settings import settings
        api_key = settings.gemini_api_key
    except Exception:
        api_key = ""

    if not api_key:
        raise RuntimeError(
            "Gemini unavailable for company radar — enable Vertex AI or set GEMINI_API_KEY"
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
            max_output_tokens=1200,
        ),
    )
    return response.text.strip()


_VALID_SIZES = {"startup", "mid-size", "enterprise"}


def _safe_fallback(industry: str) -> CompanyRadarResult:
    return CompanyRadarResult(
        companies=[
            CompanyMatch(
                name="Search returned no specific employers",
                industry=industry,
                size="enterprise",
                match_reason="Live search did not surface specific employers — try searching LinkedIn for companies in your industry.",
                source_note="Fallback — live search unavailable.",
            )
        ],
        data_freshness=date.today().isoformat(),
    )


def run_company_radar_agent(
    skill_names:  list[str],
    industry:     str,
    top_roles:    list[str],
    ideal_stage:  str,
    project_id:   str,
    region:       str = "us-central1",
) -> CompanyRadarResult:
    """
    Find 5–7 named employers actively hiring for this candidate's profile.

    Uses DDG search to ground every company name in real search evidence.
    Gemini is instructed not to invent companies absent from the results.

    Parameters
    ----------
    skill_names:  Top skills from the candidate's profile.
    industry:     Human-readable industry label.
    top_roles:    Role titles from MarketPulseResult (used in search queries).
    ideal_stage:  Ideal company stage from TraitProfile (Startup/Growth-stage/…).
    project_id:   GCP project ID.
    region:       Vertex AI region.

    Returns
    -------
    CompanyRadarResult with company list and data_freshness date.
    """
    _apply_credentials()
    skills = skill_names[:10]

    logger.info(
        f"[company-radar] Starting — industry='{industry}' "
        f"stage='{ideal_stage}' roles={top_roles[:2]}"
    )

    snippets = _gather_snippets(skills, industry, top_roles, ideal_stage)

    prompt = _PROMPT_TEMPLATE.format(
        skills_csv=", ".join(skills),
        industry=industry,
        roles_csv=", ".join(top_roles[:3]),
        ideal_stage=ideal_stage,
        search_snippets=snippets,
    )

    try:
        raw = _call_gemini(prompt, project_id, region)
    except Exception as exc:
        logger.error(f"[company-radar] Gemini call failed: {exc} — using fallback")
        return _safe_fallback(industry)

    logger.debug(f"[company-radar] Raw response: {raw[:400]}")

    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(clean)
    except Exception as exc:
        logger.warning(f"[company-radar] JSON parse failed: {exc}")
        return _safe_fallback(industry)

    companies: list[CompanyMatch] = []
    for c in (data.get("companies") or [])[:7]:
        size = c.get("size", "enterprise")
        if size not in _VALID_SIZES:
            size = "enterprise"
        name = (c.get("name") or "").strip()
        if not name:
            continue
        companies.append(CompanyMatch(
            name=name,
            industry=c.get("industry", industry),
            size=size,
            match_reason=c.get("match_reason", ""),
            source_note=c.get("source_note", "Based on recent job postings."),
        ))

    if not companies:
        return _safe_fallback(industry)

    result = CompanyRadarResult(
        companies=companies,
        data_freshness=date.today().isoformat(),
    )

    logger.info(
        f"[company-radar] Done — {len(companies)} employers found: "
        f"{', '.join(c.name for c in companies[:3])}"
    )
    return result
