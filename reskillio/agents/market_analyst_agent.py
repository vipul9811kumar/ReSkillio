"""
F10 — MarketAnalystAgent (CrewAI).

A single-agent crew that:
  1. Receives a list of skills.
  2. Calls JobDemandSearchTool for each skill (real DuckDuckGo web search).
  3. Synthesises demand_score (0–100) and trend direction per skill.
  4. Returns a structured MarketAnalysisResult.

LLM: Gemini 2.5 Flash via Vertex AI (LiteLLM bridge).
Search: DuckDuckGo — no API key required.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional, Type

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import BaseTool
from ddgs import DDGS
from loguru import logger
from pydantic import BaseModel, Field

from reskillio.models.market import MarketAnalysisResult, SkillDemand

_AGENT_MODEL    = "vertex_ai/gemini-2.5-flash"
_SEARCH_RESULTS = 4          # DDG results per skill
_MAX_SKILLS     = 10         # cap to keep latency reasonable


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class _SkillSearchInput(BaseModel):
    skill: str = Field(..., description="The skill name to search job demand for")


class JobDemandSearchTool(BaseTool):
    """
    Search DuckDuckGo for real-time job market demand data for a single skill.
    Returns concatenated snippet text from the top search results.
    """
    name:        str = "job_demand_search"
    description: str = (
        "Search the web for current job market demand, hiring trends, and salary "
        "data for a given technical skill. Input: skill name (string)."
    )
    args_schema: Type[BaseModel] = _SkillSearchInput

    def _run(self, skill: str) -> str:
        queries = [
            f"{skill} developer jobs demand 2025",
            f"{skill} skill hiring trend tech market",
        ]
        snippets: list[str] = []
        try:
            with DDGS() as ddgs:
                for q in queries:
                    results = ddgs.text(q, max_results=_SEARCH_RESULTS // 2)
                    for r in (results or []):
                        title = r.get("title", "")
                        body  = r.get("body", "")[:200]
                        snippets.append(f"[{title}] {body}")
        except Exception as exc:
            logger.warning(f"DDG search failed for '{skill}': {exc}")
            return f"Search unavailable for {skill}. Use general knowledge."

        if not snippets:
            return f"No search results found for '{skill}'."

        return f"Search results for '{skill}':\n" + "\n".join(snippets)


# ---------------------------------------------------------------------------
# LLM setup (Vertex AI via LiteLLM)
# ---------------------------------------------------------------------------

def _setup_vertex_env(project_id: str, region: str) -> None:
    """Ensure LiteLLM can find Vertex AI credentials and CrewAI telemetry is off."""
    os.environ.setdefault("VERTEXAI_PROJECT",          project_id)
    os.environ.setdefault("VERTEXAI_LOCATION",         region)
    os.environ.setdefault("CREWAI_TRACING_ENABLED",    "false")
    os.environ.setdefault("OTEL_SDK_DISABLED",         "true")
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            from config.settings import settings
            if settings.google_application_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    settings.google_application_credentials
                )
        except Exception:
            pass


def _build_llm(project_id: str, region: str) -> LLM:
    _setup_vertex_env(project_id, region)
    return LLM(
        model=_AGENT_MODEL,
        temperature=0.1,
        max_tokens=2048,
    )


# ---------------------------------------------------------------------------
# Crew builder
# ---------------------------------------------------------------------------

_SCORING_RUBRIC = """
demand_score rubric (0–100):
  90–100 : Extremely high demand — Python, Kubernetes, SQL, cloud platforms
  70–89  : High demand — most ML frameworks, modern web stacks, DevOps tools
  50–69  : Moderate demand — established but not top-priority in postings
  30–49  : Lower demand — specialised or niche tooling
  0–29   : Declining or very niche — legacy tech or falling out of favour

trend:
  growing   — job postings or employer interest visibly increasing year-on-year
  stable    — demand steady, not growing or shrinking significantly
  declining — fewer postings or replaced by newer alternatives
"""

_TASK_DESCRIPTION = """
You are analysing the job market for a displaced professional who wants to re-enter
the workforce. For each skill listed below, use the job_demand_search tool to
gather current market data, then assign a demand_score and trend.

Skills to analyse: {skills_csv}

Steps:
1. Call job_demand_search for each skill individually.
2. Based on the search evidence, score each skill using this rubric:
{rubric}
3. Return a JSON object with this EXACT structure and nothing else:

{{
  "skills_analyzed": [
    {{
      "skill": "<skill name>",
      "demand_score": <float 0-100>,
      "trend": "<growing|stable|declining>",
      "evidence": "<one sentence citing search evidence>"
    }}
  ],
  "analysis_note": "<optional brief overall note>"
}}

Important: Return ONLY the JSON object. No markdown fences, no explanation.
"""


def _build_crew(llm: LLM) -> tuple[Crew, Agent, Task]:
    analyst = Agent(
        role="Senior Job Market Analyst",
        goal=(
            "Provide accurate, evidence-based job market demand scores and trend "
            "directions for technical skills using real-time web search data."
        ),
        backstory=(
            "You are a seasoned labour market economist specialising in the tech "
            "sector. You combine web search data with your deep knowledge of hiring "
            "trends to give displaced workers an honest assessment of their skills' "
            "market value. You always cite evidence and never fabricate data."
        ),
        tools=[JobDemandSearchTool()],
        llm=llm,
        verbose=False,
        max_iter=20,
    )

    task = Task(
        description=_TASK_DESCRIPTION,
        expected_output=(
            "A JSON object with skills_analyzed list, each entry containing: "
            "skill, demand_score (0-100), trend (growing/stable/declining), evidence."
        ),
        agent=analyst,
    )

    crew = Crew(
        agents=[analyst],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    return crew, analyst, task


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _parse_output(raw: str, skills: list[str]) -> list[SkillDemand]:
    """
    Extract JSON from the agent output and build SkillDemand objects.
    Falls back to neutral scores if parsing fails.
    """
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data  = json.loads(clean)
        items = data.get("skills_analyzed", [])
        result = []
        for item in items:
            trend = item.get("trend", "stable")
            if trend not in ("growing", "stable", "declining"):
                trend = "stable"
            result.append(SkillDemand(
                skill=item.get("skill", ""),
                demand_score=float(item.get("demand_score", 50.0)),
                trend=trend,
                evidence=item.get("evidence", ""),
            ))
        if result:
            return result
    except Exception as exc:
        logger.warning(f"Output parse failed: {exc} — raw: {raw[:200]}")

    # Fallback: return neutral scores for all requested skills
    return [
        SkillDemand(skill=s, demand_score=50.0, trend="stable", evidence="Data unavailable.")
        for s in skills
    ]


def _parse_analysis_note(raw: str) -> Optional[str]:
    try:
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data  = json.loads(clean)
        return data.get("analysis_note")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_market_analyst_agent(
    skills: list[str],
    project_id: str,
    region: str = "us-central1",
) -> MarketAnalysisResult:
    """
    Run the MarketAnalystAgent for the given skill list.

    Parameters
    ----------
    skills:       List of skill names to analyse (capped at MAX_SKILLS).
    project_id:   GCP project ID (used for Vertex AI LLM).
    region:       Vertex AI region.

    Returns
    -------
    MarketAnalysisResult with demand_score and trend per skill.
    """
    if not skills:
        raise ValueError("skills list cannot be empty")

    skills = skills[:_MAX_SKILLS]
    skills_csv = ", ".join(skills)
    logger.info(f"[market-agent] Analysing {len(skills)} skills: {skills_csv}")

    llm  = _build_llm(project_id, region)
    crew, _, task = _build_crew(llm)

    # Inject inputs into task description
    task.description = _TASK_DESCRIPTION.format(
        skills_csv=skills_csv,
        rubric=_SCORING_RUBRIC,
    )

    result = crew.kickoff()
    raw_output = str(result)

    logger.debug(f"[market-agent] Raw output:\n{raw_output[:500]}")

    skill_demands  = _parse_output(raw_output, skills)
    analysis_note  = _parse_analysis_note(raw_output)

    logger.info(
        f"[market-agent] Done — {len(skill_demands)} skills scored  "
        f"top={max(skill_demands, key=lambda x: x.demand_score).skill if skill_demands else 'n/a'}"
    )

    return MarketAnalysisResult(
        skills_analyzed=skill_demands,
        analyzed_at=datetime.now(timezone.utc),
        data_sources=["DuckDuckGo web search (real-time)"],
        agent_model=_AGENT_MODEL,
        analysis_note=analysis_note,
    )