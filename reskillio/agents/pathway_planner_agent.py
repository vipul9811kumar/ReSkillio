"""
F11 — PathwayPlannerAgent (CrewAI).

Two-agent sequential crew:
  1. Researcher  — calls CourseSearchTool for each priority skill gap, returns
                   a structured list of real course links.
  2. Planner     — takes the research + gap/market context, synthesises a
                   structured 90-day reskilling roadmap.

Tools
-----
CourseSearchTool — DuckDuckGo with Coursera + Udemy site filters.
                   No API keys required.

Input
-----
- missing_skills : list[str]          — from gap analysis
- transferable   : list[str]          — partially-covered skills worth deepening
- market_scores  : dict[str, float]   — demand_score per skill from F10
- gap_score      : float              — overall JD fit score 0–100
- target_role    : str
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

from reskillio.models.pathway import CourseResource, PathwayRoadmap, RoadmapPhase

_AGENT_MODEL    = "vertex_ai/gemini-2.5-flash"
_MAX_GAP_SKILLS = 8      # cap skills searched to control latency
_COURSES_PER_SKILL = 2   # target courses per skill


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class _CourseInput(BaseModel):
    skill: str = Field(..., description="Skill name to search courses for")


class CourseSearchTool(BaseTool):
    """
    Search Coursera and Udemy for courses on a given skill.
    Uses DuckDuckGo — no API key required.
    Returns titles, URLs, platforms, and descriptions.
    """
    name:        str = "course_search"
    description: str = (
        "Find online courses for a specific skill from Coursera and Udemy. "
        "Input: skill name. Returns course titles, URLs, platforms, descriptions."
    )
    args_schema: Type[BaseModel] = _CourseInput

    def _run(self, skill: str) -> str:
        results: list[str] = []
        queries = [
            f"{skill} course coursera.org",
            f"{skill} course udemy.com",
        ]
        try:
            with DDGS() as ddgs:
                for q in queries:
                    hits = ddgs.text(q, max_results=3) or []
                    for h in hits:
                        title = h.get("title", "")
                        url   = h.get("href", h.get("url", ""))
                        body  = h.get("body", "")[:180]
                        platform = (
                            "Coursera" if "coursera.org" in url else
                            "Udemy"    if "udemy.com"    in url else
                            "Other"
                        )
                        results.append(
                            f"PLATFORM: {platform}\n"
                            f"TITLE: {title}\n"
                            f"URL: {url}\n"
                            f"DESC: {body}"
                        )
        except Exception as exc:
            logger.warning(f"CourseSearchTool failed for '{skill}': {exc}")
            return f"Search unavailable for {skill}."

        if not results:
            return f"No courses found for '{skill}'."

        return f"=== Courses for '{skill}' ===\n\n" + "\n\n".join(results)


# ---------------------------------------------------------------------------
# Env + LLM
# ---------------------------------------------------------------------------

def _setup_env(project_id: str, region: str) -> None:
    os.environ.setdefault("VERTEXAI_PROJECT",       project_id)
    os.environ.setdefault("VERTEXAI_LOCATION",      region)
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED",      "true")
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
    _setup_env(project_id, region)
    return LLM(model=_AGENT_MODEL, temperature=0.2, max_tokens=4096)


# ---------------------------------------------------------------------------
# Crew
# ---------------------------------------------------------------------------

_RESEARCHER_TASK = """
You are researching online courses for a professional re-entering the {target_role} field.

PRIORITY SKILLS TO LEARN (ordered by urgency — search all of them):
{skills_with_scores}

For EACH skill above, call the course_search tool exactly once.
Compile all findings into a structured list. For each skill return:
- skill name
- 1-2 best courses found (title, platform, URL, brief description, estimated level)

Return a JSON array like this (no markdown fences):
[
  {{
    "skill": "...",
    "courses": [
      {{"title": "...", "platform": "Coursera|Udemy|Other", "url": "...", "level": "beginner|intermediate|advanced", "description": "..."}}
    ]
  }}
]
"""

_PLANNER_TASK = """
You are designing a 90-day reskilling roadmap for a professional targeting a
{target_role} role. Their current JD fit score is {gap_score:.0f}/100.

COURSE RESEARCH (already gathered — use these resources):
{course_research}

GAP CONTEXT:
- Missing skills (required but not in profile): {missing_skills}
- Transferable skills (partially covered, worth deepening): {transferable_skills}
- Market demand scores: {market_scores}

ROADMAP RULES:
1. Three phases — allocate skills by market demand and urgency:
   Phase 1 (Weeks 1–4)  "Foundation"             — 2-3 highest-demand missing skills
   Phase 2 (Weeks 5–8)  "Core Development"        — remaining missing skills
   Phase 3 (Weeks 9–13) "Advanced & Portfolio"    — transferable deepening + project work
2. Recommend 8–12 study hours per week (realistic for working adults).
3. Each phase must have a concrete, measurable milestone.
4. Assign courses from the research above to the appropriate phase.
5. Success metrics: 3–5 verifiable outcomes the candidate can achieve.

Return ONLY this JSON (no markdown fences, no explanation):
{{
  "phases": [
    {{
      "phase": 1,
      "title": "Foundation",
      "weeks": "1-4",
      "focus_skills": ["...", "..."],
      "resources": [
        {{
          "skill": "...",
          "title": "course title",
          "platform": "Coursera",
          "url": "https://...",
          "level": "beginner",
          "duration_hours": 20,
          "description": "one sentence"
        }}
      ],
      "weekly_hours": 10,
      "milestone": "concrete measurable outcome"
    }},
    {{ "phase": 2, ... }},
    {{ "phase": 3, ... }}
  ],
  "success_metrics": ["...", "...", "..."],
  "overall_summary": "2-3 sentence overview of the roadmap"
}}
"""


def _build_crew(llm: LLM, target_role: str) -> tuple[Crew, Task, Task]:
    search_tool = CourseSearchTool()

    researcher = Agent(
        role="Learning Resources Researcher",
        goal=(
            "Find the most relevant and highly-rated online courses from Coursera "
            "and Udemy for each skill gap a displaced professional needs to fill."
        ),
        backstory=(
            "You are an expert in online learning platforms who knows exactly which "
            "courses give displaced workers the fastest, most practical path to job-readiness. "
            "You always search before recommending — no hallucinated course titles or URLs."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=False,
        max_iter=25,
    )

    planner = Agent(
        role="Career Pathway Architect",
        goal=(
            "Transform course research and gap analysis into a structured, "
            "actionable 90-day reskilling roadmap with clear milestones."
        ),
        backstory=(
            "You are a career development coach specialising in tech workforce re-entry. "
            "You create realistic, week-by-week plans that displaced workers can follow "
            "alongside part-time job searching. You never invent courses — you only use "
            "resources found by the researcher."
        ),
        tools=[],
        llm=llm,
        verbose=False,
    )

    research_task = Task(
        description=_RESEARCHER_TASK,
        expected_output=(
            "A JSON array where each entry has 'skill' and 'courses' (list with "
            "title, platform, url, level, description)."
        ),
        agent=researcher,
    )

    plan_task = Task(
        description=_PLANNER_TASK,
        expected_output=(
            "A JSON object with phases (list), success_metrics (list), "
            "and overall_summary (string)."
        ),
        agent=planner,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, planner],
        tasks=[research_task, plan_task],
        process=Process.sequential,
        verbose=False,
    )

    return crew, research_task, plan_task


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    return re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()


def _parse_roadmap(
    raw: str,
    candidate_id: str,
    target_role: str,
    gap_score: float,
) -> PathwayRoadmap:
    try:
        data = json.loads(_strip_fences(raw))
        phases = []
        for p in data.get("phases", []):
            resources = [
                CourseResource(
                    skill=r.get("skill", ""),
                    title=r.get("title", ""),
                    platform=r.get("platform", "Other"),
                    url=r.get("url", ""),
                    level=r.get("level", "intermediate"),
                    duration_hours=r.get("duration_hours"),
                    description=r.get("description", ""),
                )
                for r in p.get("resources", [])
            ]
            phases.append(RoadmapPhase(
                phase=p["phase"],
                title=p.get("title", f"Phase {p['phase']}"),
                weeks=p.get("weeks", ""),
                focus_skills=p.get("focus_skills", []),
                resources=resources,
                weekly_hours=p.get("weekly_hours", 10),
                milestone=p.get("milestone", ""),
            ))

        return PathwayRoadmap(
            candidate_id=candidate_id,
            target_role=target_role,
            gap_score=gap_score,
            total_days=90,
            phases=phases,
            success_metrics=data.get("success_metrics", []),
            overall_summary=data.get("overall_summary", ""),
            generated_at=datetime.now(timezone.utc),
            agent_model=_AGENT_MODEL,
        )
    except Exception as exc:
        logger.warning(f"Roadmap parse failed: {exc} — raw[:300]: {raw[:300]}")
        return PathwayRoadmap(
            candidate_id=candidate_id,
            target_role=target_role,
            gap_score=gap_score,
            total_days=90,
            phases=[],
            success_metrics=["Complete all identified skill gaps"],
            overall_summary="Roadmap generation encountered a parsing error.",
            generated_at=datetime.now(timezone.utc),
            agent_model=_AGENT_MODEL,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pathway_planner_agent(
    candidate_id: str,
    target_role: str,
    missing_skills: list[str],
    transferable_skills: list[str],
    gap_score: float,
    market_scores: dict[str, float],
    project_id: str,
    region: str = "us-central1",
) -> PathwayRoadmap:
    """
    Generate a 90-day reskilling roadmap.

    Parameters
    ----------
    candidate_id:       Candidate identifier.
    target_role:        Job title being targeted.
    missing_skills:     Skills required by JD but absent from profile.
    transferable_skills:Skills partially covered — worth deepening.
    gap_score:          Current JD fit 0–100.
    market_scores:      demand_score per skill from MarketAnalystAgent.
    project_id:         GCP project for Vertex AI LLM.
    region:             Vertex AI region.
    """
    if not missing_skills and not transferable_skills:
        raise ValueError("No skill gaps provided — nothing to plan.")

    # Combine and prioritise: missing skills first, then transferable
    # Sort by market demand score descending
    all_gap_skills = list(dict.fromkeys(missing_skills + transferable_skills))
    priority_skills = sorted(
        all_gap_skills[:_MAX_GAP_SKILLS],
        key=lambda s: market_scores.get(s, 50.0),
        reverse=True,
    )

    skills_with_scores = "\n".join(
        f"  {i+1}. {s} (market demand: {market_scores.get(s, 'unknown')})"
        for i, s in enumerate(priority_skills)
    )

    market_scores_str = ", ".join(
        f"{s}: {market_scores.get(s, 'N/A')}" for s in priority_skills
    )

    logger.info(
        f"[pathway-agent] Planning roadmap: candidate='{candidate_id}' "
        f"role='{target_role}' skills={len(priority_skills)} gap={gap_score:.0f}"
    )

    llm  = _build_llm(project_id, region)
    crew, research_task, plan_task = _build_crew(llm, target_role)

    # Inject context into task descriptions
    research_task.description = _RESEARCHER_TASK.format(
        target_role=target_role,
        skills_with_scores=skills_with_scores,
    )
    plan_task.description = _PLANNER_TASK.format(
        target_role=target_role,
        gap_score=gap_score,
        course_research="{course_research_placeholder}",   # filled by context
        missing_skills=", ".join(missing_skills) or "none",
        transferable_skills=", ".join(transferable_skills) or "none",
        market_scores=market_scores_str,
    )

    result = crew.kickoff()
    raw    = str(result)
    logger.debug(f"[pathway-agent] Raw output:\n{raw[:600]}")

    roadmap = _parse_roadmap(raw, candidate_id, target_role, gap_score)

    logger.info(
        f"[pathway-agent] Done — {len(roadmap.phases)} phases  "
        f"resources={sum(len(p.resources) for p in roadmap.phases)}"
    )
    return roadmap