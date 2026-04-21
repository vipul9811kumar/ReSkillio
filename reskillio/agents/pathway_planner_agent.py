"""
F11 — PathwayPlannerAgent.

Replaces the old CrewAI two-agent crew with a faster direct implementation:
  1. DDG course searches run in parallel threads (one per skill)
  2. A single Gemini call synthesises all research into a 90-day roadmap

Previous CrewAI approach: 60–120 s (sequential DDG + multi-turn agent loop)
New approach: 12–20 s (parallel DDG + single Gemini synthesis)
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from loguru import logger

from reskillio.models.pathway import CourseResource, PathwayRoadmap, RoadmapPhase

_MAX_GAP_SKILLS    = 6   # fewer skills → fewer searches → faster
_COURSES_PER_SKILL = 2
_DDG_TIMEOUT       = 8   # seconds per skill search
_GEMINI_MODEL      = "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Step 1 — parallel DDG course search
# ---------------------------------------------------------------------------

def _search_courses_for_skill(skill: str) -> dict:
    """Search Coursera + Udemy for one skill. Runs in a thread."""
    courses = []
    queries = [
        f"site:coursera.org {skill} course",
        f"site:udemy.com {skill} course",
    ]
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for q in queries:
                try:
                    hits = ddgs.text(q, max_results=2) or []
                    for h in hits:
                        url = h.get("href", h.get("url", ""))
                        platform = (
                            "Coursera" if "coursera.org" in url else
                            "Udemy"    if "udemy.com"    in url else
                            "Other"
                        )
                        courses.append({
                            "title":       h.get("title", ""),
                            "platform":    platform,
                            "url":         url,
                            "description": h.get("body", "")[:150],
                        })
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(f"[pathway] DDG search failed for '{skill}': {exc}")

    return {"skill": skill, "courses": courses[:_COURSES_PER_SKILL]}


def _fetch_all_courses(skills: list[str]) -> list[dict]:
    """Run course searches for all skills in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=min(len(skills), 6)) as pool:
        futures = {pool.submit(_search_courses_for_skill, s): s for s in skills}
        for future in as_completed(futures, timeout=_DDG_TIMEOUT * 2):
            try:
                results.append(future.result(timeout=3))
            except Exception as exc:
                skill = futures[future]
                logger.warning(f"[pathway] Course search timed out for '{skill}': {exc}")
                results.append({"skill": skill, "courses": []})
    return results


# ---------------------------------------------------------------------------
# Step 2 — single Gemini synthesis call
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a career development coach building a 90-day reskilling roadmap.

TARGET ROLE: {target_role}
CURRENT FIT SCORE: {gap_score}/100
MISSING SKILLS: {missing_skills}
TRANSFERABLE SKILLS (worth deepening): {transferable_skills}

COURSE RESEARCH (real courses found online):
{course_research}

Build a 3-phase 90-day roadmap. Return ONLY this JSON (no markdown fences):
{{
  "phases": [
    {{
      "phase": 1,
      "title": "Foundation",
      "weeks": "1-4",
      "focus_skills": ["skill1", "skill2"],
      "resources": [
        {{
          "skill": "skill name",
          "title": "course title",
          "platform": "Coursera|Udemy|Other",
          "url": "https://...",
          "level": "beginner|intermediate|advanced",
          "duration_hours": 15,
          "description": "one sentence"
        }}
      ],
      "weekly_hours": 10,
      "milestone": "concrete measurable outcome by end of week 4"
    }},
    {{"phase": 2, "title": "Core Development", "weeks": "5-8", "focus_skills": [], "resources": [], "weekly_hours": 10, "milestone": "..."}},
    {{"phase": 3, "title": "Advanced & Portfolio", "weeks": "9-13", "focus_skills": [], "resources": [], "weekly_hours": 8, "milestone": "..."}}
  ],
  "success_metrics": ["metric1", "metric2", "metric3"],
  "overall_summary": "2-3 sentence overview"
}}

Rules:
- Use only courses from the research above (real URLs). If none found for a skill, omit resources for it.
- Phase 1: highest-priority missing skills. Phase 2: remaining gaps. Phase 3: transferable deepening + portfolio.
- 8-12 study hours/week — realistic for job searching alongside learning.
- Milestones must be specific and verifiable.
"""


def _call_gemini_synthesis(
    target_role: str,
    gap_score: float,
    missing_skills: list[str],
    transferable_skills: list[str],
    course_research: list[dict],
    project_id: str,
    region: str,
) -> str:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

    research_text = ""
    for item in course_research:
        skill = item["skill"]
        courses = item.get("courses", [])
        if courses:
            lines = [f"  • [{c['platform']}] {c['title']} — {c['url']}" for c in courses]
            research_text += f"\n{skill}:\n" + "\n".join(lines)
        else:
            research_text += f"\n{skill}: (no courses found — suggest self-study resources)"

    prompt = _SYNTHESIS_PROMPT.format(
        target_role=target_role,
        gap_score=f"{gap_score:.0f}",
        missing_skills=", ".join(missing_skills) or "none identified",
        transferable_skills=", ".join(transferable_skills) or "none",
        course_research=research_text or "No courses found — use general recommendations.",
    )

    try:
        from google import genai
        from google.genai import types as gt

        _apply_credentials()
        client = genai.Client(vertexai=True, project=project_id, location=region)
        response = client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
            config=gt.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=3000,
                thinking_config=gt.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as exc:
        logger.error(f"[pathway] Gemini synthesis failed: {exc}")
        return ""


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _strip_fences(text: str) -> str:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    return text.strip()


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
            agent_model=_GEMINI_MODEL,
        )
    except Exception as exc:
        logger.warning(f"[pathway] Parse failed: {exc} — raw[:200]: {raw[:200]}")
        return PathwayRoadmap(
            candidate_id=candidate_id,
            target_role=target_role,
            gap_score=gap_score,
            total_days=90,
            phases=[],
            success_metrics=["Complete all identified skill gaps"],
            overall_summary="Roadmap generation encountered a parsing error.",
            generated_at=datetime.now(timezone.utc),
            agent_model=_GEMINI_MODEL,
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

    Replaces the old CrewAI two-agent crew.
    Step 1: parallel DDG course searches (~6-8s)
    Step 2: single Gemini synthesis call (~6-10s)
    Total: ~12-18s vs previous 60-120s
    """
    if not missing_skills and not transferable_skills:
        raise ValueError("No skill gaps provided — nothing to plan.")

    all_skills = list(dict.fromkeys(missing_skills + transferable_skills))
    priority_skills = sorted(
        all_skills[:_MAX_GAP_SKILLS],
        key=lambda s: market_scores.get(s, 50.0),
        reverse=True,
    )

    logger.info(
        f"[pathway] Planning roadmap: candidate='{candidate_id}' "
        f"role='{target_role}' skills={priority_skills} gap={gap_score:.0f}"
    )

    # Step 1 — parallel course searches
    course_research = _fetch_all_courses(priority_skills)
    logger.info(f"[pathway] Course research done — {sum(len(r['courses']) for r in course_research)} courses found")

    # Step 2 — single Gemini synthesis
    raw = _call_gemini_synthesis(
        target_role=target_role,
        gap_score=gap_score,
        missing_skills=missing_skills[:_MAX_GAP_SKILLS],
        transferable_skills=transferable_skills[:4],
        course_research=course_research,
        project_id=project_id,
        region=region,
    )

    roadmap = _parse_roadmap(raw, candidate_id, target_role, gap_score)
    logger.info(
        f"[pathway] Done — {len(roadmap.phases)} phases  "
        f"resources={sum(len(p.resources) for p in roadmap.phases)}"
    )
    return roadmap
