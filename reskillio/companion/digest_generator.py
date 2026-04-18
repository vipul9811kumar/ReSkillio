"""
Gemini-powered weekly digest generator.

Two Gemini calls per digest:
  1. Narrative (temp 0.6) — 3-sentence personalised week reflection
  2. Actions  (temp 0.2) — 4 concrete, JSON-formatted action items

Uses google.genai (same SDK as all other ReSkillio pipelines).
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.companion.models import (
    ActionItem, DigestSection, WeeklyCheckin, WeeklyDigest,
)

_MODEL = "gemini-2.5-flash"

_SYSTEM = (
    "You are the ReSkillio weekly companion — warm, honest, and direct. "
    "You write personalised weekly digests for people navigating career transitions. "
    "Be specific to this person's actual data — never generic. "
    "Every section ends with what to do, not just what happened. "
    "Acknowledge difficulty when appropriate. Displacement is hard."
)

_NARRATIVE_PROMPT = """\
Candidate: {name}
Week: {week_number} of their career transition
Gap score this week: {gap_score} ({gap_delta:+.1f} vs last week)
Target role: {target_role}
Hours on courses: {hours}
Applications sent: {apps}
Interview detail: {interview_detail}
Top skill being closed: {top_skill} ({skill_pct}% complete)
Market signal: {market_signal}
Intake persona: {persona}

Write exactly 3 sentences:
1. A specific, honest reflection on what this week's data shows about their progress
2. The most important thing they should know about their situation right now
3. What they should do first thing tomorrow morning

Be concrete. Use their actual numbers. Sound like a trusted advisor, not a chatbot.
Return ONLY the 3-sentence narrative. No labels, no formatting.
"""

_ACTIONS_PROMPT = """\
Candidate context:
- Gap score: {gap_score}/100 for {target_role}
- Skills still open: {open_skills}
- Applications out: {apps_out}
- Interview this week: {has_interview}
- Financial runway: {runway}
- Hours available per week: ~{hours_per_week}

Generate exactly 4 action items for this week.
Return as JSON array:
[
  {{
    "title": "Short, specific action (under 10 words)",
    "description": "Why this matters + how long it takes",
    "priority": "high|medium|low",
    "time_est": "X hrs or X min"
  }}
]

Rules:
- Action 1: most impactful skill gap to close
- Action 2: interview prep if interview this week, else application
- Action 3: application action
- Action 4: quick win under 30 min (LinkedIn, resume, networking)
- Be specific: "Complete Python module 2 — pandas basics" not "Study Python"
- Return ONLY valid JSON. No markdown.
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


def _call_gemini(prompt: str, project_id: str, region: str,
                 temperature: float = 0.6, max_tokens: int = 600) -> str:
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

    from google import genai
    from google.genai import types as gt

    client = genai.Client(vertexai=True, project=project_id, location=region)
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=gt.GenerateContentConfig(
            system_instruction=_SYSTEM,
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=gt.ThinkingConfig(thinking_budget=0),
        ),
    )
    return response.text.strip()


class DigestGenerator:
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region     = region
        _apply_credentials()

    def generate_digest(
        self,
        checkin:          WeeklyCheckin,
        candidate_name:   str,
        target_role:      str,
        intake_profile:   dict,
        gap_history:      list[dict],    # [{week_number, gap_score}]
        market_data:      dict,          # from MarketPulseAgent
        active_courses:   list[dict],    # [{name, pct_complete, platform}]
        application_log:  list[dict],    # [{company, role, status, applied_date}]
    ) -> WeeklyDigest:
        from reskillio.companion.checkin_store import CompanionStore
        store    = CompanionStore(project_id=self.project_id)
        prev     = store.get_previous_checkin(checkin.candidate_id, checkin.week_number)
        gap_delta = (checkin.gap_score or 0) - (prev.get("gap_score") or 0) if prev else 0.0

        course_completion = self._avg_course_completion(active_courses)
        market_signal     = self._market_signal(market_data, target_role)

        interview_detail = "yes" if checkin.interviews_scheduled > 0 else "none this week"
        top_skill  = active_courses[0]["name"] if active_courses else "unknown"
        skill_pct  = int(active_courses[0].get("pct_complete", 0)) if active_courses else 0

        narrative = self._narrative(
            name=candidate_name, week_number=checkin.week_number,
            gap_score=checkin.gap_score or 0, gap_delta=gap_delta,
            target_role=target_role, hours=checkin.hours_on_courses,
            apps=checkin.applications_sent, interview_detail=interview_detail,
            top_skill=top_skill, skill_pct=skill_pct,
            market_signal=market_signal,
            persona=intake_profile.get("persona_label", "experienced professional"),
        )

        open_skills = [c["name"] for c in active_courses if c.get("pct_complete", 0) < 100]
        apps_out    = [a for a in application_log if a.get("status") in ("applied", "interview")]
        action_items = self._actions(
            gap_score=checkin.gap_score or 0, target_role=target_role,
            open_skills=open_skills, apps_out=len(apps_out),
            has_interview=checkin.interviews_scheduled > 0,
            runway=intake_profile.get("financial_runway", "moderate"),
            hours_per_week=8,
        )

        opening    = self._opening(candidate_name, checkin.week_number,
                                   checkin.gap_score or 0, gap_delta)
        top_action = action_items[0].title if action_items else "Complete your course module this week"
        digest_id  = str(uuid.uuid4())

        sections = [
            DigestSection(
                section_type="gap_progress",
                headline=f"Gap score: {checkin.gap_score:.0f}/100 ({gap_delta:+.0f} this week)",
                body=f"You need 90 to be a strong candidate. You are {90 - (checkin.gap_score or 0):.0f} points away.",
                data={"current": checkin.gap_score, "delta": gap_delta,
                      "target": 90, "history": gap_history},
            ),
            DigestSection(
                section_type="course_progress",
                headline=f"Courses: {course_completion:.0f}% overall completion",
                body=f"Active courses: {len(active_courses)}",
                data={"courses": active_courses},
            ),
            DigestSection(
                section_type="applications",
                headline=f"{checkin.applications_sent} applications this week · {len(apps_out)} active",
                body="",
                data={"log": application_log},
            ),
            DigestSection(
                section_type="market",
                headline="Market intelligence — this week",
                body=market_signal,
                data=market_data,
            ),
        ]

        digest = WeeklyDigest(
            digest_id=digest_id, candidate_id=checkin.candidate_id,
            week_number=checkin.week_number, week_start=checkin.week_start,
            generated_at=datetime.now(timezone.utc),
            gap_score=checkin.gap_score or 0, gap_score_delta=gap_delta,
            course_completion=course_completion,
            applications_sent=checkin.applications_sent,
            interviews_active=checkin.interviews_scheduled + checkin.interviews_completed,
            opening_message=opening, market_signal=market_signal,
            gemini_narrative=narrative, top_action=top_action,
            sections=sections, action_items=action_items,
        )

        store.save_digest(digest)
        logger.info(f"[digest] Generated {digest_id} for {checkin.candidate_id} week {checkin.week_number}")
        return digest

    # ── Private helpers ──────────────────────────────────────────────────────

    def _opening(self, name: str, week: int, gap_score: float, gap_delta: float) -> str:
        direction = "improved" if gap_delta > 0 else "stayed steady" if gap_delta == 0 else "dipped"
        try:
            return _call_gemini(
                f"Write a 2-sentence opening for {name}'s week {week} digest. "
                f"Their gap score {direction} to {gap_score:.0f}/100 ({gap_delta:+.0f} points). "
                f"Be specific and warm. Reference the actual number. Under 40 words total.",
                self.project_id, self.region, temperature=0.65, max_tokens=100,
            )
        except Exception as e:
            logger.error(f"[digest] Opening failed: {e}")
            return f"Good morning, {name}. Your gap score moved to {gap_score:.0f}/100 this week."

    def _narrative(self, **kwargs) -> str:
        try:
            return _call_gemini(
                _NARRATIVE_PROMPT.format(**kwargs),
                self.project_id, self.region, temperature=0.6, max_tokens=300,
            )
        except Exception as e:
            logger.error(f"[digest] Narrative failed: {e}")
            return "Your progress this week is meaningful. Keep the momentum going."

    def _actions(self, gap_score: float, target_role: str, open_skills: list,
                 apps_out: int, has_interview: bool, runway: str,
                 hours_per_week: int) -> list[ActionItem]:
        prompt = _ACTIONS_PROMPT.format(
            gap_score=gap_score, target_role=target_role,
            open_skills=", ".join(open_skills[:3]) or "none remaining",
            apps_out=apps_out,
            has_interview="yes" if has_interview else "no",
            runway=runway, hours_per_week=hours_per_week,
        )
        try:
            text = _call_gemini(prompt, self.project_id, self.region,
                                temperature=0.2, max_tokens=400)
            text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return [ActionItem(**item) for item in json.loads(text)[:4]]
        except Exception as e:
            logger.error(f"[digest] Actions failed: {e}")
            return [
                ActionItem(title="Complete your next course module",
                           description="Keep the gap score moving", priority="high", time_est="2 hrs"),
                ActionItem(title="Apply to 2 new roles this week",
                           description="Market momentum is positive", priority="high", time_est="1 hr"),
            ]

    def _market_signal(self, market_data: dict, target_role: str) -> str:
        top_industry = market_data.get("top_industry", "your target industry")
        volume       = market_data.get("open_roles_change_pct", 0)
        if volume > 5:
            return (f"Demand for {target_role} roles in {top_industry} rose {volume:.0f}% "
                    f"this week — a good time to apply.")
        elif volume < -5:
            return (f"Hiring in {top_industry} slowed slightly ({volume:.0f}%). "
                    f"Focus on quality applications over volume.")
        return f"The {top_industry} market is holding steady — your target industry is stable."

    def _avg_course_completion(self, courses: list[dict]) -> float:
        if not courses:
            return 0.0
        return round(sum(c.get("pct_complete", 0) for c in courses) / len(courses), 1)


_generator: Optional[DigestGenerator] = None

def get_generator(project_id: str, region: str = "us-central1") -> DigestGenerator:
    global _generator
    if _generator is None:
        _generator = DigestGenerator(project_id=project_id, region=region)
    return _generator
