"""
Intake conversation engine.

Drives a 5-question, conversational onboarding that surfaces:
  Q1 — Financial runway (urgency)
  Q2 — Geographic flexibility
  Q3 — Work identity, team pref, company stage
  Q4 — What energised them in their career
  Q5 — What they want next (+ fractional openness)

Each user turn triggers two Gemini calls:
  1. Chat call (temp 0.7) — warm, coach-like reply that moves conversation forward
  2. Extract call (temp 0.1) — structured JSON extraction of key fields
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

from loguru import logger

from reskillio.models.intake import (
    IntakeMessage,
    IntakeProfile,
    IntakeSession,
)

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_sessions:    dict[str, IntakeSession] = {}
_session_ts:  dict[str, float]         = {}   # session_id → creation monotonic time
_SESSION_TTL  = 7200.0                         # 2 hours — evict stale sessions


def _cleanup_stale_sessions() -> None:
    """Evict sessions older than _SESSION_TTL. Called on each new session start."""
    cutoff = time.monotonic() - _SESSION_TTL
    stale = [sid for sid, ts in _session_ts.items() if ts < cutoff]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_ts.pop(sid, None)
    if stale:
        logger.info(f"[intake] Evicted {len(stale)} stale session(s); {len(_sessions)} active")


# ── Question guides ─────────────────────────────────────────────────────────

_QUESTIONS = {
    1: {
        "theme": "Financial runway",
        "opener": (
            "Before we look at your skills and options, I want to understand your situation. "
            "How much runway do you have right now — how long before financial pressure becomes critical for you?"
        ),
        "suggestions": [
            "Less than 3 months — I need to move fast",
            "3–6 months — some breathing room",
            "6–12 months — comfortable pace",
            "12+ months — no immediate pressure",
        ],
        "extract_prompt": (
            "From the user's message, extract:\n"
            "- financial_runway: one of [immediate, short, moderate, comfortable]\n"
            "  (immediate=<3mo, short=3-6mo, moderate=6-12mo, comfortable=12+mo)\n"
            "- urgency_score: float 0.0–1.0 (1=must find work immediately, 0=no pressure)\n"
            "Respond with only valid JSON: {\"financial_runway\": \"...\", \"urgency_score\": 0.0}"
        ),
    },
    2: {
        "theme": "Geographic flexibility",
        "opener": (
            "Got it. Now — where are you open to working? "
            "Are you tied to a specific city, open to relocating, or happy working fully remote?"
        ),
        "suggestions": [
            "Local only — I can't relocate",
            "Hybrid local — prefer local but flexible",
            "Open to relocation for the right role",
            "Fully remote — location doesn't matter",
            "Global — I work anywhere",
        ],
        "extract_prompt": (
            "From the user's message, extract:\n"
            "- geographic_flexibility: one of [local_only, hybrid_local, open_to_relocation, fully_remote, global]\n"
            "- target_locations: list of specific cities or regions mentioned (empty list if none)\n"
            "Respond with only valid JSON: {\"geographic_flexibility\": \"...\", \"target_locations\": []}"
        ),
    },
    3: {
        "theme": "Work identity",
        "opener": (
            "Let's talk about how you work best. "
            "When you've been at your best professionally — what role were you playing? "
            "Were you building something new, running complex systems, fixing broken situations, "
            "connecting people, advising decisions, or driving innovation?"
        ),
        "suggestions": [
            "Builder — I create things from scratch",
            "Operator — I run and scale systems",
            "Fixer — I turn around broken situations",
            "Advisor — I guide key decisions",
            "Connector — I build communities and partnerships",
            "Innovator — I challenge the status quo",
        ],
        "extract_prompt": (
            "From the user's message, extract:\n"
            "- work_identity: one of [Builder, Operator, Fixer, Advisor, Connector, Innovator]\n"
            "- team_preference: one of [solo, small_team, large_team, cross_functional]\n"
            "- company_stage_preference: list from [Startup, Growth-stage, Enterprise, Turnaround]\n"
            "Respond with only valid JSON: "
            "{\"work_identity\": \"...\", \"team_preference\": \"...\", \"company_stage_preference\": []}"
        ),
    },
    4: {
        "theme": "What energised you",
        "opener": (
            "Think about the work that lit you up — not the job title, but the actual day-to-day moments "
            "that made you feel most alive and effective. What was happening in those moments?"
        ),
        "suggestions": [
            "Solving hard technical problems",
            "Leading and developing people",
            "Building something that didn't exist before",
            "Making a visible impact on customers",
            "Navigating complexity and bringing order",
            "Learning and mastering new domains",
        ],
        "extract_prompt": (
            "From the user's message, extract:\n"
            "- loved_aspects: a 1–2 sentence summary of what energised them (first person → third person)\n"
            "- key_themes: list of 2–4 theme keywords\n"
            "Respond with only valid JSON: {\"loved_aspects\": \"...\", \"key_themes\": []}"
        ),
    },
    5: {
        "theme": "What next",
        "opener": (
            "Last question — if you could design your next chapter, what does it look like? "
            "What kind of problem do you want to work on, and what does 'success in 12 months' mean to you? "
            "Also — are you open to fractional, consulting, or contract work, or are you looking for a full-time role?"
        ),
        "suggestions": [
            "Full-time role — I want stability and belonging",
            "Open to contract or fractional alongside a search",
            "Consulting is where I want to be",
            "I'm genuinely open — tell me the best path",
        ],
        "extract_prompt": (
            "From the user's message, extract:\n"
            "- want_next: a 1–2 sentence summary of what they want next\n"
            "- open_to_fractional: boolean\n"
            "- engagement_format: one of [full_time_only, open_to_contract, open_to_fractional, consulting_preferred]\n"
            "Respond with only valid JSON: "
            "{\"want_next\": \"...\", \"open_to_fractional\": false, \"engagement_format\": \"...\"}"
        ),
    },
}

_CHAT_SYSTEM = (
    "You are a warm, sharp career coach helping a recently displaced professional understand "
    "what they want next — not just what they've done. Your role is to make this feel like a "
    "real conversation, not a form. Ask one follow-up if the answer is vague, then move forward. "
    "Keep replies to 2–4 sentences. Never be generic. Reference what they just said."
)


# ── Gemini helpers ──────────────────────────────────────────────────────────

def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


def _gemini_client(project_id: str, region: str):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
    from google import genai
    return genai.Client(vertexai=True, project=project_id, location=region)


def _call_gemini_chat(messages: list[dict], project_id: str, region: str) -> str:
    from google import genai
    from google.genai import types as gt

    # Build conversation string from messages
    conversation = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)

    try:
        client = _gemini_client(project_id, region)
        response = client.models.generate_content(
            model=_MODEL_VERTEX,
            contents=conversation,
            config=gt.GenerateContentConfig(
                system_instruction=_CHAT_SYSTEM,
                temperature=0.75,
                max_output_tokens=300,
                thinking_config=gt.ThinkingConfig(thinking_budget=0),
            ),
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"[intake] Vertex chat failed: {e}")
        return ""


def _call_gemini_extract(user_message: str, extract_prompt: str, project_id: str, region: str) -> dict:
    from google import genai
    from google.genai import types as gt

    prompt = f"{extract_prompt}\n\nUser said: {user_message}"

    try:
        client = _gemini_client(project_id, region)
        response = client.models.generate_content(
            model=_MODEL_VERTEX,
            contents=prompt,
            config=gt.GenerateContentConfig(
                temperature=0.05,
                max_output_tokens=300,
                thinking_config=gt.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        logger.warning(f"[intake] Extract failed: {e}")
        return {}


# ── Session management ──────────────────────────────────────────────────────

def _build_profile_from_session(session: IntakeSession) -> IntakeProfile:
    ex = session.extracted
    return IntakeProfile(
        candidate_id=session.candidate_id,
        financial_runway=ex.get("financial_runway"),
        urgency_score=float(ex.get("urgency_score", 0.5)),
        geographic_flexibility=ex.get("geographic_flexibility"),
        target_locations=ex.get("target_locations", []),
        work_identity=ex.get("work_identity"),
        team_preference=ex.get("team_preference"),
        company_stage_preference=ex.get("company_stage_preference", []),
        engagement_format=ex.get("engagement_format"),
        open_to_fractional=bool(ex.get("open_to_fractional", False)),
        loved_aspects=ex.get("loved_aspects", ""),
        want_next=ex.get("want_next", ""),
    )


class IntakeConversationEngine:
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        _apply_credentials()

    def start_session(self, candidate_id: str) -> tuple[str, str, list[str]]:
        """Create a new session and return (session_id, opening_message, suggestions)."""
        _cleanup_stale_sessions()
        session_id = str(uuid.uuid4())
        q = _QUESTIONS[1]
        session = IntakeSession(
            session_id=session_id,
            candidate_id=candidate_id,
            current_question=1,
        )
        opener = q["opener"]
        session.messages.append(IntakeMessage(role="assistant", content=opener))
        _sessions[session_id] = session
        _session_ts[session_id] = time.monotonic()
        logger.info(f"[intake] New session {session_id} for candidate {candidate_id} ({len(_sessions)} active)")
        return session_id, opener, q["suggestions"]

    def process_turn(
        self, session_id: str, user_message: str
    ) -> tuple[str, int, list[str], bool, IntakeProfile | None]:
        """
        Process one user turn.

        Returns (reply, question_n, suggestions, completed, profile_or_None).
        profile is only set when all 5 questions are done.
        """
        session = _sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        if session.completed:
            return "We've already completed your intake.", 5, [], True, _build_profile_from_session(session)

        # Record user message
        session.messages.append(IntakeMessage(role="user", content=user_message))

        q_num = session.current_question
        q = _QUESTIONS[q_num]

        # 1) Extract structured data from user message
        extracted = _call_gemini_extract(
            user_message, q["extract_prompt"], self.project_id, self.region
        )
        session.extracted.update(extracted)
        logger.info(f"[intake] Q{q_num} extracted: {extracted}")

        # 2) Generate warm chat reply
        history = [{"role": m.role, "content": m.content} for m in session.messages]

        # If more questions remain, prime the reply to transition smoothly
        next_q_num = q_num + 1
        if next_q_num <= 5:
            next_opener = _QUESTIONS[next_q_num]["opener"]
            transition_hint = (
                f"\n\nAfter acknowledging their answer warmly in 1–2 sentences, "
                f"transition naturally to: \"{next_opener}\""
            )
            history_with_hint = history[:-1] + [
                {"role": "user", "content": user_message + transition_hint}
            ]
        else:
            history_with_hint = history

        reply = _call_gemini_chat(history_with_hint, self.project_id, self.region)

        # Fallback if Gemini unavailable
        if not reply:
            if next_q_num <= 5:
                reply = f"Got it — that really helps. {_QUESTIONS[next_q_num]['opener']}"
            else:
                reply = "That's everything I needed — thank you. Your profile is ready."

        session.messages.append(IntakeMessage(role="assistant", content=reply))

        # Advance question or complete
        if next_q_num <= 5:
            session.current_question = next_q_num
            suggestions = _QUESTIONS[next_q_num]["suggestions"]
            completed = False
            profile = None
        else:
            session.completed = True
            profile = _build_profile_from_session(session)
            suggestions = []
            completed = True
            logger.info(f"[intake] Session {session_id} completed")
            # Free session from RAM immediately — profile is persisted to BQ
            _sessions.pop(session_id, None)
            _session_ts.pop(session_id, None)

        return reply, session.current_question, suggestions, completed, profile

    def get_session(self, session_id: str) -> IntakeSession | None:
        return _sessions.get(session_id)
