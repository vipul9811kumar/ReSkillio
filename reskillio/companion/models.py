"""Pydantic models for the weekly check-in and digest system."""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


# ── Weekly check-in ─────────────────────────────────────────────────────────

class WeeklyCheckin(BaseModel):
    """One row per candidate per week in BigQuery: reskillio.weekly_checkins."""
    candidate_id:         str
    week_number:          int
    week_start:           date
    checked_in_at:        datetime

    # Self-reported
    hours_on_courses:     float      = 0.0
    courses_completed:    list[str]  = Field(default_factory=list)
    applications_sent:    int        = 0
    interviews_scheduled: int        = 0
    interviews_completed: int        = 0
    offers_received:      int        = 0

    # System-computed
    gap_score:            Optional[float] = None
    gap_score_delta:      Optional[float] = None
    top_industry_score:   Optional[float] = None
    top_industry_delta:   Optional[float] = None
    top_industry:         Optional[str]   = None

    digest_generated:     bool            = False
    digest_id:            Optional[str]   = None


class ApplicationLog(BaseModel):
    candidate_id:  str
    company_name:  str
    role_title:    str
    applied_date:  date
    status:        str            = "applied"  # applied|interview|offer|rejected|withdrawn
    gap_score:     Optional[float] = None
    notes:         Optional[str]   = None


# ── Action item (defined before WeeklyDigest so forward ref resolves) ────────

class ActionItem(BaseModel):
    title:       str
    description: str
    priority:    str   # high | medium | low
    time_est:    str
    completed:   bool = False


# ── Digest section ───────────────────────────────────────────────────────────

class DigestSection(BaseModel):
    section_type: str   # gap_progress | course_progress | applications | market | narrative
    headline:     str
    body:         str
    data:         dict


# ── Weekly digest ────────────────────────────────────────────────────────────

class WeeklyDigest(BaseModel):
    """One digest per candidate per week: reskillio.companion_digests."""
    digest_id:          str
    candidate_id:       str
    week_number:        int
    week_start:         date
    generated_at:       datetime

    gap_score:          float
    gap_score_delta:    float
    course_completion:  float
    applications_sent:  int
    interviews_active:  int

    opening_message:    str
    market_signal:      str
    gemini_narrative:   str
    top_action:         str

    sections:           list[DigestSection] = Field(default_factory=list)
    action_items:       list[ActionItem]    = Field(default_factory=list)


# ── API shapes ───────────────────────────────────────────────────────────────

class CheckinSubmitRequest(BaseModel):
    candidate_id:         str
    hours_on_courses:     float      = 0.0
    courses_completed:    list[str]  = Field(default_factory=list)
    applications_sent:    int        = 0
    interviews_scheduled: int        = 0
    interviews_completed: int        = 0
    notes:                Optional[str]              = None
    application_logs:     list[ApplicationLog]       = Field(default_factory=list)


class CheckinSubmitResponse(BaseModel):
    week_number:      int
    gap_score:        float
    gap_score_delta:  float
    digest_id:        str
    digest_ready:     bool
    next_checkin:     date


class DigestResponse(BaseModel):
    digest:           WeeklyDigest
    previous_digests: list[WeeklyDigest] = Field(default_factory=list)
