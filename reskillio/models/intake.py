"""Intake profile — 5-question conversational onboarding."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

FinancialRunway     = Literal["immediate", "short", "moderate", "comfortable"]
GeographicFlex      = Literal["local_only", "hybrid_local", "open_to_relocation", "fully_remote", "global"]
WorkIdentity        = Literal["Builder", "Operator", "Fixer", "Advisor", "Connector", "Innovator"]
TeamPreference      = Literal["solo", "small_team", "large_team", "cross_functional"]
CompanyStagePrefs   = Literal["Startup", "Growth-stage", "Enterprise", "Turnaround"]
EngagementFormat    = Literal["full_time_only", "open_to_contract", "open_to_fractional", "consulting_preferred"]


class IntakeProfile(BaseModel):
    candidate_id:               str
    financial_runway:           Optional[FinancialRunway]   = None
    urgency_score:              float                       = 0.5  # 0=no pressure, 1=urgent
    geographic_flexibility:     Optional[GeographicFlex]    = None
    target_locations:           list[str]                   = Field(default_factory=list)
    work_identity:              Optional[WorkIdentity]       = None
    team_preference:            Optional[TeamPreference]     = None
    company_stage_preference:   list[CompanyStagePrefs]     = Field(default_factory=list)
    engagement_format:          Optional[EngagementFormat]  = None
    open_to_fractional:         bool                        = False
    loved_aspects:              str                         = ""
    want_next:                  str                         = ""
    created_at:                 datetime                    = Field(default_factory=datetime.utcnow)
    completed_at:               Optional[datetime]          = None


class IntakeMessage(BaseModel):
    role:      Literal["assistant", "user"]
    content:   str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IntakeSession(BaseModel):
    session_id:       str
    candidate_id:     str
    current_question: int             = 1  # 1–5
    messages:         list[IntakeMessage] = Field(default_factory=list)
    extracted:        dict            = Field(default_factory=dict)  # accumulated structured data
    completed:        bool            = False


# ── API shapes ──────────────────────────────────────────────────────────────

class IntakeStartRequest(BaseModel):
    candidate_id: str

class IntakeStartResponse(BaseModel):
    session_id:  str
    question_n:  int
    message:     str
    suggestions: list[str] = Field(default_factory=list)

class IntakeTurnRequest(BaseModel):
    session_id: str
    message:    str

class IntakeTurnResponse(BaseModel):
    reply:       str
    question_n:  int           # current question number after this turn
    suggestions: list[str]     = Field(default_factory=list)
    completed:   bool          = False
    profile:     Optional[IntakeProfile] = None
