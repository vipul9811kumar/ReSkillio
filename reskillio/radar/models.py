"""
reskillio/radar/models.py
Pydantic data models for the Opportunity Radar.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EngagementType(str, Enum):
    FRACTIONAL = "fractional"
    CONSULTING = "consulting"
    ADVISORY   = "advisory"
    INTERIM    = "interim"


class HiringSignal(str, Enum):
    ACTIVELY_HIRING = "actively_hiring"
    RECENTLY_POSTED = "recently_posted"
    INFERRED        = "inferred"
    NETWORK         = "network"


class CompanyStage(str, Enum):
    SEED       = "seed"
    SERIES_A   = "series_a"
    SERIES_B   = "series_b"
    SERIES_C   = "series_c"
    PE_BACKED  = "pe_backed"
    SMB        = "smb"
    ENTERPRISE = "enterprise"


class Opportunity(BaseModel):
    opportunity_id: str

    company_name:     str
    company_stage:    CompanyStage
    company_industry: str
    company_size_est: Optional[str] = None
    company_location: Optional[str] = None
    company_url:      Optional[str] = None

    role_title:       str
    engagement_type:  EngagementType
    commitment_days_per_week:    Optional[float] = None
    commitment_hours_per_month:  Optional[float] = None
    duration_months:  Optional[int] = None

    rate_floor:       Optional[float] = None
    rate_ceiling:     Optional[float] = None
    rate_unit:        str = "day"
    equity_offered:   bool = False
    equity_pct:       Optional[float] = None

    required_skills:  list[str] = Field(default_factory=list)
    nice_to_have:     list[str] = Field(default_factory=list)

    culture_signals:  list[str] = Field(default_factory=list)
    ideal_identity:   Optional[str] = None
    ideal_stage_exp:  Optional[str] = None

    hiring_signal:    HiringSignal = HiringSignal.INFERRED
    source_url:       Optional[str] = None
    source_text:      Optional[str] = None
    discovered_at:    Optional[datetime] = None
    expires_at:       Optional[datetime] = None

    remote_ok:         bool = True
    location_required: Optional[str] = None


class MatchScoreBreakdown(BaseModel):
    skill_overlap_score: float
    trait_fit_score:     float
    context_score:       float
    overall_score:       float


class SkillMatchDetail(BaseModel):
    matched_skills:      list[str]
    missing_skills:      list[str]
    transferable_skills: list[dict]
    overlap_pct:         float


class OpportunityMatch(BaseModel):
    match_id:      str
    candidate_id:  str
    opportunity_id: str
    opportunity:   Opportunity

    score_breakdown: MatchScoreBreakdown
    skill_detail:    SkillMatchDetail

    match_reasons:   list[str]
    fit_narrative:   str
    missing_context: Optional[str] = None

    pitch_generated: bool = False
    pitch_text:      Optional[str] = None
    engagement_tips: list[dict] = Field(default_factory=list)
    intro_angle:     Optional[str] = None

    saved:       bool = False
    status:      str = "new"
    generated_at: datetime


class RadarRequest(BaseModel):
    candidate_id: str
    max_results:  int = 12
    types:        list[EngagementType] = Field(default_factory=list)
    refresh:      bool = False

class RadarResponse(BaseModel):
    candidate_id:   str
    matches:        list[OpportunityMatch]
    total_found:    int
    avg_daily_rate: Optional[float] = None
    generated_at:   datetime
    persona_used:   Optional[str] = None

class PitchRequest(BaseModel):
    candidate_id: str
    match_id:     str

class PitchResponse(BaseModel):
    match_id:        str
    pitch_text:      str
    engagement_tips: list[dict]
    intro_angle:     Optional[str]

class SaveMatchRequest(BaseModel):
    candidate_id: str
    match_id:     str
    status:       str = "saved"
