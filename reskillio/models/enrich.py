"""
Enrichment models — output of the /enrich endpoint (Option B slow path).

Each sub-model maps to one future step:
  MarketPulseResult  — Step 3  (MarketPulseAgent, DDG live job search)
  AutoGapResult      — Step 4  (JD-less gap analysis vs top matched roles)
  SalaryIntelResult  — Step 5  (skill-attributed salary band)
  CompanyRadarResult — Step 6  (named employers actively hiring)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel

from reskillio.models.analyze import StageResult


# ---------------------------------------------------------------------------
# Step 3 — Market Pulse
# ---------------------------------------------------------------------------

class RoleSignal(BaseModel):
    title:              str
    hiring_activity:    Literal["high", "medium", "low"]
    top_skills_needed:  list[str]
    evidence:           str       # one sentence from search results


class MarketPulseResult(BaseModel):
    top_roles:       list[RoleSignal]   # top 5 roles actively hiring for this profile
    overall_demand:  Literal["high", "medium", "low"]
    market_summary:  str
    data_sources:    list[str]
    analyzed_at:     datetime


# ---------------------------------------------------------------------------
# Step 4 — Auto-gap (JD-less — uses top roles as synthetic JDs)
# ---------------------------------------------------------------------------

class RoleGap(BaseModel):
    role_title:      str
    gap_score:       float          # 0–100 readiness
    matched_skills:  list[str]
    missing_skills:  list[str]


class AutoGapResult(BaseModel):
    role_gaps:           list[RoleGap]  # one entry per top role from MarketPulse
    overall_readiness:   float          # mean gap_score across roles
    top_skills_to_add:   list[str]      # missing skills ranked by how often they appear


# ---------------------------------------------------------------------------
# Step 5 — Salary Intelligence
# ---------------------------------------------------------------------------

class SalaryDriver(BaseModel):
    label:     str              # skill, trait, or experience signal
    direction: Literal["up", "down"]
    delta_usd: int              # rough $ impact on median
    reason:    str              # one-sentence explanation


class SalaryIntelResult(BaseModel):
    floor_usd:    int
    median_usd:   int
    ceiling_usd:  int
    currency:     str = "USD"
    drivers:      list[SalaryDriver]
    note:         str           # e.g. "US national, mid-career, Operations & Analytics"


# ---------------------------------------------------------------------------
# Step 6 — Company Radar
# ---------------------------------------------------------------------------

class CompanyMatch(BaseModel):
    name:         str
    industry:     str
    size:         Literal["startup", "mid-size", "enterprise"]
    match_reason: str           # one sentence: why this company, given the profile
    source_note:  str           # "Based on job postings indexed this week"


class CompanyRadarResult(BaseModel):
    companies:       list[CompanyMatch]
    data_freshness:  str        # ISO date string — when sources were searched


# ---------------------------------------------------------------------------
# Top-level enrichment result
# ---------------------------------------------------------------------------

class EnrichmentResult(BaseModel):
    candidate_id:  str
    target_role:   str
    enriched_at:   datetime

    # Step 3
    market_pulse:  Optional[MarketPulseResult]  = None
    # Step 4
    auto_gap:      Optional[AutoGapResult]      = None
    # Step 5
    salary_intel:  Optional[SalaryIntelResult]  = None
    # Step 6
    company_radar: Optional[CompanyRadarResult] = None

    stages:            dict[str, StageResult]
    total_duration_ms: int
