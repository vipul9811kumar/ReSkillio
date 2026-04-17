"""
F14 — Medallion lakehouse Pydantic models.

Bronze: raw ingestion
Silver: validated / deduplicated
Gold:   computed business outputs
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Gold — Match Scores
# ---------------------------------------------------------------------------

class MatchScore(BaseModel):
    match_id:            str
    candidate_id:        str
    jd_id:               str
    required_coverage:   float = Field(ge=0.0, le=1.0)
    preferred_coverage:  float = Field(ge=0.0, le=1.0)
    overall_match_score: float = Field(ge=0.0, le=100.0)
    matched_skill_count: int
    missing_skill_count: int
    extra_skill_count:   int
    computed_at:         datetime


# ---------------------------------------------------------------------------
# Gold — Industry Rankings
# ---------------------------------------------------------------------------

class IndustryRanking(BaseModel):
    ranking_id:      str
    candidate_id:    str
    industry:        str
    skills_matched:  int
    skills_demanded: int
    skill_coverage:  float = Field(ge=0.0, le=1.0)
    readiness_tier:  str   # TIER_1 / TIER_2 / TIER_3
    computed_at:     datetime


# ---------------------------------------------------------------------------
# Gold — Candidate Readiness
# ---------------------------------------------------------------------------

class CandidateReadiness(BaseModel):
    candidate_id:                str
    skill_count:                 int
    high_confidence_skill_count: int
    avg_confidence:              float
    best_industry:               Optional[str]
    best_industry_coverage:      Optional[float]
    avg_match_score:             Optional[float]
    readiness_index:             float = Field(ge=0.0, le=100.0)
    readiness_tier:              str   # READY / DEVELOPING / EMERGING
    computed_at:                 datetime


# ---------------------------------------------------------------------------
# Lakehouse status
# ---------------------------------------------------------------------------

class LayerTableInfo(BaseModel):
    table:     str
    row_count: int


class LakehouseStatus(BaseModel):
    project_id: str
    bronze:     list[LayerTableInfo]
    silver:     list[LayerTableInfo]
    gold:       list[LayerTableInfo]


# ---------------------------------------------------------------------------
# Request / Response wrappers
# ---------------------------------------------------------------------------

class PromoteRequest(BaseModel):
    candidate_id: str


class PromoteResponse(BaseModel):
    candidate_id:  str
    silver_rows:   int
    message:       str


class GoldRefreshRequest(BaseModel):
    candidate_id: str
    jd_ids:       list[str] = Field(default_factory=list)


class GoldRefreshResponse(BaseModel):
    candidate_id:       str
    match_scores:       list[MatchScore]
    industry_rankings:  list[IndustryRanking]
    readiness:          Optional[CandidateReadiness]
    message:            str
