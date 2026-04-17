"""
F16 — POST /analyze response models.

AnalysisResult is the top-level response returned by the full-career-rebound
orchestration endpoint.  It contains the outputs of all five pipeline stages
plus per-stage timing and success metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from reskillio.models.gap import GapAnalysisResult
from reskillio.models.industry import IndustryMatchResult
from reskillio.models.narrative import NarrativeGrounding
from reskillio.models.pathway import PathwayRoadmap


# ---------------------------------------------------------------------------
# Per-stage metadata
# ---------------------------------------------------------------------------

class StageResult(BaseModel):
    success:     bool
    duration_ms: int
    error:       Optional[str] = None


# ---------------------------------------------------------------------------
# Skill summary (lightweight — full ExtractionResult is not included)
# ---------------------------------------------------------------------------

class SkillSummary(BaseModel):
    name:       str
    category:   str
    confidence: float


# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    candidate_id: str
    target_role:  str
    analyzed_at:  datetime

    # Stage 1 — Skill extraction
    skill_count: int
    top_skills:  list[SkillSummary]

    # Stage 2 — Gap analysis vs target JD / role
    gap: Optional[GapAnalysisResult] = None

    # Stage 3 — Industry match scores
    industry_match: Optional[IndustryMatchResult] = None

    # Stage 4 — Gemini career narrative
    narrative:            Optional[str]               = None
    narrative_grounding:  Optional[NarrativeGrounding] = None

    # Stage 5 — 90-day reskilling pathway (slow — opt-in)
    pathway: Optional[PathwayRoadmap] = None

    # Pipeline metadata
    stages:           dict[str, StageResult]
    total_duration_ms: int
