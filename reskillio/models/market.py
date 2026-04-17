"""Domain models for the MarketAnalystAgent output."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SkillDemand(BaseModel):
    skill:        str
    demand_score: float = Field(..., ge=0.0, le=100.0, description="Job market demand 0–100")
    trend:        Literal["growing", "stable", "declining"]
    evidence:     str   = Field(..., description="One-sentence evidence from search results")


class MarketAnalysisResult(BaseModel):
    skills_analyzed: list[SkillDemand]
    analyzed_at:     datetime
    data_sources:    list[str]
    agent_model:     str
    analysis_note:   Optional[str] = None