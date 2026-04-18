"""Person Gap — personalised gap analysis driven by intake context."""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AlignedSkill(BaseModel):
    skill:     str
    relevance: str  # why this skill matters for their stated goal


class GrowthSkill(BaseModel):
    skill:        str
    priority:     Literal["high", "medium", "low"]
    why_needed:   str
    how_to_build: str  # specific resource or approach


class PersonGapResult(BaseModel):
    candidate_id:        str
    narrative:           str                  # 2–3 sentence personalised context
    aligned_skills:      list[AlignedSkill]   # skills they have that fit their goal
    growth_skills:       list[GrowthSkill]    # skills to develop
    surprise_transfers:  list[str]            # transferable strengths they may not realise
    readiness_score:     float                # 0.0–1.0, how ready for stated goal
    recommended_actions: list[str]            # 3–5 concrete next steps
    context_used:        str = ""             # "intake" | "trait_only" — what drove the analysis
