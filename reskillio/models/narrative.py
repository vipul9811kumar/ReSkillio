"""Domain models for the career narrative endpoint."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from reskillio.models.jd import Industry


class NarrativeRequest(BaseModel):
    candidate_id: str = Field(..., description="Candidate to narrate")
    target_role: str = Field(..., min_length=2, description="Target role title, e.g. 'Senior Data Engineer'")
    industry: Optional[Industry] = Field(
        default=None,
        description="Target industry — auto-detected from match scores if omitted",
    )


class NarrativeGrounding(BaseModel):
    """The exact BQ-retrieved facts passed to Gemini. Shows what the model was given."""
    candidate_top_skills: list[str]
    industry_top_skills: list[str]
    skill_overlap: list[str]
    overlap_count: int
    total_industry_skills: int
    sample_jd_titles: list[str]


class NarrativeResult(BaseModel):
    candidate_id: str
    target_role: str
    industry: str
    industry_label: str
    narrative: str = Field(..., description="3-sentence career rebound narrative")
    grounding: NarrativeGrounding
    model: str
    generated_at: datetime