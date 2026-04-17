"""Domain models for the PathwayPlannerAgent output."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CourseResource(BaseModel):
    skill:          str
    title:          str
    platform:       str   = Field(..., description="Coursera, Udemy, YouTube, or Other")
    url:            str
    level:          str   = Field(..., description="beginner | intermediate | advanced")
    duration_hours: Optional[int] = None
    description:    str


class RoadmapPhase(BaseModel):
    phase:        int
    title:        str                  # "Foundation", "Core Development", "Advanced & Portfolio"
    weeks:        str                  # "1–4", "5–8", "9–13"
    focus_skills: list[str]
    resources:    list[CourseResource]
    weekly_hours: int                  # recommended study hours per week
    milestone:    str                  # measurable outcome at phase end


class PathwayRoadmap(BaseModel):
    candidate_id:    str
    target_role:     str
    gap_score:       float = Field(..., ge=0.0, le=100.0)
    total_days:      int   = 90
    phases:          list[RoadmapPhase]
    success_metrics: list[str]
    overall_summary: str
    generated_at:    datetime
    agent_model:     str