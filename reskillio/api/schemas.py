"""Request and response schemas for the ReSkillio API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from reskillio.models.skill import Skill


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Resume or job-description text")
    candidate_id: str = Field(..., min_length=1, description="Unique identifier for the candidate")


class ExtractResponse(BaseModel):
    candidate_id: str
    extraction_id: str
    skill_count: int
    skills: list[Skill]
    model_used: str
    stored: bool = Field(
        default=False,
        description="True if skills were persisted to BigQuery",
    )
