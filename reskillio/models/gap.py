"""Domain models for gap analysis results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransferableSkill(BaseModel):
    """A candidate skill that semantically covers a JD requirement."""

    jd_skill: str = Field(description="Skill required by the JD")
    candidate_skill: str = Field(description="Closest matching skill the candidate has")
    similarity: float = Field(description="Cosine similarity score (0–1)")


class GapAnalysisResult(BaseModel):
    """Full output of a gap analysis run."""

    candidate_id: str
    jd_id: str | None = None
    jd_title: str | None = None
    jd_company: str | None = None
    industry: str | None = None
    seniority: str | None = None

    gap_score: float = Field(
        description="0–100. Higher = stronger candidate-JD fit.",
        ge=0.0,
        le=100.0,
    )

    total_required: int
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Required skills the candidate has (exact match)",
    )
    transferable_skills: list[TransferableSkill] = Field(
        default_factory=list,
        description="Required skills covered by semantically similar candidate skills",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Required skills not found in candidate profile",
    )
    similarity_threshold: float = Field(
        default=0.75,
        description="Cosine similarity threshold used for transferable classification",
    )
    recommendation: str = Field(description="Brief human-readable assessment")