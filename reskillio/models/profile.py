"""Domain models for candidate skill profiles."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field

from reskillio.models.skill import SkillCategory


class ProfiledSkill(BaseModel):
    """A skill entry in a candidate's profile, aggregated across extractions."""

    skill_name: str
    category: SkillCategory
    first_seen: datetime
    last_seen: datetime
    frequency: int = Field(description="Number of extractions containing this skill")
    confidence_avg: float = Field(description="Average confidence across all extractions")
    source_count: int = Field(description="Number of distinct extraction runs")

    @property
    def days_active(self) -> int:
        """Days between first and last appearance."""
        return (self.last_seen - self.first_seen).days


class CandidateProfile(BaseModel):
    """Aggregated skill profile for a single candidate."""

    candidate_id: str
    skills: list[ProfiledSkill] = Field(default_factory=list)
    total_skills: int = 0
    last_updated: datetime | None = None

    def model_post_init(self, __context: object) -> None:
        self.total_skills = len(self.skills)
        if self.skills:
            self.last_updated = max(s.last_seen for s in self.skills)

    def by_category(self) -> dict[str, list[ProfiledSkill]]:
        """Group skills by category, sorted by frequency desc."""
        groups: dict[str, list[ProfiledSkill]] = {}
        for skill in sorted(self.skills, key=lambda s: s.frequency, reverse=True):
            groups.setdefault(skill.category.value, []).append(skill)
        return groups

    def top_skills(self, n: int = 10) -> list[ProfiledSkill]:
        """Return top-N skills ranked by frequency then confidence."""
        return sorted(
            self.skills,
            key=lambda s: (s.frequency, s.confidence_avg),
            reverse=True,
        )[:n]