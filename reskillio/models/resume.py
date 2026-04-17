"""Domain models for resume ingestion and section-level extraction."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from reskillio.models.skill import Skill


class SectionType(str, Enum):
    SUMMARY = "summary"
    EXPERIENCE = "experience"
    SKILLS = "skills"
    EDUCATION = "education"
    CERTIFICATIONS = "certifications"
    PROJECTS = "projects"
    OTHER = "other"


class ResumeSection(BaseModel):
    """A single parsed section of a resume."""

    section_type: SectionType
    heading: str = Field(description="Original heading text as found in the document")
    raw_text: str
    skills: list[Skill] = Field(default_factory=list)

    @property
    def skill_count(self) -> int:
        return len(self.skills)


class ResumeExtractionResult(BaseModel):
    """Full output of a resume ingestion run."""

    candidate_id: str
    filename: str
    sections: list[ResumeSection] = Field(default_factory=list)
    all_skills: list[Skill] = Field(
        default_factory=list,
        description="Deduplicated skills across all sections, highest confidence wins",
    )
    total_skill_count: int = 0
    model_used: str = ""
    stored: bool = False

    def model_post_init(self, __context: object) -> None:
        self.total_skill_count = len(self.all_skills)

    def skills_by_section(self) -> dict[str, list[str]]:
        """Return {section_type: [skill_name, ...]} for easy inspection."""
        return {
            s.section_type.value: [sk.name for sk in s.skills]
            for s in self.sections
        }