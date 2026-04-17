"""Domain models for skills and extraction results."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SkillCategory(str, Enum):
    TECHNICAL = "technical"
    SOFT = "soft"
    DOMAIN = "domain"
    TOOL = "tool"
    CERTIFICATION = "certification"
    UNKNOWN = "unknown"


class Skill(BaseModel):
    """A single extracted skill."""

    name: str = Field(..., description="Normalised skill name, e.g. 'Python'")
    category: SkillCategory = SkillCategory.UNKNOWN
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source_text: Optional[str] = Field(default=None, description="Raw text span that produced this skill")

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Skill):
            return self.name.lower() == other.name.lower()
        return NotImplemented


class ExtractionResult(BaseModel):
    """Output of a single skill-extraction run."""

    input_text: str
    skills: list[Skill] = Field(default_factory=list)
    skill_count: int = 0
    model_used: str = ""

    def model_post_init(self, __context: object) -> None:
        self.skill_count = len(self.skills)

    def unique_skill_names(self) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for s in self.skills:
            key = s.name.lower()
            if key not in seen:
                seen.add(key)
                result.append(s.name)
        return result
