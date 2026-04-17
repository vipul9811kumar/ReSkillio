"""Domain models for job description ingestion."""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

from reskillio.models.skill import Skill, SkillCategory


class SeniorityLevel(str, Enum):
    JUNIOR  = "junior"
    MID     = "mid"
    SENIOR  = "senior"
    LEAD    = "lead"
    STAFF   = "staff"
    MANAGER = "manager"
    UNKNOWN = "unknown"


class RequirementLevel(str, Enum):
    REQUIRED  = "required"
    PREFERRED = "preferred"


class Industry(str, Enum):
    DATA_AI              = "data_ai"
    SOFTWARE_ENGINEERING = "software_engineering"
    FINTECH              = "fintech"
    HEALTHTECH           = "healthtech"
    ECOMMERCE            = "ecommerce"
    CYBERSECURITY        = "cybersecurity"
    CLOUD_DEVOPS         = "cloud_devops"
    PRODUCT_MANAGEMENT   = "product_management"
    UNKNOWN              = "unknown"


class JDSkill(BaseModel):
    """A skill extracted from a job description, with requirement level."""

    name: str
    category: SkillCategory
    confidence: float = Field(ge=0.0, le=1.0)
    requirement: RequirementLevel = RequirementLevel.REQUIRED


class JDExtractionResult(BaseModel):
    """Full output of a JD ingestion run."""

    jd_id: str
    title: Optional[str] = None
    company: Optional[str] = None
    industry: Industry = Industry.UNKNOWN
    seniority: SeniorityLevel = SeniorityLevel.UNKNOWN
    source_url: Optional[str] = None
    required_skills: list[JDSkill] = Field(default_factory=list)
    preferred_skills: list[JDSkill] = Field(default_factory=list)
    total_skill_count: int = 0
    stored: bool = False

    def model_post_init(self, __context: object) -> None:
        self.total_skill_count = len(self.required_skills) + len(self.preferred_skills)

    def all_skills(self) -> list[JDSkill]:
        return self.required_skills + self.preferred_skills
