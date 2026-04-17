"""Trait profile inferred from resume text + optional free-text context."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ArchetypeLabel = Literal["Builder", "Operator", "Fixer", "Advisor", "Connector", "Innovator"]
WorkValue      = Literal["Stability", "Growth", "Impact", "Autonomy", "Craft"]
DecisionStyle  = Literal["Data-driven", "Intuitive", "Collaborative", "Directive"]
CompanyStage   = Literal["Startup", "Growth-stage", "Enterprise", "Turnaround"]


class TraitProfile(BaseModel):
    archetype:          ArchetypeLabel
    archetype_reason:   str          # one sentence — specific evidence from their career
    work_values:        list[WorkValue]  # top 2–3
    decision_style:     DecisionStyle
    ideal_stage:        CompanyStage
    identity_statement: str          # "You are a X who Y" — specific, never generic
    hidden_strengths:   list[str]    # 2–3 strengths the resume buries or omits
