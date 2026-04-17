"""Domain models for industry match scoring."""

from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field

_INDUSTRY_LABELS = {
    "data_ai":              "Data & AI",
    "software_engineering": "Software Engineering",
    "fintech":              "Financial Technology",
    "healthtech":           "Health Technology",
    "ecommerce":            "E-Commerce & Retail Tech",
    "cybersecurity":        "Cybersecurity",
    "cloud_devops":         "Cloud & DevOps",
    "product_management":   "Product Management",
}


class IndustryScore(BaseModel):
    rank: int
    industry: str
    industry_label: str
    match_score: float = Field(ge=0.0, le=100.0)
    cosine_distance: float


class IndustryMatchResult(BaseModel):
    candidate_id: str
    top_industry: str
    top_industry_label: str
    scores: list[IndustryScore]
    computed_at: datetime
    method: str = "bqml_cosine_ml_distance"

    @classmethod
    def from_bq_rows(
        cls, candidate_id: str, rows: list[dict]
    ) -> "IndustryMatchResult":
        scores = []
        for rank, row in enumerate(rows, 1):
            industry = row["industry"]
            scores.append(
                IndustryScore(
                    rank=rank,
                    industry=industry,
                    industry_label=_INDUSTRY_LABELS.get(industry, industry),
                    match_score=row["match_score"],
                    cosine_distance=round(row["cosine_distance"], 6),
                )
            )
        top = scores[0] if scores else None
        return cls(
            candidate_id=candidate_id,
            top_industry=top.industry if top else "",
            top_industry_label=top.industry_label if top else "",
            scores=scores,
            computed_at=datetime.utcnow(),
        )