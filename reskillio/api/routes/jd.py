"""POST /jd — job description ingestion endpoint."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from reskillio.models.jd import Industry, JDExtractionResult
from reskillio.pipelines.jd_pipeline import run_jd_pipeline
from reskillio.storage.jd_store import JDStore
from config.settings import settings

router = APIRouter(prefix="/jd", tags=["job-description"])


def _get_jd_store() -> JDStore | None:
    if not settings.gcp_project_id:
        return None
    return JDStore(project_id=settings.gcp_project_id)


class JDRequest(BaseModel):
    text: str = Field(..., min_length=50, description="Raw job description text")
    title: Optional[str] = Field(default=None, description="Job title")
    company: Optional[str] = Field(default=None, description="Company name")
    industry: Optional[Industry] = Field(
        default=None,
        description="Industry override — auto-detected if omitted"
    )
    source_url: Optional[str] = Field(default=None, description="Source URL if fetched from web")


@router.post("", response_model=JDExtractionResult, status_code=status.HTTP_200_OK)
def ingest_jd(
    body: JDRequest,
    store: JDStore | None = Depends(_get_jd_store),
) -> JDExtractionResult:
    """
    Ingest a job description and extract structured skill data.

    - Auto-detects seniority level from title and body text.
    - Auto-detects industry from keywords (override with `industry` field).
    - Splits into required vs preferred skill sections.
    - Persists to jd_profiles + jd_skills tables.
    """
    logger.info(f"JD ingest request: title='{body.title}' company='{body.company}'")

    try:
        result = run_jd_pipeline(
            text=body.text,
            title=body.title,
            company=body.company,
            industry=body.industry,
            source_url=body.source_url,
            model_name=settings.spacy_model,
            store=store,
        )
    except Exception as exc:
        logger.error(f"JD pipeline failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JD processing failed.",
        ) from exc

    return result


@router.get("/industries", response_model=dict)
def list_industries() -> dict:
    """Return the 8 supported industries and their labels."""
    return {
        industry.value: industry.value.replace("_", " ").title()
        for industry in Industry
        if industry != Industry.UNKNOWN
    }


@router.get("/{jd_id}", response_model=list)
def get_jd(
    jd_id: str,
    store: JDStore | None = Depends(_get_jd_store),
) -> list:
    """Fetch a stored JD and its extracted skills by jd_id."""
    if store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="GCP not configured.")
    rows = store.get_jd(jd_id)
    if not rows:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"JD '{jd_id}' not found.")
    return rows
