"""
F13 — Drift monitoring API routes.

Endpoints
---------
GET  /monitoring/drift/recent          — last N drift records from BQ
GET  /monitoring/drift/trend           — hourly trend for the past N days
GET  /monitoring/drift/alerts          — recent rows where alert_triggered=TRUE
GET  /monitoring/drift/alert-policy    — Cloud Monitoring alert policy status
POST /monitoring/drift/record          — manually record drift for an extraction
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings
from reskillio.models.drift import DriftMetric, DriftReport, DriftTrendPoint
from reskillio.models.skill import ExtractionResult, Skill, SkillCategory
from reskillio.monitoring.drift_monitor import compute_drift_metric, run_drift_monitor

router = APIRouter(prefix="/monitoring", tags=["drift-monitoring"])


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class RecordDriftRequest(BaseModel):
    """Manually record drift metrics for a text/extraction result."""
    candidate_id: Optional[str] = None
    model_used:   str           = Field(default="en_core_web_lg")
    skills: list[dict] = Field(
        ...,
        description=(
            "List of extracted skills. Each entry must have 'name', 'category' "
            "(technical|soft|tool|certification|unknown), and 'confidence' (0–1)."
        ),
        examples=[[
            {"name": "Python",  "category": "technical", "confidence": 1.0},
            {"name": "FooBar",  "category": "unknown",   "confidence": 0.8},
        ]],
    )


# ---------------------------------------------------------------------------
# Shared dependency
# ---------------------------------------------------------------------------

def _require_gcp() -> None:
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )


def _get_store():
    from reskillio.monitoring.drift_store import DriftMetricStore
    return DriftMetricStore(project_id=settings.gcp_project_id)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get(
    "/drift/recent",
    response_model=list[DriftMetric],
    status_code=status.HTTP_200_OK,
    summary="Recent drift metrics",
)
def get_recent_drift(
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return"),
) -> list[DriftMetric]:
    """
    Return the most recent *limit* drift records from BigQuery, newest first.
    Each record covers one extraction run: avg_confidence, unknown_skill_rate,
    taxonomy_coverage, and whether an alert was triggered.
    """
    _require_gcp()
    try:
        return _get_store().get_recent_metrics(limit=limit)
    except Exception as exc:
        logger.error(f"drift/recent failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch drift metrics.",
        ) from exc


@router.get(
    "/drift/trend",
    response_model=list[DriftTrendPoint],
    status_code=status.HTTP_200_OK,
    summary="Hourly drift trend",
)
def get_drift_trend(
    days: int = Query(default=7, ge=1, le=90, description="Look-back window in days"),
) -> list[DriftTrendPoint]:
    """
    Return hourly averages of unknown_rate, avg_confidence, and taxonomy_coverage
    for the past *days* days. Use this data to render trend charts.
    """
    _require_gcp()
    try:
        return _get_store().get_hourly_trend(days=days)
    except Exception as exc:
        logger.error(f"drift/trend failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch drift trend.",
        ) from exc


@router.get(
    "/drift/alerts",
    response_model=list[DriftMetric],
    status_code=status.HTTP_200_OK,
    summary="Recent alert-triggered drift records",
)
def get_alert_history(
    limit: int = Query(default=20, ge=1, le=200),
) -> list[DriftMetric]:
    """Return the most recent extraction runs where unknown_skill_rate > 20%."""
    _require_gcp()
    try:
        return _get_store().get_alert_history(limit=limit)
    except Exception as exc:
        logger.error(f"drift/alerts failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch alert history.",
        ) from exc


@router.get(
    "/drift/alert-policy",
    response_model=dict,
    status_code=status.HTTP_200_OK,
    summary="Cloud Monitoring alert policy status",
)
def get_alert_policy_status() -> dict:
    """
    Return whether the Cloud Monitoring alert policy exists and is enabled.
    Run `scripts/setup_monitoring.py` to create it if missing.
    """
    _require_gcp()
    try:
        from reskillio.monitoring.alert_policy import get_policy_status
        return get_policy_status(project_id=settings.gcp_project_id)
    except Exception as exc:
        logger.error(f"drift/alert-policy failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check alert policy.",
        ) from exc


@router.post(
    "/drift/record",
    response_model=DriftReport,
    status_code=status.HTTP_200_OK,
    summary="Manually record drift metrics for an extraction",
)
def record_drift(body: RecordDriftRequest) -> DriftReport:
    """
    Compute and record drift metrics for a supplied list of extracted skills.

    Useful for testing the monitoring pipeline or back-filling metrics for
    an extraction that ran outside the normal API flow.
    """
    _require_gcp()

    try:
        skills = [
            Skill(
                name=s["name"],
                category=SkillCategory(s.get("category", "unknown")),
                confidence=float(s.get("confidence", 1.0)),
                source_text=s.get("source_text"),
            )
            for s in body.skills
        ]
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid skill entry: {exc}",
        ) from exc

    result = ExtractionResult(
        input_text="[manual record]",
        skills=skills,
        model_used=body.model_used,
    )

    try:
        return run_drift_monitor(
            result=result,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            candidate_id=body.candidate_id,
        )
    except Exception as exc:
        logger.error(f"drift/record failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Drift recording failed.",
        ) from exc