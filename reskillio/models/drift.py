"""Domain models for F13 — model drift monitoring."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

# Alert fires when unknown_skill_rate exceeds this value
UNKNOWN_RATE_THRESHOLD: float = 0.20

# Cloud Monitoring custom metric types
METRIC_UNKNOWN_RATE      = "custom.googleapis.com/reskillio/unknown_skill_rate"
METRIC_AVG_CONFIDENCE    = "custom.googleapis.com/reskillio/avg_confidence"
METRIC_TAXONOMY_COVERAGE = "custom.googleapis.com/reskillio/taxonomy_coverage"


class DriftMetric(BaseModel):
    """One row in the BigQuery drift_metrics table — written after every extraction."""
    run_id:             str
    candidate_id:       Optional[str]   = None
    model_used:         str
    skill_count:        int             = Field(..., ge=0)
    unknown_count:      int             = Field(..., ge=0)
    known_count:        int             = Field(..., ge=0)
    unknown_skill_rate: float           = Field(..., ge=0.0, le=1.0)
    taxonomy_coverage:  float           = Field(..., ge=0.0, le=1.0)
    avg_confidence:     float           = Field(..., ge=0.0, le=1.0)
    min_confidence:     float           = Field(..., ge=0.0, le=1.0)
    alert_triggered:    bool
    recorded_at:        datetime


class DriftReport(BaseModel):
    """Result returned by run_drift_monitor()."""
    metric:                DriftMetric
    written_to_bq:         bool
    written_to_monitoring: bool
    alert_triggered:       bool
    message:               str


class DriftTrendPoint(BaseModel):
    """One hourly bucket in the unknown_rate trend query."""
    hour:               datetime
    avg_unknown_rate:   float
    avg_confidence:     float
    avg_taxonomy_coverage: float
    extraction_count:   int