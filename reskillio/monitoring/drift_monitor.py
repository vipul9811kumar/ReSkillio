"""
F13 — Drift monitor orchestrator.

Public entry point: run_drift_monitor()
  1. Computes DriftMetric from an ExtractionResult.
  2. Writes to BigQuery drift_metrics table (non-fatal on error).
  3. Writes to Cloud Monitoring custom metrics (non-fatal on error).
  4. Returns a DriftReport.

Called automatically by run_skill_extraction() whenever GCP is configured.
Failures never propagate to the caller — drift monitoring is best-effort.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.models.drift import (
    DriftMetric,
    DriftReport,
    UNKNOWN_RATE_THRESHOLD,
)
from reskillio.models.skill import ExtractionResult, SkillCategory


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_drift_metric(
    result:       ExtractionResult,
    candidate_id: Optional[str] = None,
) -> DriftMetric:
    """Compute drift metrics from a single ExtractionResult."""
    skills        = result.skills
    skill_count   = len(skills)
    unknown_count = sum(1 for s in skills if s.category == SkillCategory.UNKNOWN)
    known_count   = skill_count - unknown_count

    if skill_count > 0:
        unknown_rate      = unknown_count / skill_count
        taxonomy_coverage = known_count   / skill_count
        avg_confidence    = sum(s.confidence for s in skills) / skill_count
        min_confidence    = min(s.confidence for s in skills)
    else:
        unknown_rate      = 0.0
        taxonomy_coverage = 1.0
        avg_confidence    = 0.0
        min_confidence    = 0.0

    alert_triggered = unknown_rate > UNKNOWN_RATE_THRESHOLD

    return DriftMetric(
        run_id=str(uuid.uuid4()),
        candidate_id=candidate_id,
        model_used=result.model_used or "unknown",
        skill_count=skill_count,
        unknown_count=unknown_count,
        known_count=known_count,
        unknown_skill_rate=round(unknown_rate, 6),
        taxonomy_coverage=round(taxonomy_coverage, 6),
        avg_confidence=round(avg_confidence, 6),
        min_confidence=round(min_confidence, 6),
        alert_triggered=alert_triggered,
        recorded_at=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_drift_monitor(
    result:       ExtractionResult,
    project_id:   str,
    region:       str = "us-central1",
    candidate_id: Optional[str] = None,
) -> DriftReport:
    """
    Compute drift metrics, write to BQ and Cloud Monitoring.

    Parameters
    ----------
    result:       The ExtractionResult just produced by the skill extractor.
    project_id:   GCP project (for BQ and Cloud Monitoring writes).
    region:       GCP region (informational — BQ/Monitoring are global).
    candidate_id: Optional candidate identifier for label cardinality.

    Returns
    -------
    DriftReport — always returns, never raises.
    """
    metric = compute_drift_metric(result, candidate_id=candidate_id)

    if metric.alert_triggered:
        logger.warning(
            f"[drift] ALERT — unknown_rate={metric.unknown_skill_rate:.3f} "
            f"exceeds threshold={UNKNOWN_RATE_THRESHOLD}  "
            f"run={metric.run_id[:8]}  candidate={candidate_id or '—'}"
        )
    else:
        logger.debug(
            f"[drift] OK — unknown_rate={metric.unknown_skill_rate:.3f}  "
            f"avg_confidence={metric.avg_confidence:.3f}  "
            f"taxonomy_coverage={metric.taxonomy_coverage:.3f}"
        )

    written_to_bq         = False
    written_to_monitoring = False

    # ------------------------------------------------------------------ #
    # BigQuery write
    # ------------------------------------------------------------------ #
    try:
        from reskillio.monitoring.drift_store import DriftMetricStore
        store = DriftMetricStore(project_id=project_id)
        store.write_metric(metric)
        written_to_bq = True
    except Exception as exc:
        logger.warning(f"[drift] BQ write failed (non-fatal): {exc}")

    # ------------------------------------------------------------------ #
    # Cloud Monitoring write
    # ------------------------------------------------------------------ #
    try:
        from reskillio.monitoring.metrics_writer import write_drift_metrics
        write_drift_metrics(
            project_id=project_id,
            unknown_rate=metric.unknown_skill_rate,
            avg_confidence=metric.avg_confidence,
            taxonomy_coverage=metric.taxonomy_coverage,
            candidate_id=candidate_id,
            model_used=metric.model_used,
        )
        written_to_monitoring = True
    except Exception as exc:
        logger.warning(f"[drift] Cloud Monitoring write failed (non-fatal): {exc}")

    parts = []
    if metric.alert_triggered:
        parts.append(
            f"ALERT: unknown_rate={metric.unknown_skill_rate:.3f} "
            f"exceeds threshold={UNKNOWN_RATE_THRESHOLD}"
        )
    parts.append(f"BQ={'ok' if written_to_bq else 'skipped'}")
    parts.append(f"monitoring={'ok' if written_to_monitoring else 'skipped'}")

    return DriftReport(
        metric=metric,
        written_to_bq=written_to_bq,
        written_to_monitoring=written_to_monitoring,
        alert_triggered=metric.alert_triggered,
        message=" | ".join(parts),
    )