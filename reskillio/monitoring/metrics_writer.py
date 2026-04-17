"""
F13 — Cloud Monitoring custom metric writer.

Writes time-series data points for three custom metrics:
  • custom.googleapis.com/reskillio/unknown_skill_rate
  • custom.googleapis.com/reskillio/avg_confidence
  • custom.googleapis.com/reskillio/taxonomy_coverage

Uses the `global` monitored resource so no VM/container context is required.
Metric descriptors are created idempotently on first write.
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from reskillio.models.drift import (
    METRIC_AVG_CONFIDENCE,
    METRIC_TAXONOMY_COVERAGE,
    METRIC_UNKNOWN_RATE,
)

_DESCRIPTOR_CACHE: set[str] = set()   # track which descriptors already exist this process


def _get_client(project_id: str):
    from google.cloud import monitoring_v3
    return monitoring_v3.MetricServiceClient()


def _project_name(project_id: str) -> str:
    return f"projects/{project_id}"


# ---------------------------------------------------------------------------
# Metric descriptor (create once per metric type)
# ---------------------------------------------------------------------------

_METRIC_DEFS = {
    METRIC_UNKNOWN_RATE: {
        "display_name": "ReSkillio Unknown Skill Rate",
        "description":  (
            "Fraction of extracted skills whose category is UNKNOWN "
            "(not in the taxonomy). Alert triggers when > 20%."
        ),
        "unit": "1",
    },
    METRIC_AVG_CONFIDENCE: {
        "display_name": "ReSkillio Avg Extraction Confidence",
        "description":  "Mean confidence score (0–1) across all skills in one extraction run.",
        "unit": "1",
    },
    METRIC_TAXONOMY_COVERAGE: {
        "display_name": "ReSkillio Taxonomy Coverage",
        "description":  "Fraction of extracted skills that are present in the taxonomy.",
        "unit": "1",
    },
}


def _ensure_descriptor(client, project_id: str, metric_type: str) -> None:
    """Create the custom metric descriptor if not already present."""
    if metric_type in _DESCRIPTOR_CACHE:
        return

    # MetricDescriptor and LabelDescriptor live in google.api, not monitoring_v3
    from google.api import label_pb2, metric_pb2

    defn       = _METRIC_DEFS[metric_type]
    descriptor = metric_pb2.MetricDescriptor()
    descriptor.type        = metric_type
    descriptor.metric_kind = metric_pb2.MetricDescriptor.MetricKind.GAUGE
    descriptor.value_type  = metric_pb2.MetricDescriptor.ValueType.DOUBLE
    descriptor.unit        = defn["unit"]
    descriptor.display_name = defn["display_name"]
    descriptor.description  = defn["description"]

    for key, desc in [
        ("candidate_id", "Candidate ID being processed (empty = batch run)"),
        ("model_used",   "spaCy model used for extraction"),
    ]:
        label = label_pb2.LabelDescriptor()
        label.key         = key
        label.value_type  = label_pb2.LabelDescriptor.ValueType.STRING
        label.description = desc
        descriptor.labels.append(label)

    try:
        client.create_metric_descriptor(
            name=_project_name(project_id),
            metric_descriptor=descriptor,
        )
        logger.info(f"[drift] Created metric descriptor: {metric_type}")
    except Exception as exc:
        if "already exists" in str(exc).lower() or "409" in str(exc):
            pass  # already created — fine
        else:
            logger.warning(f"[drift] Descriptor creation warning for {metric_type}: {exc}")

    _DESCRIPTOR_CACHE.add(metric_type)


# ---------------------------------------------------------------------------
# Time-series writer
# ---------------------------------------------------------------------------

def _write_point(
    client,
    project_id:  str,
    metric_type: str,
    value:       float,
    labels:      dict[str, str],
) -> None:
    from google.cloud import monitoring_v3
    from google.protobuf import timestamp_pb2

    now     = time.time()
    seconds = int(now)
    nanos   = int((now - seconds) * 1e9)

    ts       = timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)
    interval = monitoring_v3.TimeInterval(end_time=ts)
    point    = monitoring_v3.Point(
        interval=interval,
        value=monitoring_v3.TypedValue(double_value=value),
    )

    series = monitoring_v3.TimeSeries()
    series.metric.type = metric_type
    series.metric.labels.update(labels)
    series.resource.type = "global"
    series.resource.labels["project_id"] = project_id
    series.points.append(point)

    client.create_time_series(
        name=_project_name(project_id),
        time_series=[series],
    )


def write_drift_metrics(
    project_id:        str,
    unknown_rate:      float,
    avg_confidence:    float,
    taxonomy_coverage: float,
    candidate_id:      Optional[str] = None,
    model_used:        str = "en_core_web_lg",
) -> None:
    """
    Write all three drift metrics to Cloud Monitoring.

    Descriptor creation is idempotent.  Individual metric failures are logged
    but do not block the remaining writes.
    """
    client = _get_client(project_id)
    labels = {
        "candidate_id": candidate_id or "",
        "model_used":   model_used,
    }

    for metric_type, value in [
        (METRIC_UNKNOWN_RATE,      unknown_rate),
        (METRIC_AVG_CONFIDENCE,    avg_confidence),
        (METRIC_TAXONOMY_COVERAGE, taxonomy_coverage),
    ]:
        try:
            _ensure_descriptor(client, project_id, metric_type)
            _write_point(client, project_id, metric_type, value, labels)
            logger.debug(f"[drift] Monitoring ← {metric_type.split('/')[-1]}={value:.4f}")
        except Exception as exc:
            logger.warning(f"[drift] Monitoring write skipped for {metric_type}: {exc}")
