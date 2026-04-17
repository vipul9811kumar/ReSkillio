"""
F13 — BigQuery store for drift_metrics table.

Schema
------
One row per extraction run, written immediately after every call to
run_skill_extraction().  Enables SQL-based trend queries and dashboards.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from loguru import logger

from reskillio.models.drift import DriftMetric, DriftTrendPoint

_TABLE_ID = "drift_metrics"

_SCHEMA = [
    bigquery.SchemaField("run_id",             "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",       "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("model_used",         "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_count",        "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("unknown_count",      "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("known_count",        "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("unknown_skill_rate", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("taxonomy_coverage",  "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("avg_confidence",     "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("min_confidence",     "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("alert_triggered",    "BOOL",      mode="REQUIRED"),
    bigquery.SchemaField("recorded_at",        "TIMESTAMP", mode="REQUIRED"),
]


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                settings.google_application_credentials
            )
    except Exception:
        pass


class DriftMetricStore:
    def __init__(
        self,
        project_id: str,
        dataset_id: str = "reskillio",
        table_id:   str = _TABLE_ID,
    ) -> None:
        self.project_id     = project_id
        self.dataset_id     = dataset_id
        self.table_id       = table_id
        self.full_table_id  = f"{project_id}.{dataset_id}.{table_id}"
        self._client: Optional[bigquery.Client] = None

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            _apply_credentials()
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create drift_metrics table if absent. Safe to call repeatedly."""
        table_ref = bigquery.Table(self.full_table_id, schema=_SCHEMA)
        try:
            self.client.get_table(table_ref)
            logger.info(f"Table {self.table_id} already exists")
        except NotFound:
            self.client.create_table(table_ref)
            logger.info(f"Created table {self.full_table_id}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_metric(self, metric: DriftMetric) -> None:
        row = {
            "run_id":             metric.run_id,
            "candidate_id":       metric.candidate_id,
            "model_used":         metric.model_used,
            "skill_count":        metric.skill_count,
            "unknown_count":      metric.unknown_count,
            "known_count":        metric.known_count,
            "unknown_skill_rate": metric.unknown_skill_rate,
            "taxonomy_coverage":  metric.taxonomy_coverage,
            "avg_confidence":     metric.avg_confidence,
            "min_confidence":     metric.min_confidence,
            "alert_triggered":    metric.alert_triggered,
            "recorded_at":        metric.recorded_at.isoformat(),
        }
        errors = self.client.insert_rows_json(self.full_table_id, [row])
        if errors:
            raise RuntimeError(f"drift_metrics insert failed: {errors}")
        logger.debug(
            f"[drift] BQ write: run={metric.run_id[:8]}  "
            f"unknown_rate={metric.unknown_skill_rate:.3f}  "
            f"alert={metric.alert_triggered}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_recent_metrics(self, limit: int = 50) -> list[DriftMetric]:
        """Return the most recent *limit* drift records, newest first."""
        query = f"""
            SELECT *
            FROM `{self.full_table_id}`
            ORDER BY recorded_at DESC
            LIMIT @limit
        """
        cfg  = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        )
        rows = self.client.query(query, job_config=cfg).result()
        return [
            DriftMetric(
                run_id=r["run_id"],
                candidate_id=r["candidate_id"],
                model_used=r["model_used"],
                skill_count=r["skill_count"],
                unknown_count=r["unknown_count"],
                known_count=r["known_count"],
                unknown_skill_rate=r["unknown_skill_rate"],
                taxonomy_coverage=r["taxonomy_coverage"],
                avg_confidence=r["avg_confidence"],
                min_confidence=r["min_confidence"],
                alert_triggered=r["alert_triggered"],
                recorded_at=r["recorded_at"],
            )
            for r in rows
        ]

    def get_hourly_trend(self, days: int = 7) -> list[DriftTrendPoint]:
        """
        Return hourly aggregates of drift metrics for the past *days* days.
        Useful for dashboard trend charts.
        """
        query = f"""
            SELECT
                TIMESTAMP_TRUNC(recorded_at, HOUR) AS hour,
                ROUND(AVG(unknown_skill_rate), 4)  AS avg_unknown_rate,
                ROUND(AVG(avg_confidence), 4)      AS avg_confidence,
                ROUND(AVG(taxonomy_coverage), 4)   AS avg_taxonomy_coverage,
                COUNT(*)                           AS extraction_count
            FROM `{self.full_table_id}`
            WHERE recorded_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days DAY)
            GROUP BY hour
            ORDER BY hour DESC
        """
        cfg  = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("days", "INT64", days)]
        )
        rows = self.client.query(query, job_config=cfg).result()
        return [
            DriftTrendPoint(
                hour=r["hour"],
                avg_unknown_rate=r["avg_unknown_rate"],
                avg_confidence=r["avg_confidence"],
                avg_taxonomy_coverage=r["avg_taxonomy_coverage"],
                extraction_count=r["extraction_count"],
            )
            for r in rows
        ]

    def get_alert_history(self, limit: int = 20) -> list[DriftMetric]:
        """Return the most recent rows where alert_triggered = TRUE."""
        query = f"""
            SELECT *
            FROM `{self.full_table_id}`
            WHERE alert_triggered = TRUE
            ORDER BY recorded_at DESC
            LIMIT @limit
        """
        cfg  = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("limit", "INT64", limit)]
        )
        rows = self.client.query(query, job_config=cfg).result()
        return [
            DriftMetric(
                run_id=r["run_id"],
                candidate_id=r["candidate_id"],
                model_used=r["model_used"],
                skill_count=r["skill_count"],
                unknown_count=r["unknown_count"],
                known_count=r["known_count"],
                unknown_skill_rate=r["unknown_skill_rate"],
                taxonomy_coverage=r["taxonomy_coverage"],
                avg_confidence=r["avg_confidence"],
                min_confidence=r["min_confidence"],
                alert_triggered=r["alert_triggered"],
                recorded_at=r["recorded_at"],
            )
            for r in rows
        ]