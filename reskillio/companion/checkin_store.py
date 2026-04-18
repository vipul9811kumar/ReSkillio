"""BigQuery storage for weekly check-ins and digests."""

from __future__ import annotations

import json
import logging
from typing import Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from reskillio.companion.models import WeeklyCheckin, WeeklyDigest

logger = logging.getLogger(__name__)

CHECKIN_SCHEMA = [
    bigquery.SchemaField("candidate_id",          "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("week_number",            "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("week_start",             "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("checked_in_at",          "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("hours_on_courses",       "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("courses_completed",      "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("applications_sent",      "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("interviews_scheduled",   "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("interviews_completed",   "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("offers_received",        "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("gap_score",              "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("gap_score_delta",        "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top_industry_score",     "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("top_industry",           "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("digest_generated",       "BOOL",      mode="NULLABLE"),
    bigquery.SchemaField("digest_id",              "STRING",    mode="NULLABLE"),
]

DIGEST_SCHEMA = [
    bigquery.SchemaField("digest_id",             "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",          "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("week_number",           "INT64",     mode="REQUIRED"),
    bigquery.SchemaField("week_start",            "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("generated_at",          "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("gap_score",             "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("gap_score_delta",       "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("course_completion",     "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("applications_sent",     "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("interviews_active",     "INT64",     mode="NULLABLE"),
    bigquery.SchemaField("opening_message",       "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("market_signal",         "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("gemini_narrative",      "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top_action",            "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("sections_json",         "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("action_items_json",     "STRING",    mode="NULLABLE"),
]


class CompanionStore:
    def __init__(self, project_id: str):
        self.project_id  = project_id
        self.dataset_id  = "reskillio"
        self._client: Optional[bigquery.Client] = None

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def _table(self, name: str) -> str:
        return f"{self.project_id}.{self.dataset_id}.{name}"

    def ensure_tables(self) -> None:
        for table_name, schema in [
            ("weekly_checkins",   CHECKIN_SCHEMA),
            ("companion_digests", DIGEST_SCHEMA),
        ]:
            full_id  = self._table(table_name)
            tbl      = bigquery.Table(full_id, schema=schema)
            try:
                self.client.get_table(tbl)
            except NotFound:
                self.client.create_table(tbl)
                logger.info(f"[companion_store] Created {table_name}")

    # ── Writes ───────────────────────────────────────────────────────────────

    def save_checkin(self, checkin: WeeklyCheckin) -> None:
        row = {
            "candidate_id":          checkin.candidate_id,
            "week_number":           checkin.week_number,
            "week_start":            checkin.week_start.isoformat(),
            "checked_in_at":         checkin.checked_in_at.isoformat(),
            "hours_on_courses":      checkin.hours_on_courses,
            "courses_completed":     json.dumps(checkin.courses_completed),
            "applications_sent":     checkin.applications_sent,
            "interviews_scheduled":  checkin.interviews_scheduled,
            "interviews_completed":  checkin.interviews_completed,
            "offers_received":       checkin.offers_received,
            "gap_score":             checkin.gap_score,
            "gap_score_delta":       checkin.gap_score_delta,
            "top_industry_score":    checkin.top_industry_score,
            "top_industry":          checkin.top_industry,
            "digest_generated":      checkin.digest_generated,
            "digest_id":             checkin.digest_id,
        }
        errors = self.client.insert_rows_json(self._table("weekly_checkins"), [row])
        if errors:
            raise RuntimeError(f"Checkin save failed: {errors}")

    def save_digest(self, digest: WeeklyDigest) -> None:
        row = {
            "digest_id":         digest.digest_id,
            "candidate_id":      digest.candidate_id,
            "week_number":       digest.week_number,
            "week_start":        digest.week_start.isoformat(),
            "generated_at":      digest.generated_at.isoformat(),
            "gap_score":         digest.gap_score,
            "gap_score_delta":   digest.gap_score_delta,
            "course_completion": digest.course_completion,
            "applications_sent": digest.applications_sent,
            "interviews_active": digest.interviews_active,
            "opening_message":   digest.opening_message,
            "market_signal":     digest.market_signal,
            "gemini_narrative":  digest.gemini_narrative,
            "top_action":        digest.top_action,
            "sections_json":     json.dumps([s.dict() for s in digest.sections]),
            "action_items_json": json.dumps([a.dict() for a in digest.action_items]),
        }
        errors = self.client.insert_rows_json(self._table("companion_digests"), [row])
        if errors:
            raise RuntimeError(f"Digest save failed: {errors}")

    # ── Reads ────────────────────────────────────────────────────────────────

    def get_checkins(self, candidate_id: str, limit: int = 12) -> list[dict]:
        query = f"""
            SELECT * FROM `{self._table("weekly_checkins")}`
            WHERE candidate_id = @cid
            ORDER BY week_number DESC
            LIMIT {limit}
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("cid", "STRING", candidate_id)
        ])
        return [dict(r) for r in self.client.query(query, job_config=cfg).result()]

    def get_previous_checkin(self, candidate_id: str, week_number: int) -> Optional[dict]:
        rows = self.get_checkins(candidate_id, limit=20)
        return next((r for r in rows if r["week_number"] == week_number - 1), None)

    def get_latest_digest(self, candidate_id: str) -> Optional[dict]:
        query = f"""
            SELECT * FROM `{self._table("companion_digests")}`
            WHERE candidate_id = @cid
            ORDER BY week_number DESC LIMIT 1
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("cid", "STRING", candidate_id)
        ])
        rows = list(self.client.query(query, job_config=cfg).result())
        return dict(rows[0]) if rows else None

    def get_digest_history(self, candidate_id: str) -> list[dict]:
        query = f"""
            SELECT digest_id, week_number, week_start, gap_score,
                   gap_score_delta, applications_sent, gemini_narrative
            FROM `{self._table("companion_digests")}`
            WHERE candidate_id = @cid
            ORDER BY week_number ASC
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("cid", "STRING", candidate_id)
        ])
        return [dict(r) for r in self.client.query(query, job_config=cfg).result()]

    def next_week_number(self, candidate_id: str) -> int:
        checkins = self.get_checkins(candidate_id, limit=100)
        return (max((c["week_number"] for c in checkins), default=0) + 1)

    def get_candidates_due_for_digest(self, week_start: str) -> list[str]:
        """Return candidate_ids who checked in on week_start but have no digest yet."""
        query = f"""
            SELECT DISTINCT c.candidate_id
            FROM `{self._table("weekly_checkins")}` c
            WHERE c.week_start = @week_start
              AND (c.digest_generated IS NULL OR c.digest_generated = FALSE)
              AND NOT EXISTS (
                SELECT 1 FROM `{self._table("companion_digests")}` d
                WHERE d.candidate_id = c.candidate_id
                  AND d.week_start = @week_start
              )
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("week_start", "STRING", week_start)
        ])
        return [r["candidate_id"] for r in self.client.query(query, job_config=cfg).result()]

    def mark_digest_generated(self, candidate_id: str, week_number: int, digest_id: str) -> None:
        """DML UPDATE — marks the checkin row as having a digest."""
        query = f"""
            UPDATE `{self._table("weekly_checkins")}`
            SET digest_generated = TRUE, digest_id = @digest_id
            WHERE candidate_id = @cid AND week_number = @week_num
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[
            bigquery.ScalarQueryParameter("cid",        "STRING", candidate_id),
            bigquery.ScalarQueryParameter("week_num",   "INT64",  week_number),
            bigquery.ScalarQueryParameter("digest_id",  "STRING", digest_id),
        ])
        self.client.query(query, job_config=cfg).result()

    def get_latest_checkin(self, candidate_id: str) -> Optional[dict]:
        checkins = self.get_checkins(candidate_id, limit=1)
        return checkins[0] if checkins else None
