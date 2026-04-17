"""
BigQuery candidate profile store.

Maintains the `candidate_profiles` table via MERGE — always derived
from `skill_extractions` as the source of truth.

One row per (candidate_id, skill_name).  Tracks:
  - first_seen / last_seen  — skill timeline
  - frequency               — how many extractions contained this skill
  - confidence_avg          — running average confidence
  - source_count            — distinct extraction_ids
"""

from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from reskillio.models.profile import CandidateProfile, ProfiledSkill
from reskillio.models.skill import SkillCategory

logger = logging.getLogger(__name__)

PROFILES_SCHEMA = [
    bigquery.SchemaField("candidate_id",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("first_seen",     "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("last_seen",      "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("frequency",      "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("confidence_avg", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("source_count",   "INTEGER",   mode="REQUIRED"),
]


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


class CandidateProfileStore:
    """
    Manages the candidate_profiles BigQuery table.

    Always reads from skill_extractions as the authoritative source —
    profiles are never written directly, only via MERGE.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "reskillio",
        profiles_table_id: str = "candidate_profiles",
        extractions_table_id: str = "skill_extractions",
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.profiles_table = f"{project_id}.{dataset_id}.{profiles_table_id}"
        self.extractions_table = f"{project_id}.{dataset_id}.{extractions_table_id}"
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
        """Create the candidate_profiles table if it doesn't exist."""
        table_ref = bigquery.Table(self.profiles_table, schema=PROFILES_SCHEMA)
        try:
            self.client.get_table(table_ref)
            logger.info(f"Table candidate_profiles already exists")
        except NotFound:
            self.client.create_table(table_ref)
            logger.info(f"Created table {self.profiles_table}")

    # ------------------------------------------------------------------
    # MERGE — upsert profile from extractions
    # ------------------------------------------------------------------

    def upsert_profile(self, candidate_id: str) -> int:
        """
        Recompute the candidate's profile from skill_extractions via MERGE.

        Safe to call after every extraction — idempotent, no double-counting.
        Returns the number of rows affected (inserted + updated).
        """
        merge_sql = f"""
        MERGE `{self.profiles_table}` T
        USING (
            SELECT
                candidate_id,
                skill_name,
                category,
                MIN(extracted_at)          AS first_seen,
                MAX(extracted_at)          AS last_seen,
                COUNT(*)                   AS frequency,
                AVG(confidence)            AS confidence_avg,
                COUNT(DISTINCT extraction_id) AS source_count
            FROM `{self.extractions_table}`
            WHERE candidate_id = @candidate_id
            GROUP BY candidate_id, skill_name, category
        ) S
        ON T.candidate_id = S.candidate_id
        AND T.skill_name   = S.skill_name

        WHEN MATCHED THEN UPDATE SET
            T.last_seen      = S.last_seen,
            T.frequency      = S.frequency,
            T.confidence_avg = S.confidence_avg,
            T.source_count   = S.source_count

        WHEN NOT MATCHED THEN INSERT (
            candidate_id, skill_name, category,
            first_seen, last_seen, frequency, confidence_avg, source_count
        ) VALUES (
            S.candidate_id, S.skill_name, S.category,
            S.first_seen,   S.last_seen,  S.frequency,
            S.confidence_avg, S.source_count
        )
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        job = self.client.query(merge_sql, job_config=job_config)
        job.result()  # wait for completion

        affected = job.num_dml_affected_rows or 0
        logger.info(
            f"Profile upserted for candidate='{candidate_id}' "
            f"({affected} rows affected)"
        )
        return affected

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_profile(self, candidate_id: str) -> CandidateProfile:
        """
        Fetch the full profile for a candidate.

        Returns an empty CandidateProfile if the candidate has no data.
        """
        query = f"""
            SELECT
                skill_name, category, first_seen, last_seen,
                frequency, confidence_avg, source_count
            FROM `{self.profiles_table}`
            WHERE candidate_id = @candidate_id
            ORDER BY frequency DESC, confidence_avg DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = self.client.query(query, job_config=job_config).result()

        skills = [
            ProfiledSkill(
                skill_name=row["skill_name"],
                category=SkillCategory(row["category"]),
                first_seen=row["first_seen"].replace(tzinfo=timezone.utc),
                last_seen=row["last_seen"].replace(tzinfo=timezone.utc),
                frequency=row["frequency"],
                confidence_avg=row["confidence_avg"],
                source_count=row["source_count"],
            )
            for row in rows
        ]

        return CandidateProfile(candidate_id=candidate_id, skills=skills)

    def get_skill_timeline(self, candidate_id: str, skill_name: str) -> dict | None:
        """
        Return timeline data for a specific skill — useful for evolution tracking.
        """
        query = f"""
            SELECT skill_name, first_seen, last_seen, frequency, confidence_avg
            FROM `{self.profiles_table}`
            WHERE candidate_id = @candidate_id
              AND LOWER(skill_name) = LOWER(@skill_name)
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                bigquery.ScalarQueryParameter("skill_name",   "STRING", skill_name),
            ]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        if not rows:
            return None
        row = rows[0]
        return {
            "skill_name":     row["skill_name"],
            "first_seen":     row["first_seen"].isoformat(),
            "last_seen":      row["last_seen"].isoformat(),
            "days_active":    (row["last_seen"] - row["first_seen"]).days,
            "frequency":      row["frequency"],
            "confidence_avg": row["confidence_avg"],
        }