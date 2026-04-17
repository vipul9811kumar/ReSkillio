"""
BigQuery storage layer for ReSkillio skill extractions.
Maps directly to reskillio.models.skill — no translation layer.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from reskillio.models.skill import ExtractionResult, Skill, SkillCategory

logger = logging.getLogger(__name__)


def _apply_credentials() -> None:
    """Ensure GOOGLE_APPLICATION_CREDENTIALS is set in the OS env from settings."""
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Schema — mirrors Skill + ExtractionResult exactly
# ---------------------------------------------------------------------------

SKILLS_SCHEMA = [
    bigquery.SchemaField("extraction_id",  "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence",     "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("source_text",    "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("model_used",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("extracted_at",   "TIMESTAMP", mode="REQUIRED"),
]

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class BigQuerySkillStore:
    """
    Writes ExtractionResult objects to BigQuery.
    One row per skill — easy to query, filter, and join.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "reskillio",
        table_id: str = "skill_extractions",
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
        self._client: Optional[bigquery.Client] = None

    # ------------------------------------------------------------------
    # Lazy client init
    # ------------------------------------------------------------------

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            _apply_credentials()
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_dataset_and_table(self) -> None:
        """Create dataset + table if they don't exist. Safe to call repeatedly."""
        # Dataset
        dataset_ref = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
        dataset_ref.location = "US"
        try:
            self.client.get_dataset(dataset_ref)
            logger.info(f"Dataset {self.dataset_id} already exists")
        except NotFound:
            self.client.create_dataset(dataset_ref)
            logger.info(f"Created dataset {self.dataset_id}")

        # Table
        table_ref = bigquery.Table(self.full_table_id, schema=SKILLS_SCHEMA)
        try:
            self.client.get_table(table_ref)
            logger.info(f"Table {self.table_id} already exists")
        except NotFound:
            self.client.create_table(table_ref)
            logger.info(f"Created table {self.table_id}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_extraction(
        self,
        result: ExtractionResult,
        candidate_id: str,
    ) -> int:
        """
        Write all skills from one ExtractionResult to BigQuery.
        Returns number of rows inserted.
        """
        if not result.skills:
            logger.warning(f"No skills to store for candidate {candidate_id}")
            return 0

        extraction_id = str(uuid.uuid4())
        extracted_at = datetime.now(timezone.utc).isoformat()

        rows = [
            {
                "extraction_id": extraction_id,
                "candidate_id":  candidate_id,
                "skill_name":    skill.name,
                "category":      skill.category.value,
                "confidence":    skill.confidence,
                "source_text":   skill.source_text,
                "model_used":    result.model_used,
                "extracted_at":  extracted_at,
            }
            for skill in result.skills
        ]

        errors = self.client.insert_rows_json(self.full_table_id, rows)

        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
            raise RuntimeError(f"Failed to insert {len(errors)} rows: {errors}")

        logger.info(
            f"Stored {len(rows)} skills for candidate {candidate_id} "
            f"[extraction_id={extraction_id}]"
        )
        return len(rows)

    def store_batch(
        self,
        results: list[ExtractionResult],
        candidate_ids: list[str],
    ) -> int:
        """Store multiple extraction results. Lists must be same length."""
        if len(results) != len(candidate_ids):
            raise ValueError("results and candidate_ids must be the same length")

        total = 0
        for result, cid in zip(results, candidate_ids):
            total += self.store_extraction(result, cid)
        return total

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_skills_for_candidate(self, candidate_id: str) -> list[dict]:
        """Fetch all skills for a candidate, ordered by confidence."""
        query = f"""
            SELECT skill_name, category, confidence, extracted_at
            FROM `{self.full_table_id}`
            WHERE candidate_id = @candidate_id
            ORDER BY confidence DESC, extracted_at DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return [dict(row) for row in rows]

    def get_top_skills_by_category(
        self, candidate_id: str, category: SkillCategory
    ) -> list[str]:
        """Return skill names for a candidate filtered by category."""
        query = f"""
            SELECT DISTINCT skill_name
            FROM `{self.full_table_id}`
            WHERE candidate_id = @candidate_id
              AND category = @category
            ORDER BY skill_name
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                bigquery.ScalarQueryParameter("category", "STRING", category.value),
            ]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return [row["skill_name"] for row in rows]