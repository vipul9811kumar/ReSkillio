"""
BigQuery storage for job descriptions and derived industry profiles.

Tables
------
jd_profiles   — one row per JD (header metadata)
jd_skills     — one row per skill per JD
industry_profiles — aggregated skill demand per industry (derived via query)
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from reskillio.models.jd import Industry, JDExtractionResult, JDSkill, RequirementLevel

logger = logging.getLogger(__name__)


def _apply_credentials() -> None:
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return
    try:
        from config.settings import settings
        if settings.google_application_credentials:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
    except Exception:
        pass


JD_PROFILES_SCHEMA = [
    bigquery.SchemaField("jd_id",                 "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("title",                 "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("company",               "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("industry",              "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("seniority",             "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("source_url",            "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("required_skill_count",  "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("preferred_skill_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("created_at",            "TIMESTAMP", mode="REQUIRED"),
]

JD_SKILLS_SCHEMA = [
    bigquery.SchemaField("jd_id",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("industry",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("seniority",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",  "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence",  "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("requirement", "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("created_at",  "TIMESTAMP", mode="REQUIRED"),
]


class JDStore:
    """Read/write for jd_profiles, jd_skills, and industry_profiles."""

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "reskillio",
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self._client: Optional[bigquery.Client] = None

        self.jd_profiles_table   = f"{project_id}.{dataset_id}.jd_profiles"
        self.jd_skills_table     = f"{project_id}.{dataset_id}.jd_skills"
        self.industry_profiles_table = f"{project_id}.{dataset_id}.industry_profiles"

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            _apply_credentials()
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_tables(self) -> None:
        """Create jd_profiles and jd_skills tables if they don't exist."""
        for table_id, schema in [
            (self.jd_profiles_table, JD_PROFILES_SCHEMA),
            (self.jd_skills_table,   JD_SKILLS_SCHEMA),
        ]:
            ref = bigquery.Table(table_id, schema=schema)
            try:
                self.client.get_table(ref)
                logger.info(f"Table {table_id.split('.')[-1]} already exists")
            except NotFound:
                self.client.create_table(ref)
                logger.info(f"Created table {table_id}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store_jd(self, result: JDExtractionResult) -> str:
        """
        Persist a JD extraction result to jd_profiles + jd_skills.
        Returns the jd_id.
        """
        created_at = datetime.now(timezone.utc).isoformat()

        # jd_profiles row
        profile_row = [{
            "jd_id":                 result.jd_id,
            "title":                 result.title,
            "company":               result.company,
            "industry":              result.industry.value,
            "seniority":             result.seniority.value,
            "source_url":            result.source_url,
            "required_skill_count":  len(result.required_skills),
            "preferred_skill_count": len(result.preferred_skills),
            "created_at":            created_at,
        }]
        errors = self.client.insert_rows_json(self.jd_profiles_table, profile_row)
        if errors:
            raise RuntimeError(f"jd_profiles insert error: {errors}")

        # jd_skills rows
        all_skills: list[JDSkill] = result.required_skills + result.preferred_skills
        if all_skills:
            skill_rows = [
                {
                    "jd_id":       result.jd_id,
                    "industry":    result.industry.value,
                    "seniority":   result.seniority.value,
                    "skill_name":  skill.name,
                    "category":    skill.category.value,
                    "confidence":  skill.confidence,
                    "requirement": skill.requirement.value,
                    "created_at":  created_at,
                }
                for skill in all_skills
            ]
            errors = self.client.insert_rows_json(self.jd_skills_table, skill_rows)
            if errors:
                raise RuntimeError(f"jd_skills insert error: {errors}")

        logger.info(
            f"Stored JD jd_id={result.jd_id} "
            f"({len(result.required_skills)} required, "
            f"{len(result.preferred_skills)} preferred skills)"
        )
        return result.jd_id

    # ------------------------------------------------------------------
    # Derive industry profiles
    # ------------------------------------------------------------------

    def refresh_industry_profiles(self) -> None:
        """
        Rebuild the industry_profiles table from jd_skills.

        Each row: (industry, skill_name, category, frequency, demand_weight, avg_confidence)
        demand_weight = skill frequency / total skills in that industry (0–1)

        Called after seeding or whenever new JDs are added.
        """
        query = f"""
        CREATE OR REPLACE TABLE `{self.industry_profiles_table}` AS
        WITH base AS (
            SELECT
                industry,
                skill_name,
                category,
                COUNT(*)        AS frequency,
                AVG(confidence) AS avg_confidence,
                COUNTIF(requirement = 'required')  AS required_count,
                COUNTIF(requirement = 'preferred') AS preferred_count
            FROM `{self.jd_skills_table}`
            GROUP BY industry, skill_name, category
        ),
        totals AS (
            SELECT industry, SUM(frequency) AS total_skills
            FROM base
            GROUP BY industry
        )
        SELECT
            b.industry,
            b.skill_name,
            b.category,
            b.frequency,
            b.avg_confidence,
            b.required_count,
            b.preferred_count,
            ROUND(b.frequency / t.total_skills, 6) AS demand_weight
        FROM base b
        JOIN totals t USING (industry)
        ORDER BY b.industry, b.frequency DESC
        """
        job = self.client.query(query)
        job.result()
        logger.info("industry_profiles table refreshed")

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_jd(self, jd_id: str) -> dict | None:
        """Fetch a JD profile + its skills by jd_id."""
        query = f"""
            SELECT p.*, s.skill_name, s.category, s.requirement, s.confidence
            FROM `{self.jd_profiles_table}` p
            JOIN `{self.jd_skills_table}` s USING (jd_id)
            WHERE p.jd_id = @jd_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("jd_id", "STRING", jd_id)]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        return [dict(r) for r in rows] if rows else None

    def get_industry_profile(self, industry: Industry) -> list[dict]:
        """Return skill demand profile for a single industry, ordered by demand_weight."""
        query = f"""
            SELECT skill_name, category, frequency, demand_weight, avg_confidence
            FROM `{self.industry_profiles_table}`
            WHERE industry = @industry
            ORDER BY demand_weight DESC
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("industry", "STRING", industry.value)
            ]
        )
        rows = self.client.query(query, job_config=job_config).result()
        return [dict(r) for r in rows]

    def get_all_industry_profiles(self) -> dict[str, list[dict]]:
        """Return all industry profiles as {industry_name: [skill_rows]}."""
        query = f"""
            SELECT industry, skill_name, category, frequency, demand_weight, avg_confidence
            FROM `{self.industry_profiles_table}`
            ORDER BY industry, demand_weight DESC
        """
        rows = self.client.query(query).result()
        profiles: dict[str, list[dict]] = {}
        for row in rows:
            profiles.setdefault(row["industry"], []).append(dict(row))
        return profiles
