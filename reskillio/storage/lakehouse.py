"""
F14 — Medallion data lakehouse manager.

Three BigQuery datasets:
  reskillio_bronze  — raw ingestion, append-only, immutable
  reskillio_silver  — validated, deduplicated, enriched
  reskillio_gold    — computed aggregates (match scores, rankings, readiness)

Existing `reskillio.*` tables remain unchanged; this layer reads from them
during Silver promotion and writes to new datasets.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from google.api_core.exceptions import NotFound, Conflict
from google.cloud import bigquery
from loguru import logger

from reskillio.models.lakehouse import (
    CandidateReadiness,
    IndustryRanking,
    LayerTableInfo,
    LakehouseStatus,
    MatchScore,
)

# ---------------------------------------------------------------------------
# Dataset names
# ---------------------------------------------------------------------------

BRONZE = "reskillio_bronze"
SILVER = "reskillio_silver"
GOLD   = "reskillio_gold"

# Source dataset (existing)
SOURCE = "reskillio"

# ---------------------------------------------------------------------------
# Bronze schemas  (raw, append-only)
# ---------------------------------------------------------------------------

_BRONZE_RESUME_SCHEMA = [
    bigquery.SchemaField("resume_id",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("filename",        "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("file_size_bytes", "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("page_count",      "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("raw_text",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("char_count",      "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("ingested_at",     "TIMESTAMP", mode="REQUIRED"),
]

_BRONZE_SKILL_EXTRACTIONS_SCHEMA = [
    bigquery.SchemaField("extraction_id", "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",  "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",      "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence",    "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("source_text",   "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("model_used",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("extracted_at",  "TIMESTAMP", mode="REQUIRED"),
]

_BRONZE_JD_RAW_SCHEMA = [
    bigquery.SchemaField("jd_id",      "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("title",      "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("company",    "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("industry",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("raw_text",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("source_url", "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("char_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("ingested_at","TIMESTAMP", mode="REQUIRED"),
]

_BRONZE_JD_EXTRACTIONS_SCHEMA = [
    bigquery.SchemaField("jd_id",      "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("industry",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("seniority",  "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name", "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("requirement","STRING",    mode="REQUIRED"),
    bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
]

# ---------------------------------------------------------------------------
# Silver schemas  (validated, deduplicated)
# ---------------------------------------------------------------------------

_SILVER_CANDIDATE_SKILLS_SCHEMA = [
    bigquery.SchemaField("candidate_id",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",         "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",           "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence_avg",     "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("frequency",          "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("validation_status",  "STRING",    mode="REQUIRED"),  # VALID / LOW_CONFIDENCE / UNKNOWN
    bigquery.SchemaField("first_seen",         "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("last_seen",          "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("promoted_at",        "TIMESTAMP", mode="REQUIRED"),
]

_SILVER_JD_PROFILES_SCHEMA = [
    bigquery.SchemaField("jd_id",                 "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("title",                 "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("company",               "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("industry",              "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("seniority",             "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("required_skill_count",  "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("preferred_skill_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("promoted_at",           "TIMESTAMP", mode="REQUIRED"),
]

_SILVER_JD_SKILLS_SCHEMA = [
    bigquery.SchemaField("jd_id",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("industry",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("seniority",    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_name",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("category",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("confidence",   "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("requirement",  "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("promoted_at",  "TIMESTAMP", mode="REQUIRED"),
]

_SILVER_DRIFT_METRICS_SCHEMA = [
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
    bigquery.SchemaField("promoted_at",        "TIMESTAMP", mode="REQUIRED"),
]

# ---------------------------------------------------------------------------
# Gold schemas  (computed)
# ---------------------------------------------------------------------------

_GOLD_MATCH_SCORES_SCHEMA = [
    bigquery.SchemaField("match_id",            "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("jd_id",               "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("required_coverage",   "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("preferred_coverage",  "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("overall_match_score", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("matched_skill_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("missing_skill_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("extra_skill_count",   "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("computed_at",         "TIMESTAMP", mode="REQUIRED"),
]

_GOLD_INDUSTRY_RANKINGS_SCHEMA = [
    bigquery.SchemaField("ranking_id",     "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("candidate_id",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("industry",       "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skills_matched", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("skills_demanded","INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("skill_coverage", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("readiness_tier", "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("computed_at",    "TIMESTAMP", mode="REQUIRED"),
]

_GOLD_CANDIDATE_READINESS_SCHEMA = [
    bigquery.SchemaField("candidate_id",                "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("skill_count",                 "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("high_confidence_skill_count", "INTEGER",   mode="REQUIRED"),
    bigquery.SchemaField("avg_confidence",              "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("best_industry",               "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("best_industry_coverage",      "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("avg_match_score",             "FLOAT64",   mode="NULLABLE"),
    bigquery.SchemaField("readiness_index",             "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("readiness_tier",              "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("computed_at",                 "TIMESTAMP", mode="REQUIRED"),
]

# (dataset, table, schema)
_ALL_TABLES: list[tuple[str, str, list]] = [
    # Bronze
    (BRONZE, "resume_raw",          _BRONZE_RESUME_SCHEMA),
    (BRONZE, "skill_extractions",   _BRONZE_SKILL_EXTRACTIONS_SCHEMA),
    (BRONZE, "jd_raw",              _BRONZE_JD_RAW_SCHEMA),
    (BRONZE, "jd_extractions",      _BRONZE_JD_EXTRACTIONS_SCHEMA),
    # Silver
    (SILVER, "candidate_skills",    _SILVER_CANDIDATE_SKILLS_SCHEMA),
    (SILVER, "jd_profiles",         _SILVER_JD_PROFILES_SCHEMA),
    (SILVER, "jd_skills",           _SILVER_JD_SKILLS_SCHEMA),
    (SILVER, "drift_metrics",       _SILVER_DRIFT_METRICS_SCHEMA),
    # Gold
    (GOLD,   "match_scores",        _GOLD_MATCH_SCORES_SCHEMA),
    (GOLD,   "industry_rankings",   _GOLD_INDUSTRY_RANKINGS_SCHEMA),
    (GOLD,   "candidate_readiness", _GOLD_CANDIDATE_READINESS_SCHEMA),
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


# ---------------------------------------------------------------------------
# LakehouseManager
# ---------------------------------------------------------------------------

class LakehouseManager:
    """
    Manages all three medallion layers in BigQuery.

    Bronze: ingested via ingest_resume_raw() / ingest_jd_raw()
    Silver: promoted via promote_candidate() / promote_jd()
    Gold:   computed via compute_match_scores() / compute_industry_rankings()
            / compute_readiness()
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._client: Optional[bigquery.Client] = None

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            _apply_credentials()
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def _fqn(self, dataset: str, table: str) -> str:
        return f"{self.project_id}.{dataset}.{table}"

    def _src(self, table: str) -> str:
        return self._fqn(SOURCE, table)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def ensure_all(self) -> None:
        """Create all three datasets + tables. Safe to call repeatedly."""
        for dataset_id in (BRONZE, SILVER, GOLD):
            ds_ref = bigquery.Dataset(f"{self.project_id}.{dataset_id}")
            ds_ref.location = "US"
            try:
                self.client.create_dataset(ds_ref)
                logger.info(f"[lakehouse] Created dataset {dataset_id}")
            except Conflict:
                logger.info(f"[lakehouse] Dataset {dataset_id} already exists")

        for dataset_id, table_id, schema in _ALL_TABLES:
            full = self._fqn(dataset_id, table_id)
            table_ref = bigquery.Table(full, schema=schema)
            try:
                self.client.create_table(table_ref)
                logger.info(f"[lakehouse] Created table {full}")
            except Conflict:
                logger.info(f"[lakehouse] Table {full} already exists")

    # ------------------------------------------------------------------
    # Bronze ingestion helpers
    # ------------------------------------------------------------------

    def ingest_resume_raw(
        self,
        candidate_id:    str,
        raw_text:        str,
        filename:        Optional[str] = None,
        file_size_bytes: Optional[int] = None,
        page_count:      Optional[int] = None,
    ) -> str:
        """Write raw resume text to bronze.resume_raw. Returns resume_id."""
        resume_id = str(uuid.uuid4())
        row = {
            "resume_id":       resume_id,
            "candidate_id":    candidate_id,
            "filename":        filename,
            "file_size_bytes": file_size_bytes,
            "page_count":      page_count,
            "raw_text":        raw_text,
            "char_count":      len(raw_text),
            "ingested_at":     datetime.now(timezone.utc).isoformat(),
        }
        errors = self.client.insert_rows_json(self._fqn(BRONZE, "resume_raw"), [row])
        if errors:
            raise RuntimeError(f"Bronze resume insert failed: {errors}")
        logger.info(f"[lakehouse] Bronze ← resume_raw candidate={candidate_id}")
        return resume_id

    def ingest_jd_raw(
        self,
        jd_id:      str,
        industry:   str,
        raw_text:   str,
        title:      Optional[str] = None,
        company:    Optional[str] = None,
        source_url: Optional[str] = None,
    ) -> None:
        """Write raw JD text to bronze.jd_raw."""
        row = {
            "jd_id":       jd_id,
            "title":       title,
            "company":     company,
            "industry":    industry,
            "raw_text":    raw_text,
            "source_url":  source_url,
            "char_count":  len(raw_text),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }
        errors = self.client.insert_rows_json(self._fqn(BRONZE, "jd_raw"), [row])
        if errors:
            raise RuntimeError(f"Bronze JD insert failed: {errors}")
        logger.info(f"[lakehouse] Bronze ← jd_raw jd_id={jd_id}")

    # ------------------------------------------------------------------
    # Silver promotion
    # ------------------------------------------------------------------

    def promote_candidate(self, candidate_id: str) -> int:
        """
        MERGE skill_extractions → silver.candidate_skills for one candidate.

        Validation rules applied during promotion:
          - confidence_avg < 0.30  → LOW_CONFIDENCE
          - category = 'UNKNOWN'   → UNKNOWN
          - otherwise              → VALID
        """
        promoted_at = datetime.now(timezone.utc).isoformat()
        silver_tbl  = self._fqn(SILVER, "candidate_skills")
        source_tbl  = self._src("skill_extractions")

        sql = f"""
        MERGE `{silver_tbl}` T
        USING (
          SELECT
            candidate_id,
            skill_name,
            category,
            AVG(confidence)       AS confidence_avg,
            COUNT(*)              AS frequency,
            MIN(extracted_at)     AS first_seen,
            MAX(extracted_at)     AS last_seen,
            CASE
              WHEN AVG(confidence) < 0.30 THEN 'LOW_CONFIDENCE'
              WHEN MAX(category) = 'UNKNOWN' THEN 'UNKNOWN'
              ELSE 'VALID'
            END AS validation_status,
            TIMESTAMP('{promoted_at}') AS promoted_at
          FROM `{source_tbl}`
          WHERE candidate_id = @candidate_id
          GROUP BY candidate_id, skill_name, category
        ) S
        ON  T.candidate_id = S.candidate_id
        AND T.skill_name   = S.skill_name

        WHEN MATCHED THEN UPDATE SET
          T.confidence_avg    = S.confidence_avg,
          T.frequency         = S.frequency,
          T.last_seen         = S.last_seen,
          T.validation_status = S.validation_status,
          T.promoted_at       = S.promoted_at

        WHEN NOT MATCHED THEN INSERT (
          candidate_id, skill_name, category, confidence_avg,
          frequency, validation_status, first_seen, last_seen, promoted_at
        ) VALUES (
          S.candidate_id, S.skill_name, S.category, S.confidence_avg,
          S.frequency, S.validation_status, S.first_seen, S.last_seen, S.promoted_at
        )
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        job = self.client.query(sql, job_config=cfg)
        job.result()
        affected = job.num_dml_affected_rows or 0
        logger.info(f"[lakehouse] Silver ← candidate_skills candidate={candidate_id} ({affected} rows)")
        return affected

    def promote_jd(self, jd_id: str) -> int:
        """MERGE jd_profiles + jd_skills → silver.jd_profiles / silver.jd_skills."""
        promoted_at  = datetime.now(timezone.utc).isoformat()
        silver_prof  = self._fqn(SILVER, "jd_profiles")
        silver_skills = self._fqn(SILVER, "jd_skills")
        src_prof     = self._src("jd_profiles")
        src_skills   = self._src("jd_skills")

        # Promote jd_profiles
        sql_prof = f"""
        MERGE `{silver_prof}` T
        USING (
          SELECT *, TIMESTAMP('{promoted_at}') AS promoted_at
          FROM `{src_prof}`
          WHERE jd_id = @jd_id
        ) S
        ON T.jd_id = S.jd_id
        WHEN MATCHED THEN UPDATE SET
          T.title                = S.title,
          T.company              = S.company,
          T.industry             = S.industry,
          T.seniority            = S.seniority,
          T.required_skill_count = S.required_skill_count,
          T.preferred_skill_count= S.preferred_skill_count,
          T.promoted_at          = S.promoted_at
        WHEN NOT MATCHED THEN INSERT (
          jd_id, title, company, industry, seniority,
          required_skill_count, preferred_skill_count, promoted_at
        ) VALUES (
          S.jd_id, S.title, S.company, S.industry, S.seniority,
          S.required_skill_count, S.preferred_skill_count, S.promoted_at
        )
        """
        # Promote jd_skills
        sql_skills = f"""
        MERGE `{silver_skills}` T
        USING (
          SELECT *, TIMESTAMP('{promoted_at}') AS promoted_at
          FROM `{src_skills}`
          WHERE jd_id = @jd_id
        ) S
        ON  T.jd_id      = S.jd_id
        AND T.skill_name = S.skill_name
        AND T.requirement= S.requirement
        WHEN MATCHED THEN UPDATE SET
          T.confidence  = S.confidence,
          T.promoted_at = S.promoted_at
        WHEN NOT MATCHED THEN INSERT (
          jd_id, industry, seniority, skill_name, category,
          confidence, requirement, promoted_at
        ) VALUES (
          S.jd_id, S.industry, S.seniority, S.skill_name, S.category,
          S.confidence, S.requirement, S.promoted_at
        )
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ScalarQueryParameter("jd_id", "STRING", jd_id)]
        )
        total = 0
        for sql in (sql_prof, sql_skills):
            job = self.client.query(sql, job_config=cfg)
            job.result()
            total += job.num_dml_affected_rows or 0

        logger.info(f"[lakehouse] Silver ← jd jd_id={jd_id} ({total} rows)")
        return total

    # ------------------------------------------------------------------
    # Gold — Match Scores
    # ------------------------------------------------------------------

    def compute_match_scores(
        self,
        candidate_id: str,
        jd_ids:       list[str],
    ) -> list[MatchScore]:
        """
        Compute match score for each (candidate_id × jd_id) pair.

        Weights: required coverage 70 %, preferred coverage 30 %.
        Writes results to gold.match_scores (MERGE, idempotent).
        """
        results: list[MatchScore] = []
        gold_tbl = self._fqn(GOLD, "match_scores")
        silver_c = self._fqn(SILVER, "candidate_skills")
        silver_j = self._fqn(SILVER, "jd_skills")

        for jd_id in jd_ids:
            match_id    = str(uuid.uuid4())
            computed_at = datetime.now(timezone.utc).isoformat()

            compute_sql = f"""
            WITH candidate AS (
              SELECT LOWER(skill_name) AS skill_name
              FROM `{silver_c}`
              WHERE candidate_id = @candidate_id
                AND validation_status = 'VALID'
            ),
            jd_req AS (
              SELECT LOWER(skill_name) AS skill_name
              FROM `{silver_j}`
              WHERE jd_id = @jd_id AND LOWER(requirement) = 'required'
            ),
            jd_pref AS (
              SELECT LOWER(skill_name) AS skill_name
              FROM `{silver_j}`
              WHERE jd_id = @jd_id AND LOWER(requirement) = 'preferred'
            ),
            req_stats AS (
              SELECT
                COUNT(r.skill_name)                        AS required_total,
                COUNTIF(c.skill_name IS NOT NULL)          AS required_matched,
                COUNT(r.skill_name) - COUNTIF(c.skill_name IS NOT NULL) AS missing_count
              FROM jd_req r LEFT JOIN candidate c USING (skill_name)
            ),
            pref_stats AS (
              SELECT
                COUNT(p.skill_name)               AS preferred_total,
                COUNTIF(c.skill_name IS NOT NULL) AS preferred_matched
              FROM jd_pref p LEFT JOIN candidate c USING (skill_name)
            ),
            extra AS (
              SELECT COUNT(*) AS extra_count
              FROM candidate c
              WHERE c.skill_name NOT IN (SELECT skill_name FROM jd_req)
                AND c.skill_name NOT IN (SELECT skill_name FROM jd_pref)
            )
            SELECT
              '{match_id}'          AS match_id,
              @candidate_id         AS candidate_id,
              @jd_id                AS jd_id,
              SAFE_DIVIDE(rs.required_matched,  rs.required_total)  AS required_coverage,
              SAFE_DIVIDE(ps.preferred_matched, ps.preferred_total) AS preferred_coverage,
              LEAST(100,
                (COALESCE(SAFE_DIVIDE(rs.required_matched,  rs.required_total),  0) * 0.7 +
                 COALESCE(SAFE_DIVIDE(ps.preferred_matched, ps.preferred_total), 0) * 0.3)
                * 100
              )                     AS overall_match_score,
              rs.required_matched   AS matched_skill_count,
              rs.missing_count      AS missing_skill_count,
              ex.extra_count        AS extra_skill_count,
              TIMESTAMP('{computed_at}') AS computed_at
            FROM req_stats rs, pref_stats ps, extra ex
            """

            merge_sql = f"""
            MERGE `{gold_tbl}` T
            USING ({compute_sql}) S
            ON T.candidate_id = S.candidate_id AND T.jd_id = S.jd_id
            WHEN MATCHED THEN UPDATE SET
              T.match_id            = S.match_id,
              T.required_coverage   = S.required_coverage,
              T.preferred_coverage  = S.preferred_coverage,
              T.overall_match_score = S.overall_match_score,
              T.matched_skill_count = S.matched_skill_count,
              T.missing_skill_count = S.missing_skill_count,
              T.extra_skill_count   = S.extra_skill_count,
              T.computed_at         = S.computed_at
            WHEN NOT MATCHED THEN INSERT (
              match_id, candidate_id, jd_id,
              required_coverage, preferred_coverage, overall_match_score,
              matched_skill_count, missing_skill_count, extra_skill_count,
              computed_at
            ) VALUES (
              S.match_id, S.candidate_id, S.jd_id,
              S.required_coverage, S.preferred_coverage, S.overall_match_score,
              S.matched_skill_count, S.missing_skill_count, S.extra_skill_count,
              S.computed_at
            )
            """
            cfg = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                    bigquery.ScalarQueryParameter("jd_id",        "STRING", jd_id),
                ]
            )
            job = self.client.query(merge_sql, job_config=cfg)
            job.result()

            # Read back the stored row
            row = self._fetch_match_score(candidate_id, jd_id)
            if row:
                results.append(row)
            logger.info(f"[lakehouse] Gold ← match_score candidate={candidate_id} jd={jd_id}")

        return results

    def _fetch_match_score(self, candidate_id: str, jd_id: str) -> Optional[MatchScore]:
        sql = f"""
        SELECT * FROM `{self._fqn(GOLD, 'match_scores')}`
        WHERE candidate_id = @candidate_id AND jd_id = @jd_id
        LIMIT 1
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id),
                bigquery.ScalarQueryParameter("jd_id",        "STRING", jd_id),
            ]
        )
        rows = list(self.client.query(sql, job_config=cfg).result())
        if not rows:
            return None
        r = rows[0]
        return MatchScore(
            match_id=r["match_id"],
            candidate_id=r["candidate_id"],
            jd_id=r["jd_id"],
            required_coverage=r["required_coverage"] or 0.0,
            preferred_coverage=r["preferred_coverage"] or 0.0,
            overall_match_score=r["overall_match_score"] or 0.0,
            matched_skill_count=r["matched_skill_count"],
            missing_skill_count=r["missing_skill_count"],
            extra_skill_count=r["extra_skill_count"],
            computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
        )

    # ------------------------------------------------------------------
    # Gold — Industry Rankings
    # ------------------------------------------------------------------

    def compute_industry_rankings(self, candidate_id: str) -> list[IndustryRanking]:
        """
        Compute skill coverage per industry for one candidate.

        Tier rules:
          ≥ 70 % coverage → TIER_1
          ≥ 40 %          → TIER_2
          < 40 %          → TIER_3
        """
        gold_tbl = self._fqn(GOLD,   "industry_rankings")
        silver_c = self._fqn(SILVER, "candidate_skills")
        silver_j = self._fqn(SILVER, "jd_skills")

        computed_at = datetime.now(timezone.utc).isoformat()

        # Delete old rankings for this candidate then insert fresh
        del_sql = f"""
        DELETE FROM `{gold_tbl}` WHERE candidate_id = @candidate_id
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        self.client.query(del_sql, job_config=cfg).result()

        insert_sql = f"""
        INSERT INTO `{gold_tbl}`
        WITH industry_skills AS (
          SELECT industry, LOWER(skill_name) AS skill_name
          FROM `{silver_j}`
          GROUP BY industry, skill_name
        ),
        candidate_skills AS (
          SELECT LOWER(skill_name) AS skill_name
          FROM `{silver_c}`
          WHERE candidate_id = @candidate_id AND validation_status = 'VALID'
        ),
        per_industry AS (
          SELECT
            GENERATE_UUID()              AS ranking_id,
            @candidate_id                AS candidate_id,
            i.industry,
            COUNT(DISTINCT i.skill_name) AS skills_demanded,
            COUNT(DISTINCT c.skill_name) AS skills_matched,
            SAFE_DIVIDE(
              COUNT(DISTINCT c.skill_name),
              COUNT(DISTINCT i.skill_name)
            )                            AS skill_coverage,
            TIMESTAMP('{computed_at}')   AS computed_at
          FROM industry_skills i
          LEFT JOIN candidate_skills c USING (skill_name)
          GROUP BY i.industry
        )
        SELECT
          ranking_id, candidate_id, industry,
          skills_matched, skills_demanded, skill_coverage,
          CASE
            WHEN skill_coverage >= 0.70 THEN 'TIER_1'
            WHEN skill_coverage >= 0.40 THEN 'TIER_2'
            ELSE 'TIER_3'
          END AS readiness_tier,
          computed_at
        FROM per_industry
        ORDER BY skill_coverage DESC
        """
        self.client.query(insert_sql, job_config=cfg).result()

        # Read back
        select_sql = f"""
        SELECT * FROM `{gold_tbl}`
        WHERE candidate_id = @candidate_id
        ORDER BY skill_coverage DESC
        """
        rows = list(self.client.query(select_sql, job_config=cfg).result())
        results = [
            IndustryRanking(
                ranking_id=r["ranking_id"],
                candidate_id=r["candidate_id"],
                industry=r["industry"],
                skills_matched=r["skills_matched"],
                skills_demanded=r["skills_demanded"],
                skill_coverage=r["skill_coverage"] or 0.0,
                readiness_tier=r["readiness_tier"],
                computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
            )
            for r in rows
        ]
        logger.info(f"[lakehouse] Gold ← industry_rankings candidate={candidate_id} ({len(results)} industries)")
        return results

    # ------------------------------------------------------------------
    # Gold — Candidate Readiness Index
    # ------------------------------------------------------------------

    def compute_readiness(self, candidate_id: str) -> Optional[CandidateReadiness]:
        """
        Composite readiness index (0–100):
          40 % avg match score across all JDs
          30 % best industry coverage
          20 % avg confidence
          10 % skill breadth (capped at 30 skills = 100 %)

        Tier: READY ≥ 70, DEVELOPING ≥ 40, EMERGING < 40.
        """
        gold_ready  = self._fqn(GOLD,   "candidate_readiness")
        gold_match  = self._fqn(GOLD,   "match_scores")
        gold_rank   = self._fqn(GOLD,   "industry_rankings")
        silver_c    = self._fqn(SILVER, "candidate_skills")
        computed_at = datetime.now(timezone.utc).isoformat()

        sql = f"""
        WITH skill_stats AS (
          SELECT
            COUNT(*)                   AS skill_count,
            COUNTIF(confidence_avg >= 0.70) AS high_conf_count,
            COALESCE(AVG(confidence_avg), 0) AS avg_confidence
          FROM `{silver_c}`
          WHERE candidate_id = @candidate_id AND validation_status = 'VALID'
        ),
        match_stats AS (
          SELECT COALESCE(AVG(overall_match_score), 0) AS avg_match_score
          FROM `{gold_match}`
          WHERE candidate_id = @candidate_id
        ),
        best_ind AS (
          SELECT industry, skill_coverage
          FROM `{gold_rank}`
          WHERE candidate_id = @candidate_id
          ORDER BY skill_coverage DESC
          LIMIT 1
        )
        SELECT
          @candidate_id AS candidate_id,
          ss.skill_count,
          ss.high_conf_count                    AS high_confidence_skill_count,
          ss.avg_confidence,
          bi.industry                           AS best_industry,
          bi.skill_coverage                     AS best_industry_coverage,
          ms.avg_match_score,
          LEAST(100,
            ms.avg_match_score              * 0.40 +
            COALESCE(bi.skill_coverage, 0) * 100 * 0.30 +
            ss.avg_confidence               * 100 * 0.20 +
            LEAST(ss.skill_count, 30) / 30.0 * 100 * 0.10
          ) AS readiness_index,
          TIMESTAMP('{computed_at}') AS computed_at
        FROM skill_stats ss, match_stats ms
        LEFT JOIN best_ind bi ON TRUE
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = list(self.client.query(sql, job_config=cfg).result())
        if not rows:
            return None

        r = rows[0]
        idx   = r["readiness_index"] or 0.0
        tier  = "READY" if idx >= 70 else ("DEVELOPING" if idx >= 40 else "EMERGING")
        ready = CandidateReadiness(
            candidate_id=candidate_id,
            skill_count=r["skill_count"],
            high_confidence_skill_count=r["high_confidence_skill_count"],
            avg_confidence=r["avg_confidence"] or 0.0,
            best_industry=r["best_industry"],
            best_industry_coverage=r["best_industry_coverage"],
            avg_match_score=r["avg_match_score"],
            readiness_index=idx,
            readiness_tier=tier,
            computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
        )

        # Upsert into gold
        del_sql = f"DELETE FROM `{gold_ready}` WHERE candidate_id = @candidate_id"
        self.client.query(del_sql, job_config=cfg).result()

        row = {
            "candidate_id":                ready.candidate_id,
            "skill_count":                 ready.skill_count,
            "high_confidence_skill_count": ready.high_confidence_skill_count,
            "avg_confidence":              ready.avg_confidence,
            "best_industry":               ready.best_industry,
            "best_industry_coverage":      ready.best_industry_coverage,
            "avg_match_score":             ready.avg_match_score,
            "readiness_index":             ready.readiness_index,
            "readiness_tier":              ready.readiness_tier,
            "computed_at":                 ready.computed_at.isoformat(),
        }
        errors = self.client.insert_rows_json(gold_ready, [row])
        if errors:
            logger.warning(f"[lakehouse] Gold readiness insert warning: {errors}")

        logger.info(
            f"[lakehouse] Gold ← candidate_readiness candidate={candidate_id} "
            f"index={idx:.1f} tier={tier}"
        )
        return ready

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> LakehouseStatus:
        """Return row counts for every table in all three layers."""
        layers: dict[str, list[LayerTableInfo]] = {BRONZE: [], SILVER: [], GOLD: []}

        for dataset_id, table_id, _ in _ALL_TABLES:
            fqn = self._fqn(dataset_id, table_id)
            try:
                row = list(
                    self.client.query(f"SELECT COUNT(*) AS n FROM `{fqn}`").result()
                )[0]
                count = row["n"]
            except Exception:
                count = -1
            layers[dataset_id].append(LayerTableInfo(table=table_id, row_count=count))

        return LakehouseStatus(
            project_id=self.project_id,
            bronze=layers[BRONZE],
            silver=layers[SILVER],
            gold=layers[GOLD],
        )

    def get_match_scores(self, candidate_id: str) -> list[MatchScore]:
        sql = f"""
        SELECT * FROM `{self._fqn(GOLD, 'match_scores')}`
        WHERE candidate_id = @candidate_id
        ORDER BY overall_match_score DESC
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = list(self.client.query(sql, job_config=cfg).result())
        return [
            MatchScore(
                match_id=r["match_id"],
                candidate_id=r["candidate_id"],
                jd_id=r["jd_id"],
                required_coverage=r["required_coverage"] or 0.0,
                preferred_coverage=r["preferred_coverage"] or 0.0,
                overall_match_score=r["overall_match_score"] or 0.0,
                matched_skill_count=r["matched_skill_count"],
                missing_skill_count=r["missing_skill_count"],
                extra_skill_count=r["extra_skill_count"],
                computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
            )
            for r in rows
        ]

    def get_industry_rankings(self, candidate_id: str) -> list[IndustryRanking]:
        sql = f"""
        SELECT * FROM `{self._fqn(GOLD, 'industry_rankings')}`
        WHERE candidate_id = @candidate_id
        ORDER BY skill_coverage DESC
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = list(self.client.query(sql, job_config=cfg).result())
        return [
            IndustryRanking(
                ranking_id=r["ranking_id"],
                candidate_id=r["candidate_id"],
                industry=r["industry"],
                skills_matched=r["skills_matched"],
                skills_demanded=r["skills_demanded"],
                skill_coverage=r["skill_coverage"] or 0.0,
                readiness_tier=r["readiness_tier"],
                computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
            )
            for r in rows
        ]

    def get_readiness(self, candidate_id: str) -> Optional[CandidateReadiness]:
        sql = f"""
        SELECT * FROM `{self._fqn(GOLD, 'candidate_readiness')}`
        WHERE candidate_id = @candidate_id
        ORDER BY computed_at DESC
        LIMIT 1
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("candidate_id", "STRING", candidate_id)
            ]
        )
        rows = list(self.client.query(sql, job_config=cfg).result())
        if not rows:
            return None
        r = rows[0]
        return CandidateReadiness(
            candidate_id=r["candidate_id"],
            skill_count=r["skill_count"],
            high_confidence_skill_count=r["high_confidence_skill_count"],
            avg_confidence=r["avg_confidence"] or 0.0,
            best_industry=r["best_industry"],
            best_industry_coverage=r["best_industry_coverage"],
            avg_match_score=r["avg_match_score"],
            readiness_index=r["readiness_index"] or 0.0,
            readiness_tier=r["readiness_tier"],
            computed_at=r["computed_at"].replace(tzinfo=timezone.utc),
        )
