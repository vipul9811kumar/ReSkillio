"""BigQuery persistence for IntakeProfile."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger

from reskillio.models.intake import IntakeProfile

_TABLE = "reskillio.candidate_intake_profiles"

_SCHEMA = [
    ("candidate_id",             "STRING",    "REQUIRED"),
    ("financial_runway",         "STRING",    "NULLABLE"),
    ("urgency_score",            "FLOAT64",   "NULLABLE"),
    ("geographic_flexibility",   "STRING",    "NULLABLE"),
    ("target_locations",         "STRING",    "NULLABLE"),  # JSON array
    ("work_identity",            "STRING",    "NULLABLE"),
    ("team_preference",          "STRING",    "NULLABLE"),
    ("company_stage_preference", "STRING",    "NULLABLE"),  # JSON array
    ("engagement_format",        "STRING",    "NULLABLE"),
    ("open_to_fractional",       "BOOL",      "NULLABLE"),
    ("loved_aspects",            "STRING",    "NULLABLE"),
    ("want_next",                "STRING",    "NULLABLE"),
    ("created_at",               "TIMESTAMP", "NULLABLE"),
    ("completed_at",             "TIMESTAMP", "NULLABLE"),
]


class IntakeStore:
    def __init__(self, project_id: str):
        self.project_id = project_id
        self._client = None

    def _bq(self):
        if self._client is None:
            from google.cloud import bigquery
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def ensure_table(self) -> None:
        from google.cloud import bigquery
        from google.api_core.exceptions import NotFound

        bq = self._bq()
        try:
            bq.get_table(_TABLE)
            return
        except NotFound:
            pass

        schema = [
            bigquery.SchemaField(name, ftype, mode=mode)
            for name, ftype, mode in _SCHEMA
        ]
        table = bigquery.Table(f"{self.project_id}.{_TABLE.split('.', 1)[1]}", schema=schema)
        bq.create_table(table)
        logger.info(f"[intake_store] Created table {_TABLE}")

    def upsert_profile(self, profile: IntakeProfile) -> None:
        import json
        bq = self._bq()

        row = {
            "candidate_id":             profile.candidate_id,
            "financial_runway":         profile.financial_runway,
            "urgency_score":            profile.urgency_score,
            "geographic_flexibility":   profile.geographic_flexibility,
            "target_locations":         json.dumps(profile.target_locations),
            "work_identity":            profile.work_identity,
            "team_preference":          profile.team_preference,
            "company_stage_preference": json.dumps(profile.company_stage_preference),
            "engagement_format":        profile.engagement_format,
            "open_to_fractional":       profile.open_to_fractional,
            "loved_aspects":            profile.loved_aspects,
            "want_next":                profile.want_next,
            "created_at":               profile.created_at.isoformat(),
            "completed_at":             profile.completed_at.isoformat() if profile.completed_at else None,
        }

        errors = bq.insert_rows_json(_TABLE, [row])
        if errors:
            logger.error(f"[intake_store] BQ insert errors: {errors}")
        else:
            logger.info(f"[intake_store] Upserted profile for {profile.candidate_id}")

    def get_profile(self, candidate_id: str) -> Optional[IntakeProfile]:
        import json
        bq = self._bq()

        query = f"""
            SELECT * FROM `{_TABLE}`
            WHERE candidate_id = @cid
            ORDER BY created_at DESC
            LIMIT 1
        """
        job_config = __import__("google.cloud.bigquery", fromlist=["QueryJobConfig"]).QueryJobConfig(
            query_parameters=[
                __import__("google.cloud.bigquery", fromlist=["ScalarQueryParameter"]).ScalarQueryParameter(
                    "cid", "STRING", candidate_id
                )
            ]
        )
        rows = list(bq.query(query, job_config=job_config).result())
        if not rows:
            return None

        r = rows[0]
        return IntakeProfile(
            candidate_id=r["candidate_id"],
            financial_runway=r.get("financial_runway"),
            urgency_score=float(r.get("urgency_score") or 0.5),
            geographic_flexibility=r.get("geographic_flexibility"),
            target_locations=json.loads(r.get("target_locations") or "[]"),
            work_identity=r.get("work_identity"),
            team_preference=r.get("team_preference"),
            company_stage_preference=json.loads(r.get("company_stage_preference") or "[]"),
            engagement_format=r.get("engagement_format"),
            open_to_fractional=bool(r.get("open_to_fractional", False)),
            loved_aspects=r.get("loved_aspects") or "",
            want_next=r.get("want_next") or "",
            created_at=r["created_at"],
            completed_at=r.get("completed_at"),
        )
