"""
BigQuery skill embeddings store.

Table: skill_embeddings
- One row per unique (skill_name, category) — global catalog, shared across candidates.
- Embedding column: ARRAY<FLOAT64> (768 dims, text-embedding-004).
- Vector search via BQ VECTOR_SEARCH function (COSINE distance).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

logger = logging.getLogger(__name__)

EMBEDDINGS_SCHEMA = [
    bigquery.SchemaField("skill_name",  "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("category",    "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("embed_text",  "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("embedding",   "FLOAT64", mode="REPEATED"),   # 768-dim vector
    bigquery.SchemaField("model",       "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("embedded_at", "TIMESTAMP", mode="REQUIRED"),
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


class EmbeddingStore:
    """
    Manages the skill_embeddings BigQuery table.

    Acts as a global skill catalog — skills are embedded once and reused
    across all candidates. Supports semantic similarity search via
    BigQuery VECTOR_SEARCH.
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str = "reskillio",
        table_id: str = "skill_embeddings",
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.full_table_id = f"{project_id}.{dataset_id}.{table_id}"
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
        """Create the skill_embeddings table if it doesn't exist."""
        table_ref = bigquery.Table(self.full_table_id, schema=EMBEDDINGS_SCHEMA)
        try:
            self.client.get_table(table_ref)
            logger.info("Table skill_embeddings already exists")
        except NotFound:
            self.client.create_table(table_ref)
            logger.info(f"Created table {self.full_table_id}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def get_existing_skills(self) -> set[tuple[str, str]]:
        """Return the set of (skill_name_lower, category) already embedded."""
        query = f"SELECT LOWER(skill_name) AS sn, category FROM `{self.full_table_id}`"
        rows = self.client.query(query).result()
        return {(row["sn"], row["category"]) for row in rows}

    def upsert_embeddings(
        self,
        skills: list[tuple[str, str, list[float]]],
        embed_text_fn,
        model_name: str = "text-embedding-004",
    ) -> int:
        """
        Insert embeddings for skills not already in the catalog.

        Parameters
        ----------
        skills:
            List of (skill_name, category, vector).
        embed_text_fn:
            Callable(skill_name, category) -> str used to populate embed_text.
        model_name:
            Model identifier stored for provenance.

        Returns
        -------
        int — number of rows inserted.
        """
        existing = self.get_existing_skills()
        embedded_at = datetime.now(timezone.utc).isoformat()

        rows = [
            {
                "skill_name":  name,
                "category":    category,
                "embed_text":  embed_text_fn(name, category),
                "embedding":   vector,
                "model":       model_name,
                "embedded_at": embedded_at,
            }
            for name, category, vector in skills
            if (name.lower(), category) not in existing
        ]

        if not rows:
            logger.info("All skills already in embedding catalog — nothing to insert")
            return 0

        errors = self.client.insert_rows_json(self.full_table_id, rows)
        if errors:
            raise RuntimeError(f"Embedding insert errors: {errors}")

        logger.info(f"Inserted {len(rows)} new skill embeddings")
        return len(rows)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def find_similar(
        self,
        query_vector: list[float],
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[dict]:
        """
        Find the top-K semantically similar skills using COSINE distance.

        Parameters
        ----------
        query_vector:
            768-dim float list (embed your query text first).
        top_k:
            Number of results to return.
        category_filter:
            Optional — restrict results to a single category.

        Returns
        -------
        list of dicts: skill_name, category, distance (lower = more similar).
        """
        # Inline the vector as a BQ ARRAY literal — safe, values are model-generated floats
        vec_literal = ", ".join(repr(float(v)) for v in query_vector)

        category_clause = (
            f"AND base.category = '{category_filter}'" if category_filter else ""
        )

        query = f"""
            SELECT
                base.skill_name,
                base.category,
                base.embed_text,
                distance
            FROM VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                'embedding',
                (SELECT [{vec_literal}] AS embedding),
                top_k => {top_k},
                distance_type => 'COSINE'
            )
            WHERE 1=1 {category_clause}
            ORDER BY distance ASC
        """
        rows = self.client.query(query).result()
        return [
            {
                "skill_name": row["skill_name"],
                "category":   row["category"],
                "embed_text": row["embed_text"],
                "distance":   round(row["distance"], 4),
            }
            for row in rows
        ]

    def get_embedding_for_skill(self, skill_name: str) -> list[float] | None:
        """Retrieve the stored embedding vector for a skill by name."""
        query = f"""
            SELECT embedding
            FROM `{self.full_table_id}`
            WHERE LOWER(skill_name) = LOWER(@skill_name)
            LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("skill_name", "STRING", skill_name)
            ]
        )
        rows = list(self.client.query(query, job_config=job_config).result())
        return list(rows[0]["embedding"]) if rows else None

    def get_embeddings_batch(
        self, skill_names: list[str]
    ) -> dict[str, list[float]]:
        """
        Retrieve embeddings for multiple skill names in one BQ query.

        Returns
        -------
        dict mapping lowercase skill_name → embedding vector.
        Missing skills are absent from the dict.
        """
        if not skill_names:
            return {}

        lower_names = [n.lower() for n in skill_names]
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("skill_names", "STRING", lower_names)
            ]
        )
        query = f"""
            SELECT LOWER(skill_name) AS key, embedding
            FROM `{self.full_table_id}`
            WHERE LOWER(skill_name) IN UNNEST(@skill_names)
        """
        rows = self.client.query(query, job_config=job_config).result()
        return {row["key"]: list(row["embedding"]) for row in rows}