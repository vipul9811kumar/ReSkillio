"""
Industry vector store.

Maintains `industry_vectors` — one 768-dim centroid embedding per industry,
computed as the demand_weight-weighted average of skill embeddings.

Scoring uses BigQuery ML.DISTANCE (COSINE) to compare a candidate vector
against all 8 industry vectors in a single SQL round-trip.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from google.cloud import bigquery
from google.api_core.exceptions import NotFound

logger = logging.getLogger(__name__)

INDUSTRY_VECTORS_SCHEMA = [
    bigquery.SchemaField("industry",    "STRING",  mode="REQUIRED"),
    bigquery.SchemaField("embedding",   "FLOAT64", mode="REPEATED"),
    bigquery.SchemaField("skill_count", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("updated_at",  "TIMESTAMP", mode="REQUIRED"),
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


class IndustryVectorStore:
    """Builds and queries the industry_vectors table."""

    def __init__(self, project_id: str, dataset_id: str = "reskillio") -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table = f"{project_id}.{dataset_id}.industry_vectors"
        self.industry_profiles = f"{project_id}.{dataset_id}.industry_profiles"
        self.skill_embeddings  = f"{project_id}.{dataset_id}.skill_embeddings"
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
        ref = bigquery.Table(self.table, schema=INDUSTRY_VECTORS_SCHEMA)
        try:
            self.client.get_table(ref)
            logger.info("Table industry_vectors already exists")
        except NotFound:
            self.client.create_table(ref)
            logger.info(f"Created table {self.table}")

    # ------------------------------------------------------------------
    # Build — demand-weight-averaged centroid per industry
    # ------------------------------------------------------------------

    def build_industry_vectors(
        self,
        embedder=None,  # VertexEmbedder instance for on-the-fly embedding
        embedding_store=None,  # EmbeddingStore for catalog upsert
    ) -> int:
        """
        (Re)compute the 768-dim centroid embedding for each industry.

        For each industry: centroid = Σ(demand_weight_i × embedding_i) / Σ(demand_weight_i)
        Only skills with an embedding in the catalog contribute.

        Returns
        -------
        int — number of industry vectors written.
        """
        # 1. Pull industry profile skill weights
        profile_rows = list(self.client.query(f"""
            SELECT industry, skill_name, category, demand_weight
            FROM `{self.industry_profiles}`
            ORDER BY industry, demand_weight DESC
        """).result())

        # 2. Collect all unique skill names
        all_skill_names = list({r["skill_name"] for r in profile_rows})

        # 3. Fetch / embed all skill vectors
        skill_vecs: dict[str, list[float]] = {}
        if embedding_store is not None:
            skill_vecs = embedding_store.get_embeddings_batch(all_skill_names)

        # Embed any missing skills on the fly
        missing = [n for n in all_skill_names if n.lower() not in skill_vecs]
        if missing and embedder is not None and embedding_store is not None:
            from reskillio.embeddings.vertex_embedder import skill_text, EMBEDDING_MODEL
            logger.info(f"Embedding {len(missing)} uncached industry skills...")
            cat_map = {
                r["skill_name"].lower(): r["category"] for r in profile_rows
            }
            pairs = [(n, cat_map.get(n.lower(), "unknown")) for n in missing]
            embedded = embedder.embed_skills(pairs)
            embedding_store.upsert_embeddings(
                skills=embedded,
                embed_text_fn=skill_text,
                model_name=EMBEDDING_MODEL,
            )
            for name, _cat, vec in embedded:
                skill_vecs[name.lower()] = vec

        # 4. Compute weighted centroid per industry
        industry_map: dict[str, list[tuple[float, list[float]]]] = {}
        for row in profile_rows:
            name  = row["skill_name"]
            vec   = skill_vecs.get(name.lower())
            if vec is None:
                continue
            w = float(row["demand_weight"])
            industry_map.setdefault(row["industry"], []).append((w, vec))

        # 5. Write centroids to BQ (full replace)
        updated_at = datetime.now(timezone.utc).isoformat()
        rows_to_insert = []
        for industry, weighted_vecs in industry_map.items():
            total_w = sum(w for w, _ in weighted_vecs)
            dims = len(weighted_vecs[0][1])
            centroid = np.zeros(dims, dtype=np.float64)
            for w, vec in weighted_vecs:
                centroid += (w / total_w) * np.array(vec, dtype=np.float64)

            rows_to_insert.append({
                "industry":    industry,
                "embedding":   centroid.tolist(),
                "skill_count": len(weighted_vecs),
                "updated_at":  updated_at,
            })
            logger.info(
                f"  {industry:25} — centroid from {len(weighted_vecs)} skills"
            )

        # Truncate and replace
        self.client.query(f"DELETE FROM `{self.table}` WHERE TRUE").result()
        if rows_to_insert:
            errors = self.client.insert_rows_json(self.table, rows_to_insert)
            if errors:
                raise RuntimeError(f"industry_vectors insert errors: {errors}")

        logger.info(f"Built {len(rows_to_insert)} industry vectors")
        return len(rows_to_insert)

    # ------------------------------------------------------------------
    # Score — BQML ML.DISTANCE
    # ------------------------------------------------------------------

    def score_candidate(self, candidate_vector: list[float]) -> list[dict]:
        """
        Score a candidate vector against all 8 industry vectors using
        BigQuery ML.DISTANCE (COSINE).

        Returns
        -------
        list of dicts ordered by match_score desc:
            industry, cosine_distance, match_score (0–100)
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(
                    "candidate_vec", "FLOAT64", candidate_vector
                )
            ]
        )

        query = f"""
        SELECT
            industry,
            ML.DISTANCE(@candidate_vec, embedding, 'COSINE') AS cosine_distance,
            ROUND(
                GREATEST(0.0,
                    (1.0 - ML.DISTANCE(@candidate_vec, embedding, 'COSINE'))
                ) * 100.0,
                1
            ) AS match_score
        FROM `{self.table}`
        ORDER BY match_score DESC
        """

        rows = self.client.query(query, job_config=job_config).result()
        return [
            {
                "industry":       row["industry"],
                "cosine_distance": float(row["cosine_distance"]),
                "match_score":    float(row["match_score"]),
            }
            for row in rows
        ]