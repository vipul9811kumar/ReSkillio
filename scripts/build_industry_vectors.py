"""
Build (or rebuild) the industry_vectors table.

Embeds all industry skills via Vertex AI (on the fly for any uncached),
then computes 8 demand-weight-averaged centroid embeddings and writes
them to BigQuery.

    python3 scripts/build_industry_vectors.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from reskillio.embeddings.vertex_embedder import VertexEmbedder
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.industry_vector_store import IndustryVectorStore


def main() -> None:
    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID is not set. Check your .env file.")
        sys.exit(1)

    creds_path = Path(settings.google_application_credentials or "")
    if not creds_path.exists():
        logger.error(f"Service account file not found: {creds_path}")
        sys.exit(1)

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    project_id = settings.gcp_project_id
    region     = settings.gcp_region

    logger.info(f"Project: {project_id}  |  Region: {region}")

    industry_store  = IndustryVectorStore(project_id=project_id)
    embedding_store = EmbeddingStore(project_id=project_id)
    embedder        = VertexEmbedder(project_id=project_id, region=region)

    # Ensure the target table exists
    industry_store.ensure_table()

    logger.info("Building industry vectors (embedding uncached skills on the fly)...")
    n = industry_store.build_industry_vectors(
        embedder=embedder,
        embedding_store=embedding_store,
    )

    logger.info(f"Done — {n} industry vectors written to BigQuery.")


if __name__ == "__main__":
    main()