"""
One-time GCP setup script.
Run after .env and service-account.json are in place.

    python3 scripts/setup_gcp.py
"""

import os
import sys
from pathlib import Path

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from reskillio.storage.bigquery_store import BigQuerySkillStore
from reskillio.storage.profile_store import CandidateProfileStore
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.jd_store import JDStore
from reskillio.storage.industry_vector_store import IndustryVectorStore
from reskillio.monitoring.drift_store import DriftMetricStore
from reskillio.storage.lakehouse import LakehouseManager


def main() -> None:
    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID is not set. Check your .env file.")
        sys.exit(1)

    if not settings.google_application_credentials:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS is not set. Check your .env file.")
        sys.exit(1)

    creds_path = Path(settings.google_application_credentials)
    if not creds_path.exists():
        logger.error(f"Service account file not found: {creds_path}")
        sys.exit(1)

    # Ensure Google auth libraries can find the credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)

    logger.info(f"Project:     {settings.gcp_project_id}")
    logger.info(f"Region:      {settings.gcp_region}")
    logger.info(f"Credentials: {creds_path}")

    store           = BigQuerySkillStore(project_id=settings.gcp_project_id)
    profile_store   = CandidateProfileStore(project_id=settings.gcp_project_id)
    embedding_store = EmbeddingStore(project_id=settings.gcp_project_id)
    jd_store        = JDStore(project_id=settings.gcp_project_id)
    industry_store  = IndustryVectorStore(project_id=settings.gcp_project_id)
    drift_store     = DriftMetricStore(project_id=settings.gcp_project_id)
    lakehouse       = LakehouseManager(project_id=settings.gcp_project_id)

    logger.info("Creating dataset and tables (safe to re-run)...")
    store.ensure_dataset_and_table()
    profile_store.ensure_table()
    embedding_store.ensure_table()
    jd_store.ensure_tables()
    industry_store.ensure_table()
    drift_store.ensure_table()

    logger.info("Creating medallion lakehouse datasets (bronze / silver / gold)...")
    lakehouse.ensure_all()

    logger.info("GCP setup complete.")
    logger.info(f"Table ready: {store.full_table_id}")
    logger.info(f"Table ready: {profile_store.profiles_table}")
    logger.info(f"Table ready: {embedding_store.full_table_id}")
    logger.info(f"Table ready: {jd_store.jd_profiles_table}")
    logger.info(f"Table ready: {jd_store.jd_skills_table}")
    logger.info(f"Table ready: {industry_store.table}")
    logger.info(f"Table ready: {drift_store.full_table_id}")
    logger.info("Lakehouse layers: reskillio_bronze / reskillio_silver / reskillio_gold")


if __name__ == "__main__":
    main()