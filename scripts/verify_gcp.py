"""
End-to-end verification: extract skills → write to BigQuery → read back.
Run after setup_gcp.py succeeds.

    python3 scripts/verify_gcp.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from reskillio.pipelines.skill_pipeline import run_skill_extraction
from reskillio.storage.bigquery_store import BigQuerySkillStore
from reskillio.models.skill import SkillCategory

SAMPLE_TEXT = (
    "Senior Python engineer with 6 years experience in FastAPI, Docker, "
    "Kubernetes, machine learning and PostgreSQL. Strong leadership skills."
)
CANDIDATE_ID = "verify-test-001"


def main() -> None:
    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID not set — run setup_gcp.py first.")
        sys.exit(1)

    # 1. Extract
    logger.info("Step 1: Extracting skills...")
    result = run_skill_extraction(SAMPLE_TEXT)
    logger.info(f"  Extracted {result.skill_count} skills: {result.unique_skill_names()}")

    # 2. Write
    logger.info("Step 2: Writing to BigQuery...")
    store = BigQuerySkillStore(project_id=settings.gcp_project_id)
    rows_written = store.store_extraction(result, candidate_id=CANDIDATE_ID)
    logger.info(f"  Wrote {rows_written} rows")

    # 3. Read back
    logger.info("Step 3: Reading back from BigQuery...")
    skills = store.get_skills_for_candidate(CANDIDATE_ID)
    logger.info(f"  Retrieved {len(skills)} rows for candidate '{CANDIDATE_ID}'")
    for row in skills:
        logger.info(f"    {row['skill_name']:25} | {row['category']:15} | {row['confidence']}")

    # 4. Filter by category
    tech_skills = store.get_top_skills_by_category(CANDIDATE_ID, SkillCategory.TECHNICAL)
    logger.info(f"  Technical skills: {tech_skills}")

    logger.info("Verification complete.")


if __name__ == "__main__":
    main()