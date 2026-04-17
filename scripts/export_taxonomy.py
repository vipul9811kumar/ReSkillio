"""
Bootstrap script — export the current in-code taxonomy to GCS as taxonomy.json.

Run this ONCE after initial deployment to seed the GCS file that the
Cloud Build CI/CD trigger will watch for changes.

    python scripts/export_taxonomy.py [--bucket BUCKET] [--dry-run]

After this file exists in GCS, edit it there (or via a review PR that copies
a new version to GCS) to trigger the retraining pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings
from reskillio.nlp.skill_extractor import ALL_SKILLS
from reskillio.registry.retrainer import taxonomy_to_json

_DEFAULT_BLOB = "taxonomy/taxonomy.json"


def _default_bucket(project_id: str) -> str:
    return f"{project_id}-models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export in-code taxonomy to GCS")
    parser.add_argument("--bucket", default="", help="GCS bucket name (default: {project}-models)")
    parser.add_argument("--blob",   default=_DEFAULT_BLOB, help=f"GCS object path (default: {_DEFAULT_BLOB})")
    parser.add_argument("--version",default="1.0.0", help="Taxonomy version string")
    parser.add_argument("--dry-run",action="store_true", help="Print JSON only, do not upload")
    args = parser.parse_args()

    if not settings.gcp_project_id:
        logger.error("GCP_PROJECT_ID not set. Check your .env file.")
        sys.exit(1)

    bucket = args.bucket or _default_bucket(settings.gcp_project_id)
    data   = taxonomy_to_json(ALL_SKILLS, version=args.version)

    tech_count  = len(data["tech"])
    soft_count  = len(data["soft"])
    tools_count = len(data["tools"])
    cert_count  = len(data["certifications"])
    total       = tech_count + soft_count + tools_count + cert_count

    logger.info(f"Taxonomy: {total} skills")
    logger.info(f"  tech={tech_count}  soft={soft_count}  tools={tools_count}  certs={cert_count}")

    if args.dry_run:
        print(json.dumps(data, indent=2))
        logger.info("[dry-run] Skipped GCS upload.")
        return

    import os
    from google.cloud import storage
    from google.cloud.exceptions import Conflict

    if settings.google_application_credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials

    client      = storage.Client(project=settings.gcp_project_id)
    bucket_name = bucket

    # Ensure bucket exists — try to get it first, then create if missing
    from google.api_core.exceptions import Forbidden, NotFound
    bucket_ok = False
    try:
        client.get_bucket(bucket_name)
        logger.info(f"Bucket exists: gs://{bucket_name}")
        bucket_ok = True
    except NotFound:
        try:
            bkt = client.bucket(bucket_name)
            bkt.storage_class = "STANDARD"
            client.create_bucket(bkt, project=settings.gcp_project_id, location="US")
            logger.info(f"Created bucket: gs://{bucket_name}")
            bucket_ok = True
        except (Conflict, Forbidden):
            bucket_ok = True  # Conflict = already exists; Forbidden = SA lacks create but bucket exists
        except Exception as exc:
            logger.error(
                f"Bucket gs://{bucket_name} does not exist and could not be created.\n"
                f"  Create it manually, then re-run this script:\n"
                f"    gsutil mb -l US gs://{bucket_name}\n"
                f"  Then grant the SA object access:\n"
                f"    gsutil iam ch serviceAccount:{settings.gcp_project_id}@"
                f"{settings.gcp_project_id}.iam.gserviceaccount.com:roles/storage.objectAdmin "
                f"gs://{bucket_name}\n"
                f"  Error: {exc}"
            )
            sys.exit(1)
    except Forbidden:
        # SA can't even read bucket metadata — assume it exists, try upload anyway
        logger.warning(f"Cannot verify bucket gs://{bucket_name} (403). Attempting upload...")
        bucket_ok = True

    if not bucket_ok:
        sys.exit(1)

    blob = client.bucket(bucket_name).blob(args.blob)
    blob.upload_from_string(json.dumps(data, indent=2), content_type="application/json")

    gcs_uri = f"gs://{bucket_name}/{args.blob}"
    logger.info(f"Taxonomy exported → {gcs_uri}")
    logger.info(f"Version: {args.version}")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Run setup_cicd.py to wire GCS notifications → Cloud Build")
    logger.info(f"  2. To trigger a retrain: edit taxonomy.json in GCS Console")
    logger.info(f"     or upload a new version:")
    logger.info(f"     gsutil cp taxonomy.json {gcs_uri}")


if __name__ == "__main__":
    main()
