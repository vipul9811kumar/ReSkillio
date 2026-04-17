"""
F15 — One-time CI/CD setup for the taxonomy retraining pipeline.

Wires GCS → Pub/Sub → Cloud Build trigger so that uploading a new
taxonomy.json to GCS automatically triggers the retraining pipeline.

Run after setup_gcp.py and export_taxonomy.py.

    python scripts/setup_cicd.py [--dry-run] [--bucket BUCKET] [--repo REPO]

What this script configures
----------------------------
1. Pub/Sub topic         : taxonomy-changed
2. GCS bucket notification: any OBJECT_FINALIZE under taxonomy/ prefix
3. Cloud Build Pub/Sub trigger: runs cloudbuild.yaml from the connected repo
4. IAM: grants Cloud Build SA the minimum required roles

Prerequisites
-------------
- gcloud CLI installed and configured (gcloud auth application-default login)
- GCS bucket exists (run export_taxonomy.py first)
- Cloud Build API enabled: gcloud services enable cloudbuild.googleapis.com
- Cloud Source Repositories or GitHub connected to Cloud Build
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config.settings import settings

_TOPIC_NAME    = "taxonomy-changed"
_TRIGGER_NAME  = "taxonomy-retrain"
_DEFAULT_BLOB_PREFIX = "taxonomy/"


def _run(cmd: list[str], dry_run: bool, description: str) -> bool:
    """Run a shell command, or print it if dry_run is set."""
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY RUN] {cmd_str}")
        return True

    logger.info(f"Running: {cmd_str}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"[{description}] FAILED (rc={result.returncode})")
        logger.warning(f"  stderr: {result.stderr.strip()}")
        return False

    logger.info(f"[{description}] OK")
    if result.stdout.strip():
        logger.debug(f"  stdout: {result.stdout.strip()}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F15 — Set up CI/CD trigger for taxonomy retraining"
    )
    parser.add_argument("--bucket",  default="",
                        help="GCS bucket name (default: {project}-models)")
    parser.add_argument("--repo",    default="",
                        help="Cloud Source Repo name or GitHub repo (owner/repo)")
    parser.add_argument("--repo-type", default="CLOUD_SOURCE_REPOSITORIES",
                        choices=["CLOUD_SOURCE_REPOSITORIES", "GITHUB"],
                        help="Source repository type (default: CLOUD_SOURCE_REPOSITORIES)")
    parser.add_argument("--branch",  default="main",
                        help="Git branch to use for cloudbuild.yaml (default: main)")
    parser.add_argument("--region",  default="us-central1",
                        help="GCP region (default: us-central1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print gcloud commands without executing them")
    args = parser.parse_args()

    project    = settings.gcp_project_id
    if not project:
        logger.error("GCP_PROJECT_ID not set.")
        sys.exit(1)

    bucket = args.bucket or f"{project}-models"
    topic_fqn = f"projects/{project}/topics/{_TOPIC_NAME}"

    print()
    print("F15 — CI/CD Retraining Pipeline Setup")
    print("=" * 50)
    print(f"  Project : {project}")
    print(f"  Bucket  : gs://{bucket}")
    print(f"  Topic   : {topic_fqn}")
    print(f"  Trigger : {_TRIGGER_NAME}")
    print(f"  Dry run : {args.dry_run}")
    print()

    ok = True

    # -----------------------------------------------------------------------
    # Step 1: Enable required APIs
    # -----------------------------------------------------------------------
    print("Step 1: Enabling required GCP APIs...")
    for api in ["cloudbuild.googleapis.com", "pubsub.googleapis.com"]:
        ok &= _run(
            ["gcloud", "services", "enable", api, f"--project={project}"],
            args.dry_run,
            f"enable {api}",
        )

    # -----------------------------------------------------------------------
    # Step 2: Create Pub/Sub topic
    # -----------------------------------------------------------------------
    print("\nStep 2: Creating Pub/Sub topic...")
    ok &= _run(
        ["gcloud", "pubsub", "topics", "create", _TOPIC_NAME,
         f"--project={project}"],
        args.dry_run,
        "create pubsub topic",
    )

    # -----------------------------------------------------------------------
    # Step 3: Grant GCS permission to publish to the topic
    # GCS sends notifications as the GCS service agent SA.
    # -----------------------------------------------------------------------
    print("\nStep 3: Granting GCS → Pub/Sub publish permission...")
    gcs_sa = f"serviceAccount:service-$(gcloud projects describe {project} " \
              f"--format='value(projectNumber)')@gs-project-accounts.iam.gserviceaccount.com"

    # Use a direct command for the IAM binding
    ok &= _run(
        ["gcloud", "projects", "add-iam-policy-binding", project,
         f"--member=serviceAccount:service-{project}@gs-project-accounts.iam.gserviceaccount.com",
         "--role=roles/pubsub.publisher"],
        args.dry_run,
        "grant GCS pubsub publisher",
    )

    # -----------------------------------------------------------------------
    # Step 4: Create GCS bucket notification
    # -----------------------------------------------------------------------
    print("\nStep 4: Creating GCS notification for taxonomy prefix...")
    ok &= _run(
        ["gsutil", "notification", "create",
         "-t", _TOPIC_NAME,
         "-f", "json",
         "-e", "OBJECT_FINALIZE",
         "-p", _DEFAULT_BLOB_PREFIX,
         f"gs://{bucket}"],
        args.dry_run,
        "create GCS notification",
    )

    # -----------------------------------------------------------------------
    # Step 5: Grant Cloud Build SA minimum required roles
    # -----------------------------------------------------------------------
    print("\nStep 5: Granting Cloud Build SA required IAM roles...")
    cb_sa = f"{project}@cloudbuild.gserviceaccount.com"
    roles = [
        "roles/storage.objectAdmin",     # read taxonomy + write artifacts
        "roles/aiplatform.user",         # Vertex AI Model Registry
        "roles/bigquery.dataViewer",     # optional: query existing versions
        "roles/logging.logWriter",       # write Cloud Build logs
    ]
    for role in roles:
        ok &= _run(
            ["gcloud", "projects", "add-iam-policy-binding", project,
             f"--member=serviceAccount:{cb_sa}",
             f"--role={role}"],
            args.dry_run,
            f"grant {role}",
        )

    # -----------------------------------------------------------------------
    # Step 6: Create Cloud Build Pub/Sub trigger
    # -----------------------------------------------------------------------
    print("\nStep 6: Creating Cloud Build Pub/Sub trigger...")

    trigger_cmd = [
        "gcloud", "builds", "triggers", "create", "pubsub",
        f"--name={_TRIGGER_NAME}",
        f"--topic={topic_fqn}",
        "--build-config=cloudbuild.yaml",
        f"--region={args.region}",
        f"--project={project}",
        "--substitutions=" + ",".join([
            f"_TAXONOMY_GCS_PATH=gs://{bucket}/taxonomy/taxonomy.json",
            f"_ARTIFACT_BUCKET={bucket}",
            f"_REGION={args.region}",
            "_SPACY_MODEL=en_core_web_lg",
        ]),
    ]

    if args.repo:
        if args.repo_type == "GITHUB":
            owner, repo_name = args.repo.split("/", 1)
            trigger_cmd += [
                "--github-owner",   owner,
                "--github-name",    repo_name,
                f"--branch-pattern={args.branch}",
            ]
        else:
            trigger_cmd += [
                f"--repo={args.repo}",
                f"--branch={args.branch}",
            ]
    else:
        logger.warning(
            "No --repo specified. Trigger created without source — "
            "attach a repo manually in GCP Console → Cloud Build → Triggers."
        )

    ok &= _run(trigger_cmd, args.dry_run, "create Cloud Build trigger")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 50)
    if ok:
        print("  Setup complete ✓")
        print()
        print("  Trigger activated. To fire a retrain:")
        print(f"    gsutil cp <new_taxonomy.json> gs://{bucket}/taxonomy/taxonomy.json")
        print()
        print("  Monitor builds:")
        print(f"    https://console.cloud.google.com/cloud-build/builds?project={project}")
    else:
        print("  Setup completed with warnings. Review errors above.")
        print("  Some steps may require manual configuration in GCP Console.")
    print()

    if args.dry_run:
        print("  [DRY RUN] No changes were made.")
        print()
        print("  GCS notification IAM note:")
        print("  The GCS SA project number is not known at script time.")
        print("  Run 'gcloud projects describe <project>' to get the project number,")
        print("  then set the --member accordingly in step 3.")
        print()


if __name__ == "__main__":
    main()
