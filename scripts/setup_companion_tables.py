"""
scripts/setup_companion_tables.py
Run once to create companion BigQuery tables.
Usage: python scripts/setup_companion_tables.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def main():
    print("=== ReSkillio Companion Setup ===\n")

    from config.settings import settings

    print("1. Creating BigQuery tables...")
    from reskillio.companion.checkin_store import CompanionStore
    store = CompanionStore(project_id=settings.gcp_project_id)
    store.ensure_tables()
    print("   reskillio.weekly_checkins    ✓")
    print("   reskillio.companion_digests  ✓")

    run_url = os.environ.get("CLOUD_RUN_URL", "https://reskillio-10933517215.us-central1.run.app")
    sa      = f"reskillio-scheduler@{settings.gcp_project_id}.iam.gserviceaccount.com"

    print("\n2. Cloud Tasks queue — run this command once:")
    print(f"""
   gcloud tasks queues create reskillio-digest-queue \\
     --project={settings.gcp_project_id} \\
     --location=us-central1 \\
     --max-concurrent-dispatches=10 \\
     --max-dispatches-per-second=2
    """)

    print("\n3. Cloud Scheduler job — run this command once:")
    print(f"""
   gcloud scheduler jobs create http reskillio-weekly-digest \\
     --project={settings.gcp_project_id} \\
     --schedule="0 6 * * 1" \\
     --uri="{run_url}/companion/trigger-digests" \\
     --message-body="{{}}" \\
     --headers="Content-Type=application/json" \\
     --oidc-service-account-email={sa} \\
     --location=us-central1 \\
     --description="Weekly digest trigger — every Monday 6am UTC"
    """)

    print("4. BigQuery progress view — paste into BQ console:")
    print(f"""
   CREATE OR REPLACE VIEW `{settings.gcp_project_id}.reskillio.companion_progress_view` AS
   SELECT
     c.candidate_id,
     c.week_number,
     c.gap_score,
     c.gap_score_delta,
     c.applications_sent,
     c.hours_on_courses,
     SUM(c.applications_sent) OVER (
       PARTITION BY c.candidate_id ORDER BY c.week_number
     ) AS total_applications,
     SUM(c.hours_on_courses) OVER (
       PARTITION BY c.candidate_id ORDER BY c.week_number
     ) AS total_hours_invested,
     FIRST_VALUE(c.gap_score) OVER (
       PARTITION BY c.candidate_id ORDER BY c.week_number
     ) AS baseline_gap_score,
     c.gap_score - FIRST_VALUE(c.gap_score) OVER (
       PARTITION BY c.candidate_id ORDER BY c.week_number
     ) AS total_gap_improvement
   FROM `{settings.gcp_project_id}.reskillio.weekly_checkins` c
   WHERE c.gap_score IS NOT NULL
   ORDER BY c.candidate_id, c.week_number;
    """)


if __name__ == "__main__":
    main()
