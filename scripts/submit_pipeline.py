"""
Submit the compiled ReSkillio ingestion pipeline to Vertex AI Pipelines.

Prerequisites
-------------
1.  Build & push the pipeline image:
        docker build -t gcr.io/reskillio-dev-2026/reskillio-pipeline:latest .
        docker push gcr.io/reskillio-dev-2026/reskillio-pipeline:latest

2.  Compile the pipeline spec:
        python3 scripts/compile_pipeline.py

3.  Create a GCS bucket for pipeline artifacts and upload the PDF:
        gsutil mb -p reskillio-dev-2026 gs://reskillio-dev-2026-artifacts
        gsutil cp /path/to/resume.pdf gs://reskillio-dev-2026-artifacts/resumes/candidate-vk-001.pdf

4.  Run this script:
        python3 scripts/submit_pipeline.py --candidate-id candidate-vk-001 \\
            --gcs-pdf-uri gs://reskillio-dev-2026-artifacts/resumes/candidate-vk-001.pdf

Usage
-----
    python3 scripts/submit_pipeline.py --help
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import aiplatform
from config.settings import settings

PIPELINE_YAML = Path(__file__).parent.parent / "pipeline.yaml"
GCS_PIPELINE_ROOT = "gs://reskillio-dev-2026-artifacts/pipeline-runs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit ReSkillio pipeline to Vertex AI")
    parser.add_argument("--candidate-id",  required=True,  help="Candidate identifier")
    parser.add_argument("--gcs-pdf-uri",   required=True,  help="gs:// URI of the PDF resume")
    parser.add_argument("--project-id",    default=settings.gcp_project_id, help="GCP project ID")
    parser.add_argument("--region",        default=settings.gcp_region,      help="Vertex AI region")
    parser.add_argument("--pipeline-yaml", default=str(PIPELINE_YAML),       help="Compiled pipeline YAML")
    parser.add_argument("--pipeline-root", default=GCS_PIPELINE_ROOT,        help="GCS root for pipeline artefacts")
    parser.add_argument("--no-wait",       action="store_true",               help="Submit and return immediately")
    args = parser.parse_args()

    if not args.project_id:
        print("ERROR: GCP_PROJECT_ID not set. Check .env or pass --project-id.")
        sys.exit(1)

    creds_path = settings.google_application_credentials
    if creds_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path

    pipeline_yaml = Path(args.pipeline_yaml)
    if not pipeline_yaml.exists():
        print(f"ERROR: Pipeline YAML not found at {pipeline_yaml}")
        print("Run:  python3 scripts/compile_pipeline.py  first.")
        sys.exit(1)

    print(f"Project:      {args.project_id}")
    print(f"Region:       {args.region}")
    print(f"Candidate:    {args.candidate_id}")
    print(f"PDF:          {args.gcs_pdf_uri}")
    print(f"Pipeline:     {pipeline_yaml}")
    print(f"Root:         {args.pipeline_root}")

    aiplatform.init(project=args.project_id, location=args.region)

    job = aiplatform.PipelineJob(
        display_name=f"reskillio-ingest-{args.candidate_id}",
        template_path=str(pipeline_yaml),
        pipeline_root=args.pipeline_root,
        parameter_values={
            "gcs_pdf_uri":   args.gcs_pdf_uri,
            "candidate_id":  args.candidate_id,
            "project_id":    args.project_id,
            "region":        args.region,
        },
        enable_caching=True,
    )

    print("\nSubmitting pipeline job...")
    job.submit()

    print(f"\nJob submitted: {job.resource_name}")
    print(f"Console:  https://console.cloud.google.com/vertex-ai/pipelines/runs"
          f"?project={args.project_id}&region={args.region}")

    if not args.no_wait:
        print("Waiting for pipeline to complete (Ctrl-C to detach)...")
        job.wait()
        print(f"\nPipeline finished  state={job.state.name}")


if __name__ == "__main__":
    main()
