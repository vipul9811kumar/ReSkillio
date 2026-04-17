"""
ReSkillio Vertex AI Pipeline — resume ingestion end-to-end.

Flow
----
  load_document  →  extract_skills  →  store_embeddings
      (GCS PDF)       (BQ persist)       (Vertex embed)

Each step is a versioned, reproducible KFP v2 component running inside
the reskillio-pipeline Docker image.
"""

from kfp import dsl

from reskillio.vertex.components import (
    load_document,
    extract_skills,
    store_embeddings,
)

PIPELINE_NAME = "reskillio-ingestion-pipeline"
PIPELINE_DESCRIPTION = (
    "ReSkillio end-to-end resume ingestion: "
    "PDF → section parse → skill extraction → embedding storage."
)


@dsl.pipeline(name=PIPELINE_NAME, description=PIPELINE_DESCRIPTION)
def ingestion_pipeline(
    gcs_pdf_uri: str,
    candidate_id: str,
    project_id: str,
    region: str = "us-central1",
) -> None:
    """
    Parameters
    ----------
    gcs_pdf_uri:
        GCS path to the candidate's PDF resume, e.g.
        gs://reskillio-dev-2026-artifacts/resumes/candidate-vk-001.pdf
    candidate_id:
        Unique candidate identifier used across all BQ tables.
    project_id:
        GCP project ID (e.g. reskillio-dev-2026).
    region:
        Vertex AI region for on-the-fly embedding (default: us-central1).
    """
    load_task = load_document(
        gcs_uri=gcs_pdf_uri,
        candidate_id=candidate_id,
    )

    extract_task = extract_skills(
        sections=load_task.outputs["sections"],
        candidate_id=candidate_id,
        project_id=project_id,
    )

    store_embeddings(
        candidate_id=candidate_id,
        project_id=project_id,
        region=region,
        skill_count=extract_task.output,  # enforces sequential execution
    )
