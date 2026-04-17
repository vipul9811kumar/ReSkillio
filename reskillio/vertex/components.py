"""
KFP v2 component definitions for the ReSkillio ingestion pipeline.

Each component runs inside PIPELINE_IMAGE and is a thin wrapper around
the reskillio package. Imports happen inside the function body so KFP
can serialise them correctly.

Build and push the image before compiling the pipeline:
    docker build -t gcr.io/reskillio-dev-2026/reskillio-pipeline:latest .
    docker push gcr.io/reskillio-dev-2026/reskillio-pipeline:latest
"""

from kfp import dsl
from kfp.dsl import Dataset, Input, Output

PIPELINE_IMAGE = "gcr.io/reskillio-dev-2026/reskillio-pipeline:latest"


# ---------------------------------------------------------------------------
# Component 1 — load_document
# ---------------------------------------------------------------------------

@dsl.component(base_image=PIPELINE_IMAGE)
def load_document(
    gcs_uri: str,
    candidate_id: str,
    sections: Output[Dataset],
) -> None:
    """
    Download a PDF from GCS, extract text, split into labelled sections.

    Writes a JSON array of {section_type, heading, text} to the sections
    artifact — the contract between this component and extract_skills.
    """
    import json
    from google.cloud import storage
    from reskillio.ingestion.pdf_reader import extract_text_from_pdf
    from reskillio.ingestion.section_parser import parse_sections

    assert gcs_uri.startswith("gs://"), f"Expected gs:// URI, got: {gcs_uri}"
    without_prefix = gcs_uri[5:]
    bucket_name, _, blob_name = without_prefix.partition("/")

    gcs = storage.Client()
    pdf_bytes = gcs.bucket(bucket_name).blob(blob_name).download_as_bytes()

    raw_text = extract_text_from_pdf(pdf_bytes)

    parsed = parse_sections(raw_text)
    sections_data = [
        {
            "section_type": st.value,
            "heading": heading,
            "text": text,
        }
        for st, heading, text in parsed
        if text.strip()
    ]

    with open(sections.path, "w") as f:
        json.dump(sections_data, f, indent=2)

    print(
        f"load_document: candidate={candidate_id!r}  "
        f"chars={len(raw_text)}  sections={len(sections_data)}"
    )


# ---------------------------------------------------------------------------
# Component 2 — extract_skills
# ---------------------------------------------------------------------------

@dsl.component(base_image=PIPELINE_IMAGE)
def extract_skills(
    sections: Input[Dataset],
    candidate_id: str,
    project_id: str,
) -> int:
    """
    Run spaCy skill extraction across all sections, persist to BigQuery, and
    refresh the candidate profile.

    Returns
    -------
    int — total unique skills extracted and stored.
    """
    import json
    from reskillio.models.skill import ExtractionResult
    from reskillio.nlp.skill_extractor import SkillExtractor, _deduplicate
    from reskillio.storage.bigquery_store import BigQuerySkillStore
    from reskillio.storage.profile_store import CandidateProfileStore

    with open(sections.path) as f:
        sections_data: list[dict] = json.load(f)

    extractor = SkillExtractor()
    all_skills = []
    full_text_parts = []

    for sec in sections_data:
        text = sec["text"].strip()
        if not text:
            continue
        full_text_parts.append(text)
        result = extractor.extract(text)
        all_skills.extend(result.skills)

    deduped = _deduplicate(all_skills)

    bq_store = BigQuerySkillStore(project_id=project_id)
    extraction = ExtractionResult(
        input_text="\n\n".join(full_text_parts),
        skills=deduped,
        model_used="en_core_web_lg",
    )
    bq_store.store_extraction(extraction, candidate_id=candidate_id)

    profile_store = CandidateProfileStore(project_id=project_id)
    profile_store.upsert_profile(candidate_id)

    print(
        f"extract_skills: candidate={candidate_id!r}  "
        f"sections={len(sections_data)}  unique_skills={len(deduped)}"
    )
    return len(deduped)


# ---------------------------------------------------------------------------
# Component 3 — store_embeddings
# ---------------------------------------------------------------------------

@dsl.component(base_image=PIPELINE_IMAGE)
def store_embeddings(
    candidate_id: str,
    project_id: str,
    region: str,
    skill_count: int,  # data-dependency pin: waits for extract_skills to finish
) -> int:
    """
    Embed all candidate profile skills not yet in the catalog, upsert to BQ.

    Delegates entirely to run_embedding_pipeline, which reads the already-
    refreshed candidate_profiles row written by extract_skills.

    Returns
    -------
    int — number of new embeddings inserted this run.
    """
    from reskillio.pipelines.embedding_pipeline import run_embedding_pipeline

    result = run_embedding_pipeline(
        candidate_id=candidate_id,
        project_id=project_id,
        region=region,
    )

    print(
        f"store_embeddings: candidate={candidate_id!r}  "
        f"found={result['skills_found']}  "
        f"embedded={result['skills_embedded']}  "
        f"skipped={result['skills_skipped']}"
    )
    return result["skills_embedded"]
