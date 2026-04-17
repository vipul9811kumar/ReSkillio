"""
Skill embedding pipeline.

Reads unique skills from a candidate's profile, filters out
those already in the global embedding catalog, embeds the new
ones via Vertex AI, and upserts them to BigQuery.
"""

from __future__ import annotations

from loguru import logger

from reskillio.embeddings.vertex_embedder import VertexEmbedder, skill_text, EMBEDDING_MODEL
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.profile_store import CandidateProfileStore


def run_embedding_pipeline(
    candidate_id: str,
    project_id: str,
    region: str = "us-central1",
) -> dict:
    """
    Embed all skills in a candidate's profile that aren't yet in the catalog.

    Parameters
    ----------
    candidate_id:
        The candidate whose profile skills should be embedded.
    project_id:
        GCP project ID.
    region:
        Vertex AI region.

    Returns
    -------
    dict with keys: candidate_id, skills_found, skills_embedded, skills_skipped
    """
    profile_store = CandidateProfileStore(project_id=project_id)
    embedding_store = EmbeddingStore(project_id=project_id)

    # 1. Fetch candidate profile skills
    profile = profile_store.get_profile(candidate_id)
    if not profile.skills:
        logger.warning(f"No profile skills found for candidate='{candidate_id}'")
        return {
            "candidate_id":    candidate_id,
            "skills_found":    0,
            "skills_embedded": 0,
            "skills_skipped":  0,
        }

    all_skills = [(s.skill_name, s.category.value) for s in profile.skills]
    logger.info(f"Found {len(all_skills)} skills in profile for candidate='{candidate_id}'")

    # 2. Filter to only skills not yet in the catalog
    existing = embedding_store.get_existing_skills()
    new_skills = [
        (name, cat)
        for name, cat in all_skills
        if (name.lower(), cat) not in existing
    ]

    skipped = len(all_skills) - len(new_skills)
    logger.info(f"  {len(new_skills)} new skills to embed, {skipped} already in catalog")

    if not new_skills:
        return {
            "candidate_id":    candidate_id,
            "skills_found":    len(all_skills),
            "skills_embedded": 0,
            "skills_skipped":  skipped,
        }

    # 3. Embed new skills
    embedder = VertexEmbedder(project_id=project_id, region=region)
    embedded = embedder.embed_skills(new_skills)

    # 4. Upsert into catalog
    inserted = embedding_store.upsert_embeddings(
        skills=embedded,
        embed_text_fn=skill_text,
        model_name=EMBEDDING_MODEL,
    )

    return {
        "candidate_id":    candidate_id,
        "skills_found":    len(all_skills),
        "skills_embedded": inserted,
        "skills_skipped":  skipped,
    }


def embed_skill_query(
    query_text: str,
    project_id: str,
    region: str = "us-central1",
) -> list[float]:
    """
    Embed an arbitrary query string for use in semantic search.
    Uses RETRIEVAL_QUERY task type (distinct from RETRIEVAL_DOCUMENT).
    """
    embedder = VertexEmbedder(
        project_id=project_id,
        region=region,
        task_type="RETRIEVAL_QUERY",
    )
    vectors = embedder.embed([query_text])
    return vectors[0]
