"""
F6 — Industry match pipeline.

Builds a frequency-weighted candidate centroid embedding, then calls
ML.DISTANCE against all 8 pre-built industry vectors in BigQuery.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from reskillio.embeddings.vertex_embedder import VertexEmbedder, skill_text, EMBEDDING_MODEL
from reskillio.models.industry import IndustryMatchResult
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.industry_vector_store import IndustryVectorStore
from reskillio.storage.profile_store import CandidateProfileStore


def _build_candidate_vector(
    candidate_id: str,
    profile_store: CandidateProfileStore,
    embedding_store: EmbeddingStore,
    embedder: VertexEmbedder,
) -> list[float]:
    """
    Compute a frequency-weighted centroid embedding for a candidate.

    centroid = Σ(frequency_i × embedding_i) / Σ(frequency_i)

    Embeds any skills not yet in the catalog on the fly.
    """
    profile = profile_store.get_profile(candidate_id)
    if not profile.skills:
        raise ValueError(f"No profile found for candidate '{candidate_id}'.")

    skill_names = [s.skill_name for s in profile.skills]
    freq_map    = {s.skill_name.lower(): s.frequency for s in profile.skills}
    cat_map     = {s.skill_name.lower(): s.category.value for s in profile.skills}

    # Fetch cached embeddings
    vecs = embedding_store.get_embeddings_batch(skill_names)

    # Embed missing ones on the fly
    missing = [n for n in skill_names if n.lower() not in vecs]
    if missing:
        logger.debug(f"Embedding {len(missing)} uncached candidate skills")
        pairs    = [(n, cat_map.get(n.lower(), "unknown")) for n in missing]
        embedded = embedder.embed_skills(pairs)
        embedding_store.upsert_embeddings(
            skills=embedded,
            embed_text_fn=skill_text,
            model_name=EMBEDDING_MODEL,
        )
        for name, _cat, vec in embedded:
            vecs[name.lower()] = vec

    # Weighted centroid
    total_freq = sum(freq_map.values())
    dims = len(next(iter(vecs.values())))
    centroid = np.zeros(dims, dtype=np.float64)

    covered = 0
    for name_lower, vec in vecs.items():
        w = freq_map.get(name_lower, 1) / total_freq
        centroid += w * np.array(vec, dtype=np.float64)
        covered += 1

    logger.info(
        f"Candidate vector built from {covered}/{len(skill_names)} skills "
        f"(freq-weighted centroid)"
    )
    return centroid.tolist()


def run_industry_match(
    candidate_id: str,
    project_id: str,
    region: str = "us-central1",
) -> IndustryMatchResult:
    """
    Score a candidate against all 8 industries using BQML ML.DISTANCE.

    Parameters
    ----------
    candidate_id:
        Candidate whose profile to score.
    project_id:
        GCP project.
    region:
        Vertex AI region for on-the-fly embedding.

    Returns
    -------
    IndustryMatchResult — ranked list of 8 industries with match_score 0–100.
    """
    profile_store  = CandidateProfileStore(project_id=project_id)
    embedding_store = EmbeddingStore(project_id=project_id)
    industry_store  = IndustryVectorStore(project_id=project_id)
    embedder        = VertexEmbedder(project_id=project_id, region=region)

    logger.info(f"Industry match started for candidate='{candidate_id}'")

    # 1. Build candidate centroid vector
    candidate_vec = _build_candidate_vector(
        candidate_id, profile_store, embedding_store, embedder
    )

    # 2. BQML cosine scoring against all industry vectors
    logger.info("Running ML.DISTANCE against industry_vectors...")
    scored_rows = industry_store.score_candidate(candidate_vec)

    if not scored_rows:
        raise RuntimeError(
            "No industry vectors found. Run scripts/build_industry_vectors.py first."
        )

    logger.info("Industry match scores:")
    for row in scored_rows:
        logger.info(
            f"  {row['industry']:25} "
            f"match_score={row['match_score']:5.1f}  "
            f"cosine_dist={row['cosine_distance']:.4f}"
        )

    return IndustryMatchResult.from_bq_rows(candidate_id, scored_rows)