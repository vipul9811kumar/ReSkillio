"""Embedding endpoints — trigger embedding pipeline and semantic skill search."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from loguru import logger

from reskillio.api.routes.extract import _get_profile_store
from reskillio.pipelines.embedding_pipeline import run_embedding_pipeline, embed_skill_query
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.profile_store import CandidateProfileStore
from config.settings import settings

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def _get_embedding_store() -> EmbeddingStore | None:
    if not settings.gcp_project_id:
        return None
    return EmbeddingStore(project_id=settings.gcp_project_id)


class SimilarSkillsRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Skill name or free-form text to search")
    top_k: int = Field(default=10, ge=1, le=50)
    category_filter: str | None = Field(default=None, description="Filter by category: technical, soft, tool, certification")


class SimilarSkillsResponse(BaseModel):
    query: str
    results: list[dict]


@router.post("/candidate/{candidate_id}", response_model=dict)
def embed_candidate_skills(
    candidate_id: str,
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> dict:
    """
    Embed all skills in a candidate's profile that aren't yet in the global catalog.

    - Reads skills from candidate_profiles table.
    - Skips skills already embedded (idempotent).
    - Calls Vertex AI text-embedding-004 for new skills.
    - Stores vectors in skill_embeddings table.
    """
    if profile_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    logger.info(f"Embedding pipeline triggered for candidate='{candidate_id}'")
    try:
        result = run_embedding_pipeline(
            candidate_id=candidate_id,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
    except Exception as exc:
        logger.error(f"Embedding pipeline failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding pipeline failed: {exc}",
        ) from exc

    return result


@router.post("/similar", response_model=SimilarSkillsResponse)
def find_similar_skills(
    body: SimilarSkillsRequest,
    embedding_store: EmbeddingStore | None = Depends(_get_embedding_store),
) -> SimilarSkillsResponse:
    """
    Find semantically similar skills using vector search.

    - Embeds the query text with RETRIEVAL_QUERY task type.
    - Runs COSINE similarity search against the skill catalog.
    - Returns top-K results ordered by similarity (distance ASC).
    """
    if embedding_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    logger.info(f"Similarity search: query='{body.query}' top_k={body.top_k}")

    try:
        query_vector = embed_skill_query(
            body.query,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
        results = embedding_store.find_similar(
            query_vector=query_vector,
            top_k=body.top_k,
            category_filter=body.category_filter,
        )
    except Exception as exc:
        logger.error(f"Similarity search failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity search failed: {exc}",
        ) from exc

    return SimilarSkillsResponse(query=body.query, results=results)
