"""
AI Studio embeddings — drop-in replacement for VertexEmbedder.

Uses Google's text-embedding-004 via the AI Studio API key (GEMINI_API_KEY).
Same model, same 768-dim output — existing BigQuery skill_embeddings and
industry_vectors tables require no schema changes.

Free tier: 1,500 requests/day (enough for demo scale).
Batch limit: 100 texts per call (vs 250 for Vertex).
"""

from __future__ import annotations

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768
_BATCH_SIZE = 100  # AI Studio limit per request


def skill_text(skill_name: str, category: str) -> str:
    return f"{skill_name} ({category} skill)"


class AIStudioEmbedder:
    """
    Embeds text strings using Google AI Studio text-embedding-004.
    Interface matches VertexEmbedder exactly so callers need no changes.
    """

    def __init__(
        self,
        project_id: str = "",
        region: str = "us-central1",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.task_type = task_type
        try:
            from config.settings import settings
            self._api_key = settings.gemini_api_key
        except Exception:
            import os
            self._api_key = os.environ.get("GEMINI_API_KEY", "")

        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY not set — cannot embed without it")

    def _client(self):
        from google import genai
        return genai.Client(api_key=self._api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types as gt
        client = self._client()
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=gt.EmbedContentConfig(
                task_type=self.task_type,
                output_dimensionality=EMBEDDING_DIMENSIONS,
            ),
        )
        return [emb.values for emb in response.embeddings]

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            logger.debug(f"[embed] Batch {i // _BATCH_SIZE + 1} ({len(batch)} texts)")
            all_vectors.extend(self._embed_batch(batch))

        logger.info(f"[embed] {len(all_vectors)} texts via {EMBEDDING_MODEL} (AI Studio)")
        return all_vectors

    def embed_skills(
        self, skills: list[tuple[str, str]]
    ) -> list[tuple[str, str, list[float]]]:
        texts = [skill_text(name, cat) for name, cat in skills]
        vectors = self.embed(texts)
        return [(name, cat, vec) for (name, cat), vec in zip(skills, vectors)]
