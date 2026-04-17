"""
Vertex AI text-embedding-004 client.

Embeds skill names + category context for semantic matching.
Handles batching (API limit: 250 texts per call) and retries.
"""

from __future__ import annotations

import os
from functools import lru_cache

import vertexai
from vertexai.generative_models import GenerativeModel  # noqa: F401 — triggers SDK init
from google.cloud.aiplatform_v1beta1.services.prediction_service import PredictionServiceClient  # noqa: F401
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIMENSIONS = 768
_BATCH_SIZE = 250  # Vertex AI hard limit per request


def _init_vertexai(project_id: str, region: str) -> None:
    """Initialise Vertex AI SDK — idempotent."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        try:
            from config.settings import settings
            if settings.google_application_credentials:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
                    settings.google_application_credentials
                )
        except Exception:
            pass
    vertexai.init(project=project_id, location=region)


@lru_cache(maxsize=1)
def _load_model() -> TextEmbeddingModel:
    return TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


def skill_text(skill_name: str, category: str) -> str:
    """
    Canonical text representation of a skill for embedding.
    Consistent format ensures comparable vector space positions.
    """
    return f"{skill_name} ({category} skill)"


class VertexEmbedder:
    """
    Embeds text strings using Vertex AI text-embedding-004.

    Parameters
    ----------
    project_id:
        GCP project ID.
    region:
        Vertex AI region (default: us-central1).
    task_type:
        'RETRIEVAL_DOCUMENT' for skills stored in the catalog.
        'RETRIEVAL_QUERY' for search queries (gap analysis).
    """

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        self.project_id = project_id
        self.region = region
        self.task_type = task_type
        _init_vertexai(project_id, region)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = _load_model()
        inputs = [TextEmbeddingInput(t, self.task_type) for t in texts]
        results = model.get_embeddings(inputs, output_dimensionality=EMBEDDING_DIMENSIONS)
        return [r.values for r in results]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts. Handles batching automatically.

        Returns
        -------
        list of float vectors, same order as input.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            logger.debug(f"Embedding batch {i // _BATCH_SIZE + 1} ({len(batch)} texts)")
            vectors = self._embed_batch(batch)
            all_vectors.extend(vectors)

        logger.info(f"Embedded {len(all_vectors)} texts via {EMBEDDING_MODEL}")
        return all_vectors

    def embed_skills(
        self, skills: list[tuple[str, str]]
    ) -> list[tuple[str, str, list[float]]]:
        """
        Embed a list of (skill_name, category) pairs.

        Returns
        -------
        list of (skill_name, category, vector)
        """
        texts = [skill_text(name, cat) for name, cat in skills]
        vectors = self.embed(texts)
        return [(name, cat, vec) for (name, cat), vec in zip(skills, vectors)]