"""
Shared Gemini client helper — Vertex AI when configured, AI Studio key as fallback.

Set GEMINI_USE_AI_STUDIO=true in .env to skip Vertex entirely and route all
LLM calls through the AI Studio API key (use when Vertex AI is unavailable).
"""

from __future__ import annotations

import warnings

from loguru import logger

_VERTEX_MODEL = "gemini-2.5-flash"
_STUDIO_MODEL = "gemini-2.0-flash"


def _prefer_studio() -> bool:
    try:
        from config.settings import settings
        return settings.gemini_use_ai_studio or not settings.gcp_project_id
    except Exception:
        return True


def _studio_key() -> str:
    try:
        from config.settings import settings
        return settings.gemini_api_key
    except Exception:
        import os
        return os.environ.get("GEMINI_API_KEY", "")


def call_gemini(
    prompt: str,
    system_instruction: str,
    project_id: str,
    region: str = "us-central1",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call Gemini and return response text.

    Routing:
      GEMINI_USE_AI_STUDIO=true (or GCP_PROJECT_ID unset) → AI Studio key, gemini-2.0-flash
      Otherwise → Vertex AI gemini-2.5-flash, falls back to AI Studio on failure.

    Raises RuntimeError if both paths are unavailable.
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")
    from google import genai
    from google.genai import types as gt

    studio_config = gt.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
        candidate_count=1,
    )

    if not _prefer_studio():
        try:
            client = genai.Client(vertexai=True, project=project_id, location=region)
            resp = client.models.generate_content(
                model=_VERTEX_MODEL,
                contents=prompt,
                config=gt.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    thinking_config=gt.ThinkingConfig(thinking_budget=0),
                ),
            )
            return resp.text.strip()
        except Exception as exc:
            logger.warning(f"[gemini] Vertex failed ({type(exc).__name__}) — falling back to AI Studio")

    api_key = _studio_key()
    if not api_key:
        raise RuntimeError(
            "Gemini unavailable — set GEMINI_API_KEY in .env or enable Vertex AI"
        )

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=_STUDIO_MODEL,
        contents=prompt,
        config=studio_config,
    )
    return resp.text.strip()
