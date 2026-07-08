"""
Shared LLM client helper.

Routing priority (first available wins):
  1. Groq  — if GROQ_API_KEY is set (llama-3.3-70b-versatile, completely free)
  2. AI Studio — if GEMINI_API_KEY is set and GEMINI_USE_AI_STUDIO=true
  3. Vertex AI  — if GCP project configured and above two are absent

Set GROQ_API_KEY in .env for the fully free path with no Google billing required.
"""

from __future__ import annotations

import warnings

from loguru import logger

_GROQ_MODEL      = "llama-3.3-70b-versatile"
_GROQ_MODEL_FAST = "llama-3.1-8b-instant"   # 500K TPD — fallback when 70B hits daily cap
_VERTEX_MODEL    = "gemini-2.5-flash"
_STUDIO_MODEL    = "gemini-2.0-flash"


def _groq_key() -> str:
    try:
        from config.settings import settings
        return settings.groq_api_key
    except Exception:
        import os
        return os.environ.get("GROQ_API_KEY", "")


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


def _call_groq(
    prompt: str,
    system_instruction: str,
    temperature: float,
    max_tokens: int,
    model: str = _GROQ_MODEL,
) -> str:
    from groq import Groq
    client = Groq(api_key=_groq_key())
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def call_gemini(
    prompt: str,
    system_instruction: str,
    project_id: str,
    region: str = "us-central1",
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> str:
    """
    Call the configured LLM and return response text.

    Routing:
      GROQ_API_KEY set        → Groq llama-3.3-70b-versatile (free, no billing)
      GEMINI_USE_AI_STUDIO=true → Google AI Studio gemini-2.0-flash
      Otherwise               → Vertex AI gemini-2.5-flash → AI Studio fallback
    """
    # 1. Groq — free tier, no Google billing required
    groq_key = _groq_key()
    if groq_key:
        try:
            result = _call_groq(prompt, system_instruction, temperature, max_tokens)
            logger.debug(f"[llm] Groq {_GROQ_MODEL} OK ({len(result)} chars)")
            return result
        except Exception as exc:
            exc_str = str(exc)
            # TPD = tokens per day limit on the 70B model — retry with 8B (500K TPD)
            if "tokens per day" in exc_str or ("rate_limit_exceeded" in exc_str and "TPD" in exc_str):
                logger.warning(f"[llm] Groq 70B daily cap hit — retrying with {_GROQ_MODEL_FAST}")
                try:
                    result = _call_groq(prompt, system_instruction, temperature, max_tokens, model=_GROQ_MODEL_FAST)
                    logger.debug(f"[llm] Groq {_GROQ_MODEL_FAST} OK ({len(result)} chars)")
                    return result
                except Exception as exc2:
                    logger.warning(f"[llm] Groq 8B also failed ({type(exc2).__name__}: {exc2}) — trying Gemini")
            else:
                logger.warning(f"[llm] Groq failed ({type(exc).__name__}: {exc}) — trying Gemini")

    # 2. Google Gemini (AI Studio or Vertex)
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
            logger.warning(f"[llm] Vertex failed ({type(exc).__name__}) — falling back to AI Studio")

    api_key = _studio_key()
    if not api_key:
        raise RuntimeError(
            "No LLM available — set GROQ_API_KEY (free) or GEMINI_API_KEY in .env"
        )

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=_STUDIO_MODEL,
        contents=prompt,
        config=studio_config,
    )
    return resp.text.strip()
