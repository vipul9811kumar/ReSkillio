"""
Top-level pipeline: accepts raw resume/job-description text,
runs skill extraction, and returns a structured ExtractionResult.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from reskillio.models.skill import ExtractionResult
from reskillio.nlp.skill_extractor import SkillExtractor

_extractor: SkillExtractor | None = None


def _get_extractor(model_name: str = "en_core_web_lg") -> SkillExtractor:
    """Return a module-level singleton extractor (lazy init)."""
    global _extractor
    if _extractor is None or _extractor.model_name != model_name:
        logger.info(f"Initialising SkillExtractor with model='{model_name}'")
        _extractor = SkillExtractor(model_name=model_name)
    return _extractor


def _maybe_run_drift_monitor(
    result: ExtractionResult,
    candidate_id: Optional[str],
) -> None:
    """Fire-and-forget drift monitoring. Failures are logged but never raised."""
    try:
        from config.settings import settings
        if not settings.gcp_project_id:
            return
        from reskillio.monitoring.drift_monitor import run_drift_monitor
        run_drift_monitor(
            result=result,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            candidate_id=candidate_id,
        )
    except Exception as exc:
        logger.warning(f"Drift monitor skipped: {exc}")


def run_skill_extraction(
    text: str,
    model_name: str = "en_core_web_lg",
    candidate_id: Optional[str] = None,
) -> ExtractionResult:
    """
    Extract skills from a single piece of text.

    Parameters
    ----------
    text:
        Raw resume or job-description text.
    model_name:
        spaCy model to use.
    candidate_id:
        Optional candidate identifier forwarded to the drift monitor for
        per-candidate metric labelling.

    Returns
    -------
    ExtractionResult
        Structured result with deduplicated, categorised skills.
    """
    if not text or not text.strip():
        logger.warning("run_skill_extraction received empty text")
        return ExtractionResult(input_text=text, skills=[], model_used=model_name)

    extractor = _get_extractor(model_name)
    logger.debug(f"Extracting skills from text ({len(text)} chars)")
    result = extractor.extract(text)
    logger.info(f"Extracted {result.skill_count} skills: {result.unique_skill_names()}")

    _maybe_run_drift_monitor(result, candidate_id=candidate_id)

    return result


def run_skill_extraction_batch(
    texts: list[str],
    model_name: str = "en_core_web_lg",
) -> list[ExtractionResult]:
    """Process a list of texts in one efficient spaCy batch."""
    extractor = _get_extractor(model_name)
    logger.info(f"Batch extracting skills from {len(texts)} documents")
    return extractor.extract_batch(texts)
