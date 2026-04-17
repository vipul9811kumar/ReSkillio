"""
Job description ingestion pipeline.

JD text → seniority + industry detection → required/preferred section split
       → skill extraction per section → BQ storage → embedding catalog update.
"""

from __future__ import annotations

import uuid

from loguru import logger

from reskillio.ingestion.jd_parser import (
    detect_industry,
    detect_seniority,
    split_required_preferred,
)
from reskillio.models.jd import Industry, JDExtractionResult, JDSkill, RequirementLevel
from reskillio.models.skill import Skill
from reskillio.nlp.skill_extractor import SkillExtractor, _deduplicate
from reskillio.storage.jd_store import JDStore

_extractor: SkillExtractor | None = None


def _get_extractor(model_name: str = "en_core_web_lg") -> SkillExtractor:
    global _extractor
    if _extractor is None or _extractor.model_name != model_name:
        logger.info(f"Initialising SkillExtractor with model='{model_name}'")
        _extractor = SkillExtractor(model_name=model_name)
    return _extractor


def _to_jd_skills(skills: list[Skill], requirement: RequirementLevel) -> list[JDSkill]:
    return [
        JDSkill(
            name=s.name,
            category=s.category,
            confidence=s.confidence,
            requirement=requirement,
        )
        for s in skills
    ]


def run_jd_pipeline(
    text: str,
    title: str | None = None,
    company: str | None = None,
    industry: Industry | None = None,
    source_url: str | None = None,
    model_name: str = "en_core_web_lg",
    store: JDStore | None = None,
) -> JDExtractionResult:
    """
    Full JD ingestion: text → parse → extract skills → (optional) store.

    Parameters
    ----------
    text:
        Raw job description text.
    title:
        Job title (used for seniority + industry detection).
    company:
        Company name (stored for reference).
    industry:
        If provided, overrides auto-detection.
    source_url:
        URL if JD was fetched from the web.
    model_name:
        spaCy model.
    store:
        If provided, results are persisted to BigQuery.
    """
    jd_id = str(uuid.uuid4())
    logger.info(f"JD pipeline started jd_id={jd_id} title='{title}'")

    # 1. Detect seniority and industry
    seniority = detect_seniority(title or "", text)
    detected_industry = industry or detect_industry(title or "", text)
    logger.info(f"  Seniority={seniority.value}  Industry={detected_industry.value}")

    # 2. Split required / preferred sections
    required_text, preferred_text = split_required_preferred(text)

    # 3. Extract skills per section
    extractor = _get_extractor(model_name)

    req_extraction = extractor.extract(required_text) if required_text else None
    pref_extraction = extractor.extract(preferred_text) if preferred_text else None

    # Deduplicate within each section
    req_skills_raw  = _deduplicate(req_extraction.skills if req_extraction else [])
    pref_skills_raw = _deduplicate(pref_extraction.skills if pref_extraction else [])

    # Remove preferred skills already in required (no duplicates across sections)
    req_names = {s.name.lower() for s in req_skills_raw}
    pref_skills_raw = [s for s in pref_skills_raw if s.name.lower() not in req_names]

    required_skills  = _to_jd_skills(req_skills_raw, RequirementLevel.REQUIRED)
    preferred_skills = _to_jd_skills(pref_skills_raw, RequirementLevel.PREFERRED)

    logger.info(
        f"  Required={len(required_skills)} skills  "
        f"Preferred={len(preferred_skills)} skills"
    )

    result = JDExtractionResult(
        jd_id=jd_id,
        title=title,
        company=company,
        industry=detected_industry,
        seniority=seniority,
        source_url=source_url,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
    )

    # 4. Persist to BigQuery
    if store is not None:
        try:
            store.store_jd(result)
            result.stored = True
        except Exception as exc:
            logger.warning(f"JD BQ write failed (non-fatal): {exc}")

    return result
