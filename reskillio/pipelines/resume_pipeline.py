"""
Resume ingestion pipeline.

PDF bytes → section text → per-section skill extraction → ResumeExtractionResult.
Optionally persists to BigQuery.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from reskillio.ingestion.pdf_reader import extract_text_from_pdf
from reskillio.ingestion.section_parser import parse_sections
from reskillio.models.resume import ResumeExtractionResult, ResumeSection
from reskillio.models.skill import Skill
from reskillio.nlp.skill_extractor import SkillExtractor, _deduplicate
from reskillio.storage.bigquery_store import BigQuerySkillStore
from reskillio.storage.profile_store import CandidateProfileStore

_extractor: SkillExtractor | None = None


def _get_extractor(model_name: str = "en_core_web_lg") -> SkillExtractor:
    global _extractor
    if _extractor is None or _extractor.model_name != model_name:
        logger.info(f"Initialising SkillExtractor with model='{model_name}'")
        _extractor = SkillExtractor(model_name=model_name)
    return _extractor


def run_resume_pipeline(
    source: str | Path | bytes,
    candidate_id: str,
    filename: str = "resume.pdf",
    model_name: str = "en_core_web_lg",
    store: BigQuerySkillStore | None = None,
    profile_store: CandidateProfileStore | None = None,
) -> ResumeExtractionResult:
    """
    Full resume ingestion: PDF → sections → skills → (optional) BigQuery.

    Parameters
    ----------
    source:
        PDF file path or raw bytes.
    candidate_id:
        Unique identifier for the candidate.
    filename:
        Original filename (stored for reference).
    model_name:
        spaCy model to use.
    store:
        If provided, extracted skills are persisted to BigQuery.

    Returns
    -------
    ResumeExtractionResult
    """
    logger.info(f"Resume pipeline started for candidate='{candidate_id}' file='{filename}'")

    # 1. PDF → text
    raw_text = extract_text_from_pdf(source)
    logger.debug(f"Extracted {len(raw_text)} chars from PDF")

    # 2. Text → sections
    parsed = parse_sections(raw_text)
    logger.info(f"Detected {len(parsed)} section(s): {[s[1] for s in parsed]}")

    # 3. Per-section skill extraction
    extractor = _get_extractor(model_name)
    sections: list[ResumeSection] = []

    for section_type, heading, body in parsed:
        if not body.strip():
            continue
        extraction = extractor.extract(body)
        section = ResumeSection(
            section_type=section_type,
            heading=heading,
            raw_text=body,
            skills=extraction.skills,
        )
        sections.append(section)
        logger.debug(f"  [{heading}] → {section.skill_count} skills")

    # 4. Deduplicate across all sections (highest confidence wins)
    all_skills: list[Skill] = []
    for section in sections:
        all_skills.extend(section.skills)
    deduped = _deduplicate(all_skills)

    result = ResumeExtractionResult(
        candidate_id=candidate_id,
        filename=filename,
        sections=sections,
        all_skills=deduped,
        model_used=model_name,
    )

    logger.info(
        f"Resume pipeline complete: {len(sections)} sections, "
        f"{result.total_skill_count} unique skills"
    )

    # 5. Persist to BigQuery
    if store is not None:
        try:
            from reskillio.models.skill import ExtractionResult
            full_result = ExtractionResult(
                input_text=raw_text,
                skills=deduped,
                model_used=model_name,
            )
            store.store_extraction(full_result, candidate_id=candidate_id)
            result.stored = True
        except Exception as exc:
            logger.warning(f"BigQuery write failed (non-fatal): {exc}")

    # 6. Update candidate profile
    if result.stored and profile_store is not None:
        try:
            profile_store.upsert_profile(candidate_id)
        except Exception as exc:
            logger.warning(f"Profile upsert failed (non-fatal): {exc}")

    return result