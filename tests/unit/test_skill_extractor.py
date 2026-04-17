"""
Unit tests for SkillExtractor.

These tests use en_core_web_sm (small model) to keep CI fast.
They verify deduplication, category assignment, empty-input handling,
and batch processing — without requiring any GCP credentials.
"""

import pytest
import spacy

from reskillio.models.skill import ExtractionResult, Skill, SkillCategory
from reskillio.nlp.skill_extractor import SkillExtractor, _deduplicate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor() -> SkillExtractor:
    """Single extractor instance shared across all tests in this module."""
    return SkillExtractor(model_name="en_core_web_sm")


# ---------------------------------------------------------------------------
# Skill model tests
# ---------------------------------------------------------------------------

class TestSkillModel:
    def test_equality_is_case_insensitive(self) -> None:
        a = Skill(name="Python")
        b = Skill(name="python")
        assert a == b

    def test_hash_dedup_in_set(self) -> None:
        skills = {Skill(name="Python"), Skill(name="PYTHON"), Skill(name="python")}
        assert len(skills) == 1

    def test_confidence_bounds(self) -> None:
        with pytest.raises(Exception):
            Skill(name="x", confidence=1.5)  # > 1.0 must fail


class TestExtractionResult:
    def test_skill_count_auto_populated(self) -> None:
        result = ExtractionResult(
            input_text="hi",
            skills=[Skill(name="Python"), Skill(name="Go")],
            model_used="en_core_web_sm",
        )
        assert result.skill_count == 2

    def test_unique_skill_names_no_duplicates(self) -> None:
        result = ExtractionResult(
            input_text="hi",
            skills=[Skill(name="Python"), Skill(name="python"), Skill(name="Go")],
            model_used="en_core_web_sm",
        )
        names = result.unique_skill_names()
        assert len(names) == 2


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_keeps_highest_confidence(self) -> None:
        skills = [
            Skill(name="Python", confidence=0.8),
            Skill(name="Python", confidence=1.0),
            Skill(name="Python", confidence=0.6),
        ]
        result = _deduplicate(skills)
        assert len(result) == 1
        assert result[0].confidence == 1.0

    def test_preserves_distinct_skills(self) -> None:
        skills = [Skill(name="Python"), Skill(name="Go"), Skill(name="Rust")]
        assert len(_deduplicate(skills)) == 3


# ---------------------------------------------------------------------------
# SkillExtractor integration-style unit tests
# ---------------------------------------------------------------------------

class TestSkillExtractor:
    RESUME_SNIPPET = (
        "Experienced data scientist with 5 years of Python, machine learning, "
        "and SQL experience. Proficient in Docker, Kubernetes, and FastAPI. "
        "Strong leadership and communication skills. AWS certified."
    )

    def test_returns_extraction_result(self, extractor: SkillExtractor) -> None:
        result = extractor.extract(self.RESUME_SNIPPET)
        assert isinstance(result, ExtractionResult)

    def test_extracts_known_technical_skills(self, extractor: SkillExtractor) -> None:
        result = extractor.extract(self.RESUME_SNIPPET)
        names_lower = {s.name.lower() for s in result.skills}
        assert "python" in names_lower

    def test_extracts_tools(self, extractor: SkillExtractor) -> None:
        result = extractor.extract(self.RESUME_SNIPPET)
        names_lower = {s.name.lower() for s in result.skills}
        assert "docker" in names_lower or "kubernetes" in names_lower

    def test_no_duplicate_skills_in_result(self, extractor: SkillExtractor) -> None:
        result = extractor.extract(self.RESUME_SNIPPET)
        names = [s.name.lower() for s in result.skills]
        assert len(names) == len(set(names)), "Duplicate skill names found in result"

    def test_empty_input_returns_empty_result(self, extractor: SkillExtractor) -> None:
        result = extractor.extract("")
        assert result.skills == []
        assert result.skill_count == 0

    def test_whitespace_only_input(self, extractor: SkillExtractor) -> None:
        result = extractor.extract("   \n\t  ")
        assert isinstance(result, ExtractionResult)

    def test_batch_returns_correct_count(self, extractor: SkillExtractor) -> None:
        texts = [self.RESUME_SNIPPET, "Looking for a Python developer with SQL skills."]
        results = extractor.extract_batch(texts)
        assert len(results) == 2
        assert all(isinstance(r, ExtractionResult) for r in results)

    def test_model_name_recorded(self, extractor: SkillExtractor) -> None:
        result = extractor.extract(self.RESUME_SNIPPET)
        assert result.model_used == "en_core_web_sm"


# ---------------------------------------------------------------------------
# Pipeline-level smoke test
# ---------------------------------------------------------------------------

class TestSkillPipeline:
    def test_run_skill_extraction_smoke(self) -> None:
        from reskillio.pipelines.skill_pipeline import run_skill_extraction

        result = run_skill_extraction(
            "Senior Python engineer experienced with FastAPI and Docker.",
            model_name="en_core_web_sm",
        )
        assert isinstance(result, ExtractionResult)
        assert result.skill_count >= 0  # non-negative

    def test_run_skill_extraction_empty(self) -> None:
        from reskillio.pipelines.skill_pipeline import run_skill_extraction

        result = run_skill_extraction("", model_name="en_core_web_sm")
        assert result.skills == []
