"""
spaCy-based skill extraction pipeline.

Strategy
--------
1. Noun-chunk pass   — extract all noun chunks and filter by a curated skill
                       taxonomy (TECH_SKILLS, SOFT_SKILLS, TOOLS).
2. Entity pass       — keep any PRODUCT / ORG / WORK_OF_ART entities that map
                       to known tools / frameworks.
3. Pattern matcher   — PhraseMatcher against the full taxonomy for anything the
                       dependency parser might have missed.

All three passes are deduped and returned as an ExtractionResult.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc

from reskillio.models.skill import ExtractionResult, Skill, SkillCategory

# ---------------------------------------------------------------------------
# Taxonomy — extend these sets as the product grows
# ---------------------------------------------------------------------------

TECH_SKILLS: set[str] = {
    # Languages
    "python", "java", "javascript", "typescript", "go", "golang", "rust",
    "c++", "c#", "scala", "kotlin", "swift", "ruby", "php", "r", "matlab",
    "sql", "nosql", "bash", "shell scripting",
    # ML / AI
    "machine learning", "deep learning", "natural language processing", "nlp",
    "computer vision", "reinforcement learning", "transfer learning",
    "large language models", "llm", "generative ai",
    # Cloud
    "gcp", "google cloud", "aws", "azure", "cloud computing",
    "vertex ai", "bigquery", "cloud run", "kubernetes", "docker",
    # Data
    "data engineering", "data science", "data analysis", "etl",
    "apache spark", "apache kafka", "airflow", "dbt",
    # Web / API
    "fastapi", "flask", "django", "rest api", "graphql", "microservices",
    "react", "next.js", "vue", "angular",
    # Databases
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "firestore",
}

SOFT_SKILLS: set[str] = {
    "leadership", "communication", "teamwork", "collaboration",
    "problem solving", "critical thinking", "project management",
    "agile", "scrum", "kanban", "stakeholder management",
    "mentoring", "coaching", "presentation", "public speaking",
}

TOOLS: set[str] = {
    "git", "github", "gitlab", "jira", "confluence", "slack",
    "terraform", "ansible", "jenkins", "github actions", "circleci",
    "langchain", "llamaindex", "hugging face", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
    "spacy", "nltk", "transformers",
    "postman", "swagger", "vs code", "pycharm",
}

CERTIFICATIONS: set[str] = {
    "aws certified", "google cloud certified", "gcp professional",
    "azure certified", "pmp", "cissp", "cka", "ckad",
}

# Combine for PhraseMatcher
ALL_SKILLS: dict[str, SkillCategory] = (
    {s: SkillCategory.TECHNICAL for s in TECH_SKILLS}
    | {s: SkillCategory.SOFT for s in SOFT_SKILLS}
    | {s: SkillCategory.TOOL for s in TOOLS}
    | {s: SkillCategory.CERTIFICATION for s in CERTIFICATIONS}
)

# NER entity labels that often carry skill-like information
SKILL_ENTITY_LABELS: frozenset[str] = frozenset({"PRODUCT", "ORG", "WORK_OF_ART"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _category_for(name: str) -> SkillCategory:
    return ALL_SKILLS.get(_normalise(name), SkillCategory.UNKNOWN)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SkillExtractor:
    """
    Extract skills from free-form text using spaCy.

    Parameters
    ----------
    model_name:
        spaCy model to load.  Defaults to ``en_core_web_lg``.
    min_confidence:
        Threshold below which extracted skills are discarded.
    custom_taxonomy:
        Optional replacement for the built-in ALL_SKILLS dict.
        Used by the F15 retraining pipeline to hot-swap a new taxonomy
        downloaded from GCS without touching the source module.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_lg",
        min_confidence: float = 0.5,
        custom_taxonomy: dict[str, "SkillCategory"] | None = None,
    ) -> None:
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._taxonomy = custom_taxonomy if custom_taxonomy is not None else ALL_SKILLS
        self._nlp: Language | None = None
        self._matcher: PhraseMatcher | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation — model loads once, on first use
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._nlp is not None:
            return
        self._nlp = _load_spacy_model(self.model_name)
        self._matcher = _build_matcher(self._nlp, self._taxonomy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> ExtractionResult:
        """Run all extraction passes and return a deduplicated result."""
        self._load()
        assert self._nlp is not None and self._matcher is not None

        doc = self._nlp(text)
        skills: list[Skill] = []

        skills.extend(self._phrase_matcher_pass(doc))
        skills.extend(self._noun_chunk_pass(doc))
        skills.extend(self._entity_pass(doc))

        deduped = _deduplicate(skills)
        filtered = [s for s in deduped if s.confidence >= self.min_confidence]

        return ExtractionResult(
            input_text=text,
            skills=filtered,
            model_used=self.model_name,
        )

    def extract_batch(self, texts: Iterable[str]) -> list[ExtractionResult]:
        """Process multiple texts efficiently using spaCy's pipe."""
        self._load()
        assert self._nlp is not None and self._matcher is not None

        results = []
        for doc in self._nlp.pipe(texts, batch_size=32):
            skills: list[Skill] = []
            skills.extend(self._phrase_matcher_pass(doc))
            skills.extend(self._noun_chunk_pass(doc))
            skills.extend(self._entity_pass(doc))

            deduped = _deduplicate(skills)
            filtered = [s for s in deduped if s.confidence >= self.min_confidence]
            results.append(
                ExtractionResult(
                    input_text=doc.text,
                    skills=filtered,
                    model_used=self.model_name,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Extraction passes (private)
    # ------------------------------------------------------------------

    def _phrase_matcher_pass(self, doc: Doc) -> list[Skill]:
        """High-precision pass: exact match against taxonomy."""
        skills = []
        for _, start, end in self._matcher(doc):  # type: ignore[arg-type]
            span_text = doc[start:end].text
            name = _normalise(span_text)
            category = _category_for(name)
            skills.append(
                Skill(
                    name=span_text.title(),
                    category=category,
                    confidence=1.0,
                    source_text=span_text,
                )
            )
        return skills

    def _noun_chunk_pass(self, doc: Doc) -> list[Skill]:
        """Match noun chunks against taxonomy (handles multi-word skills)."""
        skills = []
        for chunk in doc.noun_chunks:
            name = _normalise(chunk.text)
            if name in self._taxonomy:
                skills.append(
                    Skill(
                        name=chunk.text.title(),
                        category=self._taxonomy[name],
                        confidence=0.9,
                        source_text=chunk.text,
                    )
                )
        return skills

    def _entity_pass(self, doc: Doc) -> list[Skill]:
        """Promote named entities that appear in our taxonomy."""
        skills = []
        for ent in doc.ents:
            if ent.label_ not in SKILL_ENTITY_LABELS:
                continue
            name = _normalise(ent.text)
            if name in self._taxonomy:
                skills.append(
                    Skill(
                        name=ent.text.title(),
                        category=self._taxonomy[name],
                        confidence=0.8,
                        source_text=ent.text,
                    )
                )
        return skills


# ---------------------------------------------------------------------------
# Module-level helpers (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _load_spacy_model(model_name: str) -> Language:
    """Load and cache a spaCy model by name."""
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc


def _build_matcher(
    nlp: Language,
    taxonomy: dict[str, "SkillCategory"] | None = None,
) -> PhraseMatcher:
    """Build a PhraseMatcher from the given taxonomy (defaults to ALL_SKILLS)."""
    skills = taxonomy if taxonomy is not None else ALL_SKILLS
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = list(nlp.tokenizer.pipe(skills.keys()))
    matcher.add("SKILLS", patterns)
    return matcher


def _deduplicate(skills: list[Skill]) -> list[Skill]:
    """Keep the highest-confidence instance of each skill name."""
    seen: dict[str, Skill] = {}
    for skill in skills:
        key = skill.name.lower()
        if key not in seen or skill.confidence > seen[key].confidence:
            seen[key] = skill
    return list(seen.values())
