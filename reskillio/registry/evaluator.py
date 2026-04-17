"""
F12 — Skill extractor evaluation harness.

Computes precision, recall, and F1 against a curated golden test set.
All expected skills are exact lowercase entries from the taxonomy so the
phrase-matcher pass can find them deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import NamedTuple

from reskillio.models.registry import (
    EvaluationResult,
    ExampleEvaluation,
    F1_GATE_THRESHOLD,
)
from reskillio.nlp.skill_extractor import ALL_SKILLS, SkillExtractor


# ---------------------------------------------------------------------------
# Golden test set
# Each `expected` set contains lowercase canonical skill names that appear
# verbatim in the taxonomy (ALL_SKILLS keys).  Texts mention only those
# skills so false-positive rate stays low.
# ---------------------------------------------------------------------------

class _GoldenExample(NamedTuple):
    text:     str
    expected: frozenset[str]   # lowercase exact taxonomy entries


_GOLDEN: list[_GoldenExample] = [
    _GoldenExample(
        text=(
            "Experienced Python developer with 5 years building REST APIs using "
            "FastAPI and Django. Proficient in SQL, PostgreSQL, and Redis. "
            "Deploys with Docker and Kubernetes."
        ),
        expected=frozenset({"python", "rest api", "fastapi", "django", "sql",
                            "postgresql", "redis", "docker", "kubernetes"}),
    ),
    _GoldenExample(
        text=(
            "Machine learning engineer skilled in Deep Learning, TensorFlow, "
            "PyTorch, Pandas, and NumPy. Applies transfer learning techniques "
            "for Computer Vision tasks."
        ),
        expected=frozenset({"machine learning", "deep learning", "tensorflow",
                            "pytorch", "pandas", "numpy", "transfer learning",
                            "computer vision"}),
    ),
    _GoldenExample(
        text=(
            "DevOps engineer with expertise in Kubernetes, Docker, Terraform, "
            "Ansible, Jenkins, and GitHub Actions. AWS certified professional."
        ),
        expected=frozenset({"kubernetes", "docker", "terraform", "ansible",
                            "jenkins", "github actions", "aws certified"}),
    ),
    _GoldenExample(
        text=(
            "Data analyst using SQL, Python, Pandas, and Data Analysis. "
            "Visualises insights with Matplotlib and Seaborn."
        ),
        expected=frozenset({"sql", "python", "pandas", "data analysis",
                            "matplotlib", "seaborn"}),
    ),
    _GoldenExample(
        text=(
            "Full-stack developer building React and Angular frontends backed by "
            "FastAPI microservices. Stores data in MongoDB, PostgreSQL, and Redis."
        ),
        expected=frozenset({"react", "angular", "fastapi", "microservices",
                            "mongodb", "postgresql", "redis"}),
    ),
    _GoldenExample(
        text=(
            "Cloud architect designing solutions on GCP and AWS with BigQuery, "
            "Cloud Run, Kubernetes, and Terraform. Uses Airflow for orchestration."
        ),
        expected=frozenset({"gcp", "aws", "bigquery", "cloud run", "kubernetes",
                            "terraform", "airflow"}),
    ),
    _GoldenExample(
        text=(
            "NLP researcher working with Natural Language Processing, spaCy, NLTK, "
            "and Transformers. Builds Large Language Models using Python and PyTorch."
        ),
        expected=frozenset({"natural language processing", "spacy", "nltk",
                            "transformers", "large language models", "python", "pytorch"}),
    ),
    _GoldenExample(
        text=(
            "Engineering lead demonstrating strong Leadership, Communication, and "
            "Project Management. Runs Agile and Scrum ceremonies and mentors junior engineers."
        ),
        expected=frozenset({"leadership", "communication", "project management",
                            "agile", "scrum", "mentoring"}),
    ),
    _GoldenExample(
        text=(
            "Data engineer building pipelines with Apache Spark, Apache Kafka, "
            "dbt, Airflow, and Python. Stores results in BigQuery and Elasticsearch."
        ),
        expected=frozenset({"apache spark", "apache kafka", "dbt", "airflow",
                            "python", "bigquery", "elasticsearch"}),
    ),
    _GoldenExample(
        text=(
            "Backend developer using Python, Go, and TypeScript. Builds GraphQL "
            "and REST APIs with FastAPI and Flask. Uses Git, GitHub, and Docker "
            "for development workflow."
        ),
        expected=frozenset({"python", "go", "typescript", "graphql", "rest api",
                            "fastapi", "flask", "git", "github", "docker"}),
    ),
    _GoldenExample(
        text=(
            "Security engineer holding CISSP and CKA certifications. Automates "
            "infrastructure with Terraform, Ansible, and GitHub Actions on AWS and Azure."
        ),
        expected=frozenset({"cissp", "cka", "terraform", "ansible",
                            "github actions", "aws", "azure"}),
    ),
    _GoldenExample(
        text=(
            "ML ops engineer integrating LangChain and HuggingFace models into "
            "production pipelines on Vertex AI. Applies Generative AI for "
            "automation using scikit-learn and Python."
        ),
        expected=frozenset({"langchain", "hugging face", "vertex ai",
                            "generative ai", "scikit-learn", "python"}),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text.strip().lower())


def _evaluate_example(
    extractor: SkillExtractor,
    example: _GoldenExample,
) -> ExampleEvaluation:
    result   = extractor.extract(example.text)
    predicted = {_normalise(s.name) for s in result.skills}

    tp = example.expected & predicted
    fp = predicted - example.expected
    fn = example.expected - predicted

    p  = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 1.0
    r  = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 1.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return ExampleEvaluation(
        text_preview=example.text[:80] + ("…" if len(example.text) > 80 else ""),
        expected_skills=sorted(example.expected),
        predicted_skills=sorted(predicted),
        true_positives=sorted(tp),
        false_positives=sorted(fp),
        false_negatives=sorted(fn),
        precision=round(p, 4),
        recall=round(r, 4),
        f1=round(f1, 4),
    )


def run_evaluation(
    spacy_model: str = "en_core_web_lg",
    taxonomy: dict | None = None,
) -> EvaluationResult:
    """
    Run the full golden-set evaluation and return aggregate F1 metrics.

    Uses micro-averaging: pools all TP/FP/FN across examples before computing
    the final precision/recall/F1 (avoids giving equal weight to short and
    long examples).

    Pass *taxonomy* to evaluate a custom taxonomy instead of the built-in
    ALL_SKILLS dict (used by the F15 retraining pipeline).
    """
    extractor = SkillExtractor(
        model_name=spacy_model,
        min_confidence=0.5,
        custom_taxonomy=taxonomy,
    )

    per_example: list[ExampleEvaluation] = []
    total_tp = total_fp = total_fn = 0

    for example in _GOLDEN:
        ev = _evaluate_example(extractor, example)
        per_example.append(ev)
        total_tp += len(ev.true_positives)
        total_fp += len(ev.false_positives)
        total_fn += len(ev.false_negatives)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    f1_score  = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    taxonomy_size = len(taxonomy) if taxonomy is not None else len(ALL_SKILLS)

    return EvaluationResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1_score, 4),
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        taxonomy_size=taxonomy_size,
        spacy_model=spacy_model,
        evaluated_at=datetime.now(timezone.utc),
        passes_gate=f1_score >= F1_GATE_THRESHOLD,
        per_example=per_example,
    )