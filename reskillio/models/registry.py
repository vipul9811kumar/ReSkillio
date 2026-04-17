"""Domain models for Vertex AI Model Registry integration (F12)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

F1_GATE_THRESHOLD = 0.85


class ExampleEvaluation(BaseModel):
    text_preview:    str
    expected_skills: list[str]
    predicted_skills: list[str]
    true_positives:  list[str]
    false_positives: list[str]
    false_negatives: list[str]
    precision:       float
    recall:          float
    f1:              float


class EvaluationResult(BaseModel):
    precision:       float = Field(..., ge=0.0, le=1.0)
    recall:          float = Field(..., ge=0.0, le=1.0)
    f1_score:        float = Field(..., ge=0.0, le=1.0)
    true_positives:  int
    false_positives: int
    false_negatives: int
    taxonomy_size:   int
    spacy_model:     str
    evaluated_at:    datetime
    passes_gate:     bool                   # True when f1_score > F1_GATE_THRESHOLD
    per_example:     list[ExampleEvaluation]


class ModelVersion(BaseModel):
    resource_name:  str        # full Vertex AI resource name
    display_name:   str
    version_id:     str
    f1_score:       float
    taxonomy_size:  int
    spacy_model:    str
    registered_at:  str        # ISO-8601 string from registry label
    gcs_artifact:   str


class RegisterResult(BaseModel):
    model_resource_name: str
    version_id:          str
    gcs_artifact_uri:    str
    evaluation:          EvaluationResult
    promoted:            bool
    message:             str