"""
F12 — Vertex AI Model Registry API routes.

Endpoints
---------
POST /registry/evaluate   — run F1 evaluation (no GCP write)
POST /registry/register   — evaluate + gate check + register in Vertex AI
GET  /registry/versions   — list all registered model versions
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from config.settings import settings
from reskillio.models.registry import (
    EvaluationResult,
    F1_GATE_THRESHOLD,
    ModelVersion,
    RegisterResult,
)
from reskillio.registry.evaluator import run_evaluation
from reskillio.registry.model_registry import list_model_versions, run_registration

router = APIRouter(prefix="/registry", tags=["model-registry"])

_DEFAULT_BUCKET = lambda project_id: f"{project_id}-reskillio-models"


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    spacy_model: str = Field(
        default="en_core_web_lg",
        description="spaCy model name to evaluate.",
    )


class RegisterRequest(BaseModel):
    spacy_model: str = Field(
        default="en_core_web_lg",
        description="spaCy model name to evaluate and register.",
    )
    bucket_name: Optional[str] = Field(
        default=None,
        description=(
            "GCS bucket for model artifacts. "
            "Defaults to '{project_id}-reskillio-models' (created if absent)."
        ),
    )
    force: bool = Field(
        default=False,
        description=f"Skip the F1 promotion gate (threshold={F1_GATE_THRESHOLD}).",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/evaluate",
    response_model=EvaluationResult,
    status_code=status.HTTP_200_OK,
    summary="Evaluate skill extractor F1 against golden test set",
)
def evaluate(body: EvaluateRequest) -> EvaluationResult:
    """
    Run precision / recall / F1 evaluation against the 12-example golden test set.

    No GCP credentials required — runs entirely locally. Returns per-example
    breakdown and an overall `passes_gate` flag (True when F1 ≥ 0.85).
    """
    logger.info(f"Registry evaluate request: spacy_model={body.spacy_model}")
    try:
        return run_evaluation(spacy_model=body.spacy_model)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Evaluation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evaluation failed.",
        ) from exc


@router.post(
    "/register",
    response_model=RegisterResult,
    status_code=status.HTTP_201_CREATED,
    summary="Register skill extractor in Vertex AI Model Registry",
)
def register(body: RegisterRequest) -> RegisterResult:
    """
    Evaluate the skill extractor, check the promotion gate (F1 ≥ 0.85), then:
    1. Upload the model artifact (taxonomy JSON) to GCS.
    2. Register (or version) the model in Vertex AI Model Registry with
       metadata labels: f1-millipct, taxonomy-size, spacy-model, version-tag.

    Set `force=true` to bypass the gate and register regardless of F1 score.
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )

    bucket = body.bucket_name or _DEFAULT_BUCKET(settings.gcp_project_id)

    logger.info(
        f"Registry register request: model={body.spacy_model} "
        f"bucket={bucket} force={body.force}"
    )

    try:
        result = run_registration(
            project_id=settings.gcp_project_id,
            bucket_name=bucket,
            region=settings.gcp_region,
            spacy_model=body.spacy_model,
            force=body.force,
        )
    except ValueError as exc:
        # Gate failure
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.error(f"Model registration failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model registration failed.",
        ) from exc

    return result


@router.get(
    "/versions",
    response_model=list[ModelVersion],
    status_code=status.HTTP_200_OK,
    summary="List all registered skill-extractor model versions",
)
def versions() -> list[ModelVersion]:
    """
    Return all versions of `reskillio-skill-extractor` registered in
    Vertex AI Model Registry, ordered newest first.
    """
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured.",
        )
    try:
        return list_model_versions(
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
    except Exception as exc:
        logger.error(f"Version listing failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list model versions.",
        ) from exc