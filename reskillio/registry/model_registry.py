"""
F12 — Vertex AI Model Registry client.

Responsibilities
----------------
1. Serialise the skill-extractor artifact (taxonomy JSON) to GCS.
2. Register (or version) the model in Vertex AI Model Registry with
   metadata labels: f1_score, taxonomy_size, spacy_model, registered_at.
3. Enforce the promotion gate: F1 ≥ 0.85 required before registration.
4. List all registered versions of the model.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.models.registry import (
    EvaluationResult,
    F1_GATE_THRESHOLD,
    ModelVersion,
    RegisterResult,
)
from reskillio.nlp.skill_extractor import ALL_SKILLS
from reskillio.registry.evaluator import run_evaluation

# Vertex AI serving container used as placeholder (model is rule-based, not served).
# A pre-built sklearn image satisfies the registry requirement without needing
# a custom container build.
_SERVING_CONTAINER = (
    "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
)
_MODEL_DISPLAY_NAME = "reskillio-skill-extractor"
_ARTIFACT_FILENAME  = "model.json"


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def _gcs_artifact_dir(bucket: str, version_tag: str) -> str:
    """Return the GCS directory URI for a given version tag."""
    bucket = bucket.rstrip("/")
    if not bucket.startswith("gs://"):
        bucket = f"gs://{bucket}"
    return f"{bucket}/skill-extractor/{version_tag}"


def _ensure_bucket(storage_client, bucket_name: str, project_id: str, region: str) -> None:
    from google.cloud.exceptions import Conflict
    try:
        bucket = storage_client.bucket(bucket_name)
        bucket.storage_class = "STANDARD"
        storage_client.create_bucket(bucket, project=project_id, location=region)
        logger.info(f"Created GCS bucket: {bucket_name}")
    except Conflict:
        pass  # bucket already exists


def save_artifact_to_gcs(
    evaluation: EvaluationResult,
    project_id: str,
    bucket_name: str,
    region: str = "us-central1",
    version_tag: Optional[str] = None,
) -> str:
    """
    Serialize the model artifact (taxonomy + eval metadata) to GCS.

    Returns the GCS directory URI where the artifact was written.
    """
    from google.cloud import storage

    if version_tag is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version_tag = f"v{ts}"

    artifact = {
        "model_type":    "rule_based_skill_extractor",
        "spacy_model":   evaluation.spacy_model,
        "taxonomy_size": evaluation.taxonomy_size,
        "taxonomy":      {k: v.value for k, v in ALL_SKILLS.items()},
        "extraction_passes": ["phrase_matcher", "noun_chunk", "entity"],
        "version_tag":   version_tag,
        "f1_score":      evaluation.f1_score,
        "precision":     evaluation.precision,
        "recall":        evaluation.recall,
        "true_positives":  evaluation.true_positives,
        "false_positives": evaluation.false_positives,
        "false_negatives": evaluation.false_negatives,
        "registered_at": evaluation.evaluated_at.isoformat(),
    }

    client = storage.Client(project=project_id)
    _ensure_bucket(client, bucket_name, project_id, region)

    gcs_dir  = _gcs_artifact_dir(bucket_name, version_tag)
    blob_path = f"skill-extractor/{version_tag}/{_ARTIFACT_FILENAME}"
    bucket   = client.bucket(bucket_name)
    blob     = bucket.blob(blob_path)
    blob.upload_from_string(
        json.dumps(artifact, indent=2),
        content_type="application/json",
    )

    logger.info(f"Artifact saved → {gcs_dir}/{_ARTIFACT_FILENAME}")
    return gcs_dir


# ---------------------------------------------------------------------------
# Vertex AI Model Registry helpers
# ---------------------------------------------------------------------------

def _find_parent_model(
    project_id: str,
    region: str,
    display_name: str,
) -> Optional[str]:
    """
    Return the resource name of an existing model with *display_name*, or None.
    Used to create a new version instead of a new model on repeated registrations.
    """
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    models = aiplatform.Model.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )
    return models[0].resource_name if models else None


def _build_labels(evaluation: EvaluationResult, version_tag: str) -> dict[str, str]:
    """
    Build Vertex AI label dict (string:string only, lowercase keys).
    f1_score stored as integer millipercent (e.g. 912 = 91.2%) to stay
    within the label value character set.
    """
    return {
        "f1-millipct":    str(int(round(evaluation.f1_score * 1000))),
        "taxonomy-size":  str(evaluation.taxonomy_size),
        "spacy-model":    evaluation.spacy_model.replace("_", "-"),
        "version-tag":    version_tag,
        "passes-gate":    "true" if evaluation.passes_gate else "false",
        "environment":    os.environ.get("ENVIRONMENT", "development"),
    }


def _build_description(evaluation: EvaluationResult, version_tag: str) -> str:
    return json.dumps({
        "f1_score":      evaluation.f1_score,
        "precision":     evaluation.precision,
        "recall":        evaluation.recall,
        "taxonomy_size": evaluation.taxonomy_size,
        "spacy_model":   evaluation.spacy_model,
        "version_tag":   version_tag,
        "registered_at": evaluation.evaluated_at.isoformat(),
        "gate_threshold": F1_GATE_THRESHOLD,
    }, separators=(",", ":"))


def register_in_vertex(
    evaluation: EvaluationResult,
    gcs_artifact_dir: str,
    project_id: str,
    region: str,
    version_tag: str,
    display_name: str = _MODEL_DISPLAY_NAME,
) -> tuple[str, str]:
    """
    Register (or version) the model in Vertex AI Model Registry.

    Returns (resource_name, version_id).
    """
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    labels      = _build_labels(evaluation, version_tag)
    description = _build_description(evaluation, version_tag)
    parent      = _find_parent_model(project_id, region, display_name)

    kwargs: dict = dict(
        display_name=display_name,
        artifact_uri=gcs_artifact_dir,
        serving_container_image_uri=_SERVING_CONTAINER,
        labels=labels,
        description=description,
        is_default_version=True,
        version_aliases=[version_tag],
        version_description=f"F1={evaluation.f1_score:.4f} taxonomy={evaluation.taxonomy_size}",
    )
    if parent:
        kwargs["parent_model"] = parent
        logger.info(f"Adding version to existing model: {parent}")
    else:
        logger.info(f"Creating new model: {display_name}")

    model = aiplatform.Model.upload(**kwargs)
    model.wait()

    version_id = getattr(model, "version_id", version_tag)
    logger.info(f"Registered → {model.resource_name}  version={version_id}")
    return model.resource_name, str(version_id)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_registration(
    project_id: str,
    bucket_name: str,
    region: str = "us-central1",
    spacy_model: str = "en_core_web_lg",
    force: bool = False,
) -> RegisterResult:
    """
    Full registration flow: evaluate → gate check → GCS upload → Vertex AI registry.

    Parameters
    ----------
    project_id:   GCP project.
    bucket_name:  GCS bucket for model artifacts (created if absent).
    region:       Vertex AI region.
    spacy_model:  spaCy model name to evaluate.
    force:        Skip the promotion gate and register regardless of F1.

    Raises
    ------
    ValueError if F1 < gate threshold and force=False.
    """
    logger.info(f"[registry] Evaluating '{spacy_model}' against golden test set …")
    evaluation = run_evaluation(spacy_model=spacy_model)

    logger.info(
        f"[registry] F1={evaluation.f1_score:.4f}  "
        f"P={evaluation.precision:.4f}  R={evaluation.recall:.4f}  "
        f"gate={'PASS' if evaluation.passes_gate else 'FAIL'}"
    )

    if not evaluation.passes_gate and not force:
        raise ValueError(
            f"Promotion gate failed: F1={evaluation.f1_score:.4f} < "
            f"threshold={F1_GATE_THRESHOLD}. Use force=True to override."
        )

    ts          = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version_tag = f"v{ts}"

    logger.info(f"[registry] Saving artifact to GCS (version={version_tag}) …")
    gcs_dir = save_artifact_to_gcs(
        evaluation=evaluation,
        project_id=project_id,
        bucket_name=bucket_name,
        region=region,
        version_tag=version_tag,
    )

    logger.info("[registry] Registering model in Vertex AI Model Registry …")
    resource_name, version_id = register_in_vertex(
        evaluation=evaluation,
        gcs_artifact_dir=gcs_dir,
        project_id=project_id,
        region=region,
        version_tag=version_tag,
    )

    return RegisterResult(
        model_resource_name=resource_name,
        version_id=version_id,
        gcs_artifact_uri=f"{gcs_dir}/{_ARTIFACT_FILENAME}",
        evaluation=evaluation,
        promoted=True,
        message=(
            f"Model registered as version {version_id} with "
            f"F1={evaluation.f1_score:.4f} (gate={F1_GATE_THRESHOLD})."
        ),
    )


def list_model_versions(
    project_id: str,
    region: str = "us-central1",
    display_name: str = _MODEL_DISPLAY_NAME,
) -> list[ModelVersion]:
    """Return all registered versions of the skill-extractor model."""
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=region)

    models = aiplatform.Model.list(
        filter=f'display_name="{display_name}"',
        order_by="create_time desc",
    )

    versions: list[ModelVersion] = []
    for m in models:
        labels = m.labels or {}
        f1_millipct   = int(labels.get("f1-millipct", "0"))
        taxonomy_size = int(labels.get("taxonomy-size", "0"))
        spacy_model   = labels.get("spacy-model", "").replace("-", "_")
        version_tag   = labels.get("version-tag", "")

        desc_data: dict = {}
        try:
            desc_data = json.loads(m.description or "{}")
        except Exception:
            pass

        versions.append(ModelVersion(
            resource_name=m.resource_name,
            display_name=m.display_name,
            version_id=getattr(m, "version_id", version_tag) or version_tag,
            f1_score=desc_data.get("f1_score", f1_millipct / 1000),
            taxonomy_size=taxonomy_size,
            spacy_model=spacy_model,
            registered_at=desc_data.get("registered_at", ""),
            gcs_artifact=m.artifact_uri or "",
        ))

    return versions