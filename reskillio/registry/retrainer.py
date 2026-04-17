"""
F15 — CI/CD retraining pipeline core logic.

Triggered by Cloud Build when taxonomy.json changes in GCS.

Workflow
--------
1. Download taxonomy JSON from GCS (or load from local file for testing)
2. Parse into {skill_name: SkillCategory} dict
3. Rebuild SkillExtractor with the new taxonomy (hot-swap, no source edits)
4. Run golden evaluation (F12 harness with micro-averaged F1)
5. Gate: if F1 < F1_GATE_THRESHOLD raise ValueError (Cloud Build exits non-zero)
6. Save artifact to GCS: gs://{bucket}/skill-extractor/{version_tag}/model.json
7. Register new version in Vertex AI Model Registry

The GCS taxonomy JSON format
-----------------------------
{
  "version": "1.2.0",
  "updated_at": "2026-04-17T00:00:00Z",
  "tech":           ["python", "java", ...],
  "soft":           ["leadership", ...],
  "tools":          ["git", ...],
  "certifications": ["aws certified", ...]
}
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from reskillio.models.registry import F1_GATE_THRESHOLD, RegisterResult
from reskillio.models.skill import SkillCategory


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------

def _parse_taxonomy_json(data: dict) -> dict[str, SkillCategory]:
    """Convert the GCS taxonomy JSON structure into {skill: SkillCategory}."""
    taxonomy: dict[str, SkillCategory] = {}
    for skill in data.get("tech", []):
        taxonomy[skill.strip().lower()] = SkillCategory.TECHNICAL
    for skill in data.get("soft", []):
        taxonomy[skill.strip().lower()] = SkillCategory.SOFT
    for skill in data.get("tools", []):
        taxonomy[skill.strip().lower()] = SkillCategory.TOOL
    for skill in data.get("certifications", []):
        taxonomy[skill.strip().lower()] = SkillCategory.CERTIFICATION
    return taxonomy


def load_taxonomy_from_gcs(gcs_uri: str, project_id: str) -> dict[str, SkillCategory]:
    """
    Download taxonomy.json from GCS and return as {skill_name: SkillCategory}.

    gcs_uri format: gs://bucket-name/path/to/taxonomy.json
    """
    from google.cloud import storage

    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    without_scheme = gcs_uri[len("gs://"):]
    bucket_name, blob_path = without_scheme.split("/", 1)

    client = storage.Client(project=project_id)
    blob   = client.bucket(bucket_name).blob(blob_path)
    raw    = blob.download_as_text()
    data   = json.loads(raw)

    taxonomy = _parse_taxonomy_json(data)
    logger.info(
        f"[retrain] Loaded taxonomy from {gcs_uri}: "
        f"{len(taxonomy)} skills "
        f"(version={data.get('version', 'unknown')})"
    )
    return taxonomy


def load_taxonomy_from_file(path: str) -> dict[str, SkillCategory]:
    """Load taxonomy from a local JSON file — same format as the GCS version."""
    with open(path) as f:
        data = json.load(f)
    taxonomy = _parse_taxonomy_json(data)
    logger.info(f"[retrain] Loaded taxonomy from {path}: {len(taxonomy)} skills")
    return taxonomy


# ---------------------------------------------------------------------------
# Retraining pipeline
# ---------------------------------------------------------------------------

def run_retraining(
    taxonomy:    dict[str, SkillCategory],
    project_id:  str,
    bucket:      str,
    region:      str = "us-central1",
    spacy_model: str = "en_core_web_lg",
    force:       bool = False,
) -> RegisterResult:
    """
    Rebuild SkillExtractor from *taxonomy*, evaluate, and register if gate passes.

    Parameters
    ----------
    taxonomy:
        {skill_name: SkillCategory} dict loaded from GCS or a local file.
    project_id:
        GCP project for Vertex AI + GCS artifact storage.
    bucket:
        GCS bucket name for model artifacts.
    region:
        Vertex AI region.
    spacy_model:
        spaCy model name used for PhraseMatcher.
    force:
        If True, register even when F1 < gate threshold (use only for debugging).

    Returns
    -------
    RegisterResult
        .promoted is True when F1 passed the gate; False if force-registered.

    Raises
    ------
    ValueError
        When F1 fails the gate and force=False (causes Cloud Build to exit 1).
    """
    from reskillio.registry.evaluator import run_evaluation
    from reskillio.registry.model_registry import register_in_vertex, save_artifact_to_gcs

    logger.info(
        f"[retrain] Starting retraining — "
        f"taxonomy_size={len(taxonomy)}, spacy_model={spacy_model}"
    )

    # 1. Evaluate with the new taxonomy
    evaluation = run_evaluation(spacy_model=spacy_model, taxonomy=taxonomy)

    logger.info(
        f"[retrain] Evaluation complete — "
        f"F1={evaluation.f1_score:.4f} "
        f"P={evaluation.precision:.4f} R={evaluation.recall:.4f} "
        f"TP={evaluation.true_positives} FP={evaluation.false_positives} "
        f"FN={evaluation.false_negatives} "
        f"gate={'PASS ✓' if evaluation.passes_gate else 'FAIL ✗'}"
    )

    # 2. Gate check
    if not evaluation.passes_gate:
        msg = (
            f"F1={evaluation.f1_score:.4f} is below the gate threshold "
            f"{F1_GATE_THRESHOLD:.2f}. Taxonomy rejected."
        )
        if not force:
            logger.error(f"[retrain] {msg}")
            raise ValueError(msg)
        logger.warning(f"[retrain] Gate FAILED but --force set. Continuing. {msg}")

    # 3. Save artifact to GCS
    ts          = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version_tag = f"v{ts}"

    gcs_dir = save_artifact_to_gcs(
        evaluation=evaluation,
        project_id=project_id,
        bucket_name=bucket,
        region=region,
        version_tag=version_tag,
    )
    logger.info(f"[retrain] Artifact → {gcs_dir}/model.json")

    # 4. Register in Vertex AI Model Registry
    resource_name, version_id = register_in_vertex(
        evaluation=evaluation,
        gcs_artifact_dir=gcs_dir,
        project_id=project_id,
        region=region,
        version_tag=version_tag,
    )
    logger.info(f"[retrain] Registered → {resource_name} (version {version_id})")

    return RegisterResult(
        model_resource_name=resource_name,
        version_id=version_id,
        gcs_artifact_uri=gcs_dir,
        evaluation=evaluation,
        promoted=evaluation.passes_gate,
        message=(
            f"{'Promoted' if evaluation.passes_gate else 'Force-registered'} "
            f"version {version_id}: "
            f"F1={evaluation.f1_score:.4f}, "
            f"taxonomy_size={len(taxonomy)}"
        ),
    )


# ---------------------------------------------------------------------------
# Taxonomy serialiser (used by export_taxonomy.py)
# ---------------------------------------------------------------------------

def taxonomy_to_json(
    taxonomy: dict[str, SkillCategory],
    version:  str = "1.0.0",
) -> dict:
    """
    Serialise a taxonomy dict back to the canonical JSON format.

    Used by scripts/export_taxonomy.py to bootstrap the GCS file from
    the in-code taxonomy on first deployment.
    """
    result: dict[str, list[str]] = {
        "version":    version,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tech":           [],
        "soft":           [],
        "tools":          [],
        "certifications": [],
    }
    _category_key = {
        SkillCategory.TECHNICAL:    "tech",
        SkillCategory.SOFT:         "soft",
        SkillCategory.TOOL:         "tools",
        SkillCategory.CERTIFICATION:"certifications",
    }
    for skill, category in sorted(taxonomy.items()):
        key = _category_key.get(category)
        if key:
            result[key].append(skill)
    return result
