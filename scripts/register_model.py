"""
F12 — CLI script: evaluate and register the skill extractor in Vertex AI.

Usage
-----
# Dry-run: evaluate only (no GCP writes)
python scripts/register_model.py --project-id reskillio-dev-2026 --dry-run

# Full registration (gate enforced: F1 must be ≥ 0.85)
python scripts/register_model.py --project-id reskillio-dev-2026

# Custom bucket
python scripts/register_model.py \
    --project-id reskillio-dev-2026 \
    --bucket my-models-bucket \
    --region us-central1

# Override gate (use with care — CI/CD should never pass --force)
python scripts/register_model.py --project-id reskillio-dev-2026 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the project root is on the path when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from reskillio.models.registry import F1_GATE_THRESHOLD
from reskillio.registry.evaluator import run_evaluation
from reskillio.registry.model_registry import run_registration


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate and register the ReSkillio skill extractor in Vertex AI.",
    )
    p.add_argument("--project-id",   required=True, help="GCP project ID")
    p.add_argument("--region",       default="us-central1", help="Vertex AI region")
    p.add_argument("--bucket",       default=None,
                   help="GCS bucket for artifacts (default: {project_id}-reskillio-models)")
    p.add_argument("--spacy-model",  default="en_core_web_lg",
                   help="spaCy model name (default: en_core_web_lg)")
    p.add_argument("--dry-run",      action="store_true",
                   help="Evaluate only — skip GCS upload and Vertex AI registration")
    p.add_argument("--force",        action="store_true",
                   help="Bypass the F1 promotion gate (not recommended in CI/CD)")
    p.add_argument("--json",         action="store_true",
                   help="Emit result as JSON to stdout (useful for CI/CD pipelines)")
    return p.parse_args()


def _print_evaluation_table(ev) -> None:
    gate_symbol = "✓ PASS" if ev.passes_gate else "✗ FAIL"
    print(f"\n{'─'*60}")
    print(f"  Skill Extractor Evaluation — {ev.spacy_model}")
    print(f"{'─'*60}")
    print(f"  Precision   : {ev.precision:.4f}")
    print(f"  Recall      : {ev.recall:.4f}")
    print(f"  F1 Score    : {ev.f1_score:.4f}")
    print(f"  Gate ({F1_GATE_THRESHOLD:.2f}) : {gate_symbol}")
    print(f"  Taxonomy    : {ev.taxonomy_size} skills")
    print(f"  TP / FP / FN: {ev.true_positives} / {ev.false_positives} / {ev.false_negatives}")
    print(f"{'─'*60}")
    print(f"\n  Per-example breakdown:")
    for i, ex in enumerate(ev.per_example, 1):
        icon = "✓" if ex.f1 >= F1_GATE_THRESHOLD else "△"
        print(f"  {icon} [{i:2d}] F1={ex.f1:.3f}  {ex.text_preview}")
        if ex.false_negatives:
            print(f"       missed: {', '.join(ex.false_negatives)}")
        if ex.false_positives:
            print(f"       extra:  {', '.join(ex.false_positives)}")
    print()


def main() -> None:
    args = _parse_args()
    bucket = args.bucket or f"{args.project_id}-reskillio-models"

    # ------------------------------------------------------------------ #
    # Dry-run mode: evaluate only
    # ------------------------------------------------------------------ #
    if args.dry_run:
        logger.info("Dry-run mode — evaluation only, no GCP writes.")
        ev = run_evaluation(spacy_model=args.spacy_model)

        if args.json:
            print(ev.model_dump_json(indent=2))
            sys.exit(0 if ev.passes_gate else 1)

        _print_evaluation_table(ev)
        if not ev.passes_gate:
            print(f"  Gate FAILED — F1={ev.f1_score:.4f} < {F1_GATE_THRESHOLD}")
            sys.exit(1)
        print("  Gate PASSED — model is eligible for registration.")
        sys.exit(0)

    # ------------------------------------------------------------------ #
    # Full registration
    # ------------------------------------------------------------------ #
    try:
        result = run_registration(
            project_id=args.project_id,
            bucket_name=bucket,
            region=args.region,
            spacy_model=args.spacy_model,
            force=args.force,
        )
    except ValueError as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "promoted": False}))
        else:
            print(f"\n  ERROR: {exc}\n")
        sys.exit(1)

    if args.json:
        print(result.model_dump_json(indent=2))
        sys.exit(0)

    _print_evaluation_table(result.evaluation)
    print(f"  {'─'*56}")
    print(f"  Vertex AI resource : {result.model_resource_name}")
    print(f"  Version ID         : {result.version_id}")
    print(f"  GCS artifact       : {result.gcs_artifact_uri}")
    print(f"  {result.message}")
    print(f"  {'─'*56}\n")


if __name__ == "__main__":
    main()