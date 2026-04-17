"""
F15 — Retraining pipeline CLI entrypoint.

Called by Cloud Build when taxonomy.json changes in GCS.

    python scripts/retrain.py \\
        --taxonomy-gcs gs://reskillio-dev-2026-models/taxonomy/taxonomy.json \\
        --project reskillio-dev-2026 \\
        --bucket  reskillio-dev-2026-models \\
        --region  us-central1

Exit codes
----------
0  Evaluation passed gate, new version registered in Vertex AI.
1  Evaluation failed gate (F1 < threshold) — Cloud Build marks build as FAILURE.
2  Unexpected error (auth, GCS unavailable, etc.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from reskillio.models.registry import F1_GATE_THRESHOLD
from reskillio.registry.retrainer import (
    load_taxonomy_from_file,
    load_taxonomy_from_gcs,
    run_retraining,
)


def _print_summary(result) -> None:
    ev = result.evaluation
    print()
    print("=" * 60)
    print("  ReSkillio Retraining Pipeline — Summary")
    print("=" * 60)
    print(f"  Status       : {'PROMOTED ✓' if result.promoted else 'FORCE-REGISTERED ⚠'}")
    print(f"  Version      : {result.version_id}")
    print(f"  F1 Score     : {ev.f1_score:.4f}  (gate ≥ {F1_GATE_THRESHOLD:.2f})")
    print(f"  Precision    : {ev.precision:.4f}")
    print(f"  Recall       : {ev.recall:.4f}")
    print(f"  TP / FP / FN : {ev.true_positives} / {ev.false_positives} / {ev.false_negatives}")
    print(f"  Taxonomy size: {ev.taxonomy_size}")
    print(f"  spaCy model  : {ev.spacy_model}")
    print(f"  GCS artifact : {result.gcs_artifact_uri}")
    print(f"  Vertex AI    : {result.model_resource_name}")
    print("=" * 60)
    print()

    # Print per-example table
    print(f"  {'Example':<52} {'F1':>6}  {'TP':>3} {'FP':>3} {'FN':>3}")
    print("  " + "-" * 68)
    for ex in ev.per_example:
        icon = "✓" if ex.f1 >= 0.80 else "△"
        print(
            f"  {icon} {ex.text_preview:<50} "
            f"{ex.f1:>6.3f}  {len(ex.true_positives):>3} "
            f"{len(ex.false_positives):>3} {len(ex.false_negatives):>3}"
        )
        if ex.false_negatives:
            print(f"      missed : {', '.join(sorted(ex.false_negatives))}")
        if ex.false_positives:
            print(f"      extra  : {', '.join(sorted(ex.false_positives))}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ReSkillio F15 — CI/CD Retraining Pipeline"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--taxonomy-gcs",
        metavar="GCS_URI",
        help="GCS URI of taxonomy.json, e.g. gs://bucket/taxonomy/taxonomy.json",
    )
    source.add_argument(
        "--taxonomy-file",
        metavar="PATH",
        help="Local path to taxonomy.json (for testing without GCS)",
    )

    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--bucket",  required=True, help="GCS bucket for model artifacts")
    parser.add_argument("--region",  default="us-central1", help="GCP region")
    parser.add_argument("--spacy-model", default="en_core_web_lg", help="spaCy model name")
    parser.add_argument("--force", action="store_true",
                        help="Register even if F1 fails gate (debug only)")
    parser.add_argument("--json-output", action="store_true",
                        help="Print result as JSON (for Cloud Build logging)")

    args = parser.parse_args()

    try:
        # 1. Load taxonomy
        if args.taxonomy_gcs:
            taxonomy = load_taxonomy_from_gcs(args.taxonomy_gcs, args.project)
        else:
            taxonomy = load_taxonomy_from_file(args.taxonomy_file)

        logger.info(f"Taxonomy loaded: {len(taxonomy)} skills")

        # 2. Run retraining pipeline
        result = run_retraining(
            taxonomy=taxonomy,
            project_id=args.project,
            bucket=args.bucket,
            region=args.region,
            spacy_model=args.spacy_model,
            force=args.force,
        )

        if args.json_output:
            print(json.dumps({
                "status":             "promoted" if result.promoted else "force_registered",
                "version_id":         result.version_id,
                "f1_score":           result.evaluation.f1_score,
                "taxonomy_size":      result.evaluation.taxonomy_size,
                "gcs_artifact_uri":   result.gcs_artifact_uri,
                "model_resource_name":result.model_resource_name,
            }, indent=2))
        else:
            _print_summary(result)

        return 0

    except ValueError as exc:
        # Gate failure — Cloud Build should mark as FAILURE
        logger.error(f"Gate check failed: {exc}")
        print(f"\n  ✗ GATE FAILED: {exc}\n", file=sys.stderr)
        return 1

    except Exception as exc:
        logger.exception(f"Unexpected error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
