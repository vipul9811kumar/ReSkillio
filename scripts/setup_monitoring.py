"""
F13 — One-time Cloud Monitoring setup script.

Creates:
  1. Custom metric descriptors (unknown_skill_rate, avg_confidence, taxonomy_coverage)
  2. Alert policy: fires when unknown_skill_rate > 20%
  3. BigQuery drift_metrics table

Safe to re-run — all operations are idempotent.

Usage
-----
    python scripts/setup_monitoring.py
    python scripts/setup_monitoring.py --threshold 0.15   # stricter gate
    python scripts/setup_monitoring.py --status-only      # just show current state
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from config.settings import settings
from reskillio.models.drift import (
    METRIC_AVG_CONFIDENCE,
    METRIC_TAXONOMY_COVERAGE,
    METRIC_UNKNOWN_RATE,
    UNKNOWN_RATE_THRESHOLD,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Set up ReSkillio drift monitoring in GCP.")
    p.add_argument(
        "--threshold", type=float, default=UNKNOWN_RATE_THRESHOLD,
        help=f"Unknown-rate alert threshold (default: {UNKNOWN_RATE_THRESHOLD})",
    )
    p.add_argument(
        "--status-only", action="store_true",
        help="Print current state without creating anything",
    )
    return p.parse_args()


def _setup_bq_table(project_id: str) -> None:
    from reskillio.monitoring.drift_store import DriftMetricStore
    store = DriftMetricStore(project_id=project_id)
    store.ensure_table()
    print(f"  [ok] BigQuery table: {store.full_table_id}")


def _setup_metric_descriptors(project_id: str) -> None:
    from google.cloud import monitoring_v3
    from reskillio.monitoring.metrics_writer import _ensure_descriptor, _get_client

    client = _get_client(project_id)
    for metric_type in [METRIC_UNKNOWN_RATE, METRIC_AVG_CONFIDENCE, METRIC_TAXONOMY_COVERAGE]:
        _ensure_descriptor(client, project_id, metric_type)
        print(f"  [ok] Metric descriptor: {metric_type}")


def _setup_alert_policy(project_id: str, threshold: float) -> str:
    from reskillio.monitoring.alert_policy import ensure_alert_policy
    resource_name = ensure_alert_policy(project_id=project_id, threshold=threshold)
    print(f"  [ok] Alert policy: {resource_name}")
    print(
        f"       Condition: unknown_skill_rate > {threshold:.0%}\n"
        f"       View in console:\n"
        f"       https://console.cloud.google.com/monitoring/alerting?project={project_id}"
    )
    return resource_name


def _print_status(project_id: str) -> None:
    from reskillio.monitoring.alert_policy import get_policy_status

    print(f"\n{'─'*60}")
    print(f"  ReSkillio Drift Monitoring — Status")
    print(f"  Project: {project_id}")
    print(f"{'─'*60}")

    policy = get_policy_status(project_id)
    if policy.get("exists"):
        enabled = policy.get("enabled", True)
        print(f"  Alert policy : EXISTS  enabled={enabled}")
        print(f"  Resource     : {policy.get('resource_name', '—')}")
        print(f"  Conditions   : {policy.get('conditions', '—')}")
    else:
        print("  Alert policy : NOT FOUND (run without --status-only to create)")

    print(f"\n  Custom metrics:")
    for m in [METRIC_UNKNOWN_RATE, METRIC_AVG_CONFIDENCE, METRIC_TAXONOMY_COVERAGE]:
        print(f"    {m}")

    print(f"\n  BQ table: {project_id}.reskillio.drift_metrics")
    print(f"{'─'*60}\n")


def main() -> None:
    args = _parse_args()

    if not settings.gcp_project_id:
        print("ERROR: GCP_PROJECT_ID not set. Check your .env file.")
        sys.exit(1)

    project_id = settings.gcp_project_id

    if args.status_only:
        _print_status(project_id)
        return

    print(f"\n  Setting up drift monitoring for project: {project_id}")
    print(f"  Alert threshold: unknown_skill_rate > {args.threshold:.0%}\n")

    try:
        _setup_bq_table(project_id)
    except Exception as exc:
        print(f"  [WARN] BQ table setup failed: {exc}")

    try:
        _setup_metric_descriptors(project_id)
    except Exception as exc:
        print(f"  [WARN] Metric descriptor setup failed: {exc}")

    try:
        _setup_alert_policy(project_id, threshold=args.threshold)
    except Exception as exc:
        print(f"  [WARN] Alert policy setup failed: {exc}")

    print(f"\n  Setup complete.")
    print(
        f"\n  NEXT STEP — attach a notification channel to receive alerts:\n"
        f"  GCP Console → Monitoring → Alerting → "
        f"'ReSkillio — Unknown Skill Rate Alert' → Edit → Notifications\n"
    )


if __name__ == "__main__":
    main()