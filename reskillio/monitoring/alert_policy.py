"""
F13 — Cloud Monitoring alert policy for unknown_skill_rate.

Creates (idempotently) an alert policy that opens an incident whenever
unknown_skill_rate exceeds UNKNOWN_RATE_THRESHOLD for any single data point.

The policy is visible in:
  GCP Console → Monitoring → Alerting → Policies
"""

from __future__ import annotations

from loguru import logger

from reskillio.models.drift import METRIC_UNKNOWN_RATE, UNKNOWN_RATE_THRESHOLD

_POLICY_DISPLAY_NAME = "ReSkillio — Unknown Skill Rate Alert"


def _project_name(project_id: str) -> str:
    return f"projects/{project_id}"


def _find_existing_policy(client, project_id: str) -> str | None:
    """Return resource_name of an existing policy with our display_name, or None."""
    for policy in client.list_alert_policies(name=_project_name(project_id)):
        if policy.display_name == _POLICY_DISPLAY_NAME:
            return policy.name
    return None


def ensure_alert_policy(
    project_id: str,
    threshold:  float = UNKNOWN_RATE_THRESHOLD,
) -> str:
    """
    Create the unknown_skill_rate alert policy if it doesn't exist.

    Returns the policy resource name.
    The policy fires when any single metric point exceeds *threshold*.

    Note: notification channels are not added here.  To receive email/PagerDuty
    alerts, attach a channel via:
        GCP Console → Monitoring → Alerting → Policies → Edit → Notifications
    """
    from google.cloud import monitoring_v3

    client = monitoring_v3.AlertPolicyServiceClient()

    existing = _find_existing_policy(client, project_id)
    if existing:
        logger.info(f"[drift] Alert policy already exists: {existing}")
        return existing

    metric_filter = (
        f'metric.type="{METRIC_UNKNOWN_RATE}" '
        f'AND resource.type="global"'
    )

    from google.protobuf.duration_pb2 import Duration

    condition = monitoring_v3.AlertPolicy.Condition(
        display_name=f"Unknown skill rate > {threshold:.0%}",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter=metric_filter,
            comparison=monitoring_v3.ComparisonType.COMPARISON_GT,
            threshold_value=threshold,
            duration=Duration(seconds=0),      # alert on first point that exceeds threshold
            aggregations=[
                monitoring_v3.Aggregation(
                    alignment_period=Duration(seconds=60),
                    per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MEAN,
                    cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_MEAN,
                )
            ],
            trigger=monitoring_v3.AlertPolicy.Condition.Trigger(count=1),
        ),
    )

    policy = monitoring_v3.AlertPolicy(
        display_name=_POLICY_DISPLAY_NAME,
        conditions=[condition],
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        enabled={"value": True},
        documentation=monitoring_v3.AlertPolicy.Documentation(
            content=(
                f"**ReSkillio skill extractor drift alert.**\n\n"
                f"The fraction of extracted skills classified as UNKNOWN "
                f"has exceeded {threshold:.0%}. This indicates either:\n"
                f"- New terminology appearing in resumes not yet in the taxonomy\n"
                f"- A taxonomy update is needed\n"
                f"- spaCy model degradation\n\n"
                f"**Action:** review recent `drift_metrics` rows in BigQuery "
                f"(`reskillio.drift_metrics`) and update `skill_extractor.py` "
                f"taxonomy if emerging skills are identified.\n\n"
                f"Threshold: `unknown_skill_rate > {threshold}`\n"
                f"Metric: `{METRIC_UNKNOWN_RATE}`"
            ),
            mime_type="text/markdown",
        ),
    )

    created = client.create_alert_policy(
        name=_project_name(project_id),
        alert_policy=policy,
    )
    logger.info(f"[drift] Created alert policy: {created.name}")
    return created.name


def get_policy_status(project_id: str) -> dict:
    """Return basic info about the alert policy, or empty dict if not found."""
    from google.cloud import monitoring_v3

    client   = monitoring_v3.AlertPolicyServiceClient()
    resource = _find_existing_policy(client, project_id)
    if not resource:
        return {"exists": False}

    for policy in client.list_alert_policies(name=_project_name(project_id)):
        if policy.name == resource:
            return {
                "exists":        True,
                "resource_name": policy.name,
                "display_name":  policy.display_name,
                "enabled":       policy.enabled.value if policy.enabled else True,
                "conditions":    len(policy.conditions),
            }
    return {"exists": False}