"""
Bootstrap GCP credentials from Railway environment.

If GOOGLE_SERVICE_ACCOUNT_JSON is set (base64-encoded service account JSON),
decode it to a temp file and point GOOGLE_APPLICATION_CREDENTIALS at it.
This is the recommended pattern for platforms that don't support file mounts.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile

from loguru import logger


def bootstrap_service_account() -> None:
    b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if not b64:
        return

    # Always decode from GOOGLE_SERVICE_ACCOUNT_JSON when it is present —
    # do NOT skip if GOOGLE_APPLICATION_CREDENTIALS is already set, because
    # it might point to a non-existent path (e.g. a placeholder value).
    try:
        json_bytes = base64.b64decode(b64)
        json.loads(json_bytes)  # validate before writing
    except Exception as exc:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON is set but could not be decoded as "
            "base64 JSON. Re-encode with: base64 -w 0 service-account.json"
        ) from exc

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb")
    tmp.write(json_bytes)
    tmp.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name
    logger.info(f"GCP credentials bootstrapped from GOOGLE_SERVICE_ACCOUNT_JSON → {tmp.name}")

    # Patch table expirations for BigQuery Sandbox compatibility
    try:
        from config.settings import settings
        if settings.gcp_project_id:
            _patch_bq_table_expirations(settings.gcp_project_id)
    except Exception as exc:
        logger.warning(f"[bq-sandbox] Expiration patch skipped: {exc}")


def _patch_bq_table_expirations(
    project_id: str, dataset_id: str = "reskillio", days: int = 59
) -> None:
    """
    BigQuery Sandbox requires BOTH dataset default expiration AND table
    expiration to be < 60 days before batch load jobs work.
    Patch dataset first, then each table — all metadata-only operations.
    """
    from datetime import datetime, timezone, timedelta
    from google.cloud import bigquery

    ms_per_day = 24 * 60 * 60 * 1000
    expiry_ms  = days * ms_per_day
    expiry_ts  = datetime.now(timezone.utc) + timedelta(days=days)
    client     = bigquery.Client(project=project_id)

    # 1. Patch dataset default expiration first (Sandbox prerequisite)
    try:
        dataset = client.get_dataset(f"{project_id}.{dataset_id}")
        needs_patch = (
            dataset.default_table_expiration_ms is None
            or dataset.default_table_expiration_ms > expiry_ms
        )
        if needs_patch:
            dataset.default_table_expiration_ms = expiry_ms
            client.update_dataset(dataset, ["default_table_expiration_ms"])
            logger.info(f"[bq-sandbox] Set dataset default expiration to {days} days")
    except Exception as exc:
        logger.warning(f"[bq-sandbox] Could not patch dataset expiration: {exc}")
        return  # table patches will also fail — no point continuing

    # 2. Patch each table that still has expiration=NEVER
    updated = 0
    try:
        tables = list(client.list_tables(f"{project_id}.{dataset_id}"))
    except Exception as exc:
        logger.warning(f"[bq-sandbox] Could not list tables: {exc}")
        return

    for table_ref in tables:
        try:
            table = client.get_table(table_ref)
            if table.expires is None:
                table.expires = expiry_ts
                client.update_table(table, ["expires"])
                updated += 1
        except Exception as exc:
            logger.warning(f"[bq-sandbox] Could not patch {table_ref.table_id}: {exc}")

    if updated:
        logger.info(f"[bq-sandbox] Set 59-day expiration on {updated} tables (Sandbox compat)")
