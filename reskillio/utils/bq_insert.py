"""
BigQuery row insertion for Sandbox mode.

Uses load_table_from_json (batch Load Job) — free in Sandbox as long as
the destination table has expiration < 60 days. Table expirations are patched
to 59 days at startup in config/gcp_auth.py._patch_bq_table_expirations().

DML INSERT is blocked in Sandbox ("DML queries are not allowed in the free tier").
Streaming inserts (insert_rows_json) are also blocked.
"""

from __future__ import annotations

from loguru import logger


def _patch_table_expiration(client, table_id: str, days: int = 59) -> None:
    """
    Ensure a table has an expiration ≤ 59 days (BigQuery Sandbox requirement).
    Called before every load job so existing NEVER-expiry tables are self-healed
    even if the startup patch in config/gcp_auth.py missed them.
    """
    from datetime import datetime, timezone, timedelta
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound

    try:
        table = client.get_table(table_id)
        if table.expires is None:
            table.expires = datetime.now(timezone.utc) + timedelta(days=days)
            client.update_table(table, ["expires"])
            logger.debug(f"[bq_insert] Patched {days}-day expiration on {table_id}")
    except NotFound:
        pass  # table doesn't exist yet; load job will create it with dataset default
    except Exception as exc:
        logger.debug(f"[bq_insert] Could not patch expiration for {table_id}: {exc}")


def bq_insert(client, table_id: str, rows: list[dict]) -> None:
    """
    Append rows to a BigQuery table via a batch Load Job.
    Works in BigQuery Sandbox when destination table has expiration < 60 days.
    Raises RuntimeError on job failure.
    """
    if not rows:
        return

    from google.cloud import bigquery

    _patch_table_expiration(client, table_id)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        autodetect=False,
    )

    job = client.load_table_from_json(rows, table_id, job_config=job_config)
    job.result()

    if job.errors:
        raise RuntimeError(f"BQ insert failed for {table_id}: {job.errors}")

    logger.debug(f"[bq_insert] {len(rows)} rows → {table_id}")
