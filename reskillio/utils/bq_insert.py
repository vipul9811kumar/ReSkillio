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


def bq_insert(client, table_id: str, rows: list[dict]) -> None:
    """
    Append rows to a BigQuery table via a batch Load Job.
    Works in BigQuery Sandbox when destination table has expiration < 60 days.
    Raises RuntimeError on job failure.
    """
    if not rows:
        return

    from google.cloud import bigquery

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
