"""
BigQuery row insertion for Sandbox mode.

BigQuery Sandbox blocks streaming inserts (insert_rows_json).
load_table_from_json uses a batch Load Job instead — free in Sandbox.
"""

from __future__ import annotations

from loguru import logger


def bq_insert(client, table_id: str, rows: list[dict]) -> None:
    """
    Append rows to a BigQuery table via a Load Job.
    Works in BigQuery Sandbox (streaming inserts do not).
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
