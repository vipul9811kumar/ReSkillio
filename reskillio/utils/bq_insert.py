"""
BigQuery row insertion for Sandbox mode.

Uses DML INSERT ... UNION ALL — fully supported in BigQuery Sandbox.
batch load jobs (load_table_from_json) fail when the destination table
has expiration=NEVER (set before Sandbox mode was active); DML has no
such restriction.
"""

from __future__ import annotations

from loguru import logger

_BATCH = 10  # rows per DML statement — keeps queries well under 1 MB


def _lit(value) -> str:
    """Convert a Python value to a BigQuery SQL literal."""
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = (
            value
            .replace("\\", "\\\\")
            .replace("'",  "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        )
        return f"'{escaped}'"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_lit(v) for v in value) + "]"
    return f"'{value}'"


def bq_insert(client, table_id: str, rows: list[dict]) -> None:
    """
    Append rows to a BigQuery table via DML INSERT … UNION ALL.
    Works in BigQuery Sandbox (load jobs and streaming inserts do not).
    Raises RuntimeError on query failure.
    """
    if not rows:
        return

    columns  = list(rows[0].keys())
    col_list = ", ".join(f"`{c}`" for c in columns)

    for i in range(0, len(rows), _BATCH):
        batch = rows[i : i + _BATCH]

        selects = [
            "SELECT " + ", ".join(_lit(row.get(col)) for col in columns)
            for row in batch
        ]
        sql = f"INSERT INTO `{table_id}` ({col_list})\n" + "\nUNION ALL\n".join(selects)

        job = client.query(sql)
        job.result()

    logger.debug(f"[bq_insert] {len(rows)} rows → {table_id}")
