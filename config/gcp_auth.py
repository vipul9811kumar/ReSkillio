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

    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.debug("GOOGLE_APPLICATION_CREDENTIALS already set — skipping bootstrap")
        return

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
