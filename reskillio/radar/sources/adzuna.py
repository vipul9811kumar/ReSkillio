"""
Adzuna API source — structured job data, multi-country, free tier (5000 calls/month).
https://developer.adzuna.com/

Requires:  ADZUNA_APP_ID  and  ADZUNA_APP_KEY  in settings / env.
Until those are set this module raises NotConfiguredError on fetch().
"""
from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
_TIMEOUT  = 12
_MAX_DESC = 3000

COUNTRY_CODES = {
    "us": "us", "uk": "gb", "gb": "gb",
    "in": "in", "india": "in",
    "au": "au", "ca": "ca", "de": "de",
}


class NotConfiguredError(Exception):
    pass


def fetch(
    search_term: str,
    country: str = "us",
    results_per_page: int = 20,
) -> list[dict]:
    """
    Fetch jobs from Adzuna matching search_term.

    Raises NotConfiguredError if ADZUNA_APP_ID / ADZUNA_APP_KEY are not set.
    Returns list of raw job dicts with keys:
      source, url, title, company_name, location, salary_min, salary_max, description
    """
    app_id, app_key = _credentials()

    cc = COUNTRY_CODES.get(country.lower(), "us")
    url = f"{_BASE_URL}/{cc}/search/1"

    try:
        resp = requests.get(
            url,
            params={
                "app_id":           app_id,
                "app_key":          app_key,
                "results_per_page": results_per_page,
                "what":             search_term,
                "content-type":     "application/json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.warning(f"[adzuna] API call failed for '{search_term}': {exc}")
        return []

    jobs = resp.json().get("results", [])
    logger.info(f"[adzuna] '{search_term}' → {len(jobs)} results")

    results = []
    for j in jobs:
        desc = _html_to_text(j.get("description", ""))
        company = (j.get("company") or {}).get("display_name", "")
        location = (j.get("location") or {}).get("display_name", "")
        results.append({
            "source":       "adzuna",
            "url":          j.get("redirect_url", ""),
            "title":        j.get("title", ""),
            "company_name": company,
            "location":     location,
            "salary_min":   j.get("salary_min"),
            "salary_max":   j.get("salary_max"),
            "description":  desc[:_MAX_DESC],
            "published_at": j.get("created", ""),
        })

    return results


def _credentials() -> tuple[str, str]:
    try:
        from config.settings import settings
        app_id  = getattr(settings, "adzuna_app_id",  None)
        app_key = getattr(settings, "adzuna_app_key", None)
    except Exception:
        app_id = app_key = None

    import os
    app_id  = app_id  or os.environ.get("ADZUNA_APP_ID")
    app_key = app_key or os.environ.get("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        raise NotConfiguredError(
            "Adzuna credentials not set. Add ADZUNA_APP_ID and ADZUNA_APP_KEY "
            "to settings or environment variables."
        )
    return app_id, app_key


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text)
