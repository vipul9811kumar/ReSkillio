"""
Remotive.com public API — no auth required.
https://remotive.com/api/remote-jobs

Notes:
  - Multi-word search terms reliably return 0; use single keywords only.
  - The `limit` param is unsupported — API always returns all matches.
  - Better coverage comes from category-based fetching than keyword search.
  - Relevant category slugs for operations/consulting profiles:
      sales-business, project-management, finance, all-others
"""
from __future__ import annotations

import logging
import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_API_URL  = "https://remotive.com/api/remote-jobs"
_TIMEOUT  = 12
_MAX_DESC = 3000

# Categories that surface ops / consulting / management roles
_OPS_CATEGORIES = [
    "sales-business",
    "project-management",
    "finance",
    "all-others",
]


def fetch_by_keyword(keyword: str) -> list[dict]:
    """
    Fetch jobs matching a single short keyword (1–2 words only).
    Returns [] for multi-word phrases — caller should strip to key noun.
    """
    short = " ".join(keyword.strip().split()[:2])
    resp = None
    try:
        resp = requests.get(_API_URL, params={"search": short}, timeout=_TIMEOUT)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
    except Exception as exc:
        logger.warning(f"[remotive] keyword fetch failed for '{short}': {exc}")
        return []
    finally:
        if resp is not None:
            resp.close()

    logger.info(f"[remotive] keyword='{short}' → {len(jobs)} jobs")
    return [_normalise(j) for j in jobs]


def fetch_by_category(slug: str) -> list[dict]:
    """
    Fetch all current remote jobs in a Remotive category.
    More reliable than keyword search for broad role types.
    """
    resp = None
    try:
        resp = requests.get(_API_URL, params={"category": slug}, timeout=_TIMEOUT)
        resp.raise_for_status()
        jobs = resp.json().get("jobs", [])
    except Exception as exc:
        logger.warning(f"[remotive] category fetch failed for '{slug}': {exc}")
        return []
    finally:
        if resp is not None:
            resp.close()

    logger.info(f"[remotive] category='{slug}' → {len(jobs)} jobs")
    return [_normalise(j) for j in jobs]


def fetch_ops_categories() -> list[dict]:
    """
    Fetch all jobs across the four operations-relevant categories.
    Deduplicates by job id before returning.
    """
    seen: set[int] = set()
    results: list[dict] = []

    for slug in _OPS_CATEGORIES:
        for job in fetch_by_category(slug):
            jid = job.get("_remotive_id")
            if jid and jid in seen:
                continue
            if jid:
                seen.add(jid)
            results.append(job)

    logger.info(f"[remotive] ops categories total: {len(results)} jobs")
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(j: dict) -> dict:
    desc_text = _html_to_text(j.get("description", ""))
    return {
        "source":        "remotive",
        "_remotive_id":  j.get("id"),
        "url":           j.get("url", ""),
        "title":         j.get("title", ""),
        "company_name":  j.get("company_name", ""),
        "tags":          j.get("tags", []),
        "job_type":      j.get("job_type", ""),
        "salary":        j.get("salary", ""),
        "description":   desc_text[:_MAX_DESC],
        "published_at":  j.get("publication_date", ""),
    }


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
    soup.decompose()
    return text
