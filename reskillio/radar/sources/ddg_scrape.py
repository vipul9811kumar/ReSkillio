"""
DuckDuckGo search + real page scraping.

Strategy:
  1. Use site-targeted DDG queries to land on scrapable job boards
     (weworkremotely.com, workingnomads.com, jobs.lever.co, greenhouse.io, etc.)
     rather than relying on DDG to return any page and hoping it's not 403'd.
  2. Fetch each URL → extract full job text with BeautifulSoup.
  3. Fall back to DDG snippet when page fetch fails / returns <200 chars.
"""
from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_SCRAPE_TIMEOUT = 10
_MAX_DESC       = 3000
_USER_AGENT     = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Domains that actively block scrapers — fall back to DDG snippet text
_BLOCKED_DOMAINS = {
    "linkedin.com",
    "glassdoor.com",
    "glassdoor.co.uk",
    "indeed.com",
    "simplyhired.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "reddit.com",
    "wellfound.com",
    "angellist.com",
}

# Scrapable job boards — prefix DDG queries with "site:" to target these
_TARGET_SITES = [
    "weworkremotely.com",
    "workingnomads.com",
    "jobs.lever.co",
    "boards.greenhouse.io",
    "app.dover.com",
    "jobs.ashbyhq.com",
]

_STRIP_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "noscript", "iframe", "svg",
]


def search_and_scrape(query: str, max_results: int = 4) -> list[dict]:
    """
    Run a DDG search for `query`, preferring scrapable job board URLs.

    First tries a site-targeted search against _TARGET_SITES; if that
    yields nothing, falls back to an open DDG search.

    Returns list of raw job dicts: source, url, title, company_name, description.
    """
    results: list[dict] = []

    # Round 1 — site-targeted searches for scrapable boards
    for site in _TARGET_SITES[:3]:
        if len(results) >= max_results:
            break
        site_results = _ddg_search(f"site:{site} {query}", max_results=3)
        results.extend(_scrape_results(site_results, max_per_domain=2))

    # Round 2 — open search fallback if we haven't hit quota
    if len(results) < max_results:
        open_results = _ddg_search(query, max_results=(max_results - len(results)) * 2)
        results.extend(_scrape_results(open_results, max_per_domain=1))

    logger.info(f"[ddg_scrape] '{query}' → {len(results)} usable pages")
    return results[:max_results]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    """Run DDG text search, return raw result dicts."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results)) or []
    except Exception as exc:
        logger.warning(f"[ddg_scrape] DDG failed for '{query}': {exc}")
        return []


def _scrape_results(
    ddg_results: list[dict],
    max_per_domain: int = 1,
) -> list[dict]:
    """Scrape pages from DDG results, respecting domain limits and block list."""
    raw_jobs: list[dict] = []
    domain_counts: dict[str, int] = {}

    for r in ddg_results:
        url    = r.get("href", "")
        domain = _domain(url)

        if not url or not domain:
            continue
        if domain_counts.get(domain, 0) >= max_per_domain:
            continue

        if domain in _BLOCKED_DOMAINS:
            snippet = r.get("body", "")
            if snippet and len(snippet) > 80:
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                raw_jobs.append({
                    "source":       "ddg_snippet",
                    "url":          url,
                    "title":        r.get("title", ""),
                    "company_name": "",
                    "description":  snippet[:_MAX_DESC],
                })
            continue

        page_text = _fetch_page_text(url)
        if len(page_text) < 200:
            page_text = r.get("body", "")

        if page_text:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            raw_jobs.append({
                "source":       "ddg_scrape",
                "url":          url,
                "title":        r.get("title", ""),
                "company_name": "",
                "description":  page_text[:_MAX_DESC],
            })

    return raw_jobs


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _fetch_page_text(url: str) -> str:
    resp = None
    try:
        resp = requests.get(
            url,
            timeout=_SCRAPE_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
        )
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(_STRIP_TAGS):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
        soup.decompose()
        return text
    except Exception as exc:
        logger.debug(f"[ddg_scrape] page fetch failed {url}: {exc}")
        return ""
    finally:
        if resp is not None:
            resp.close()
