"""
JobFetcher — orchestrates multiple job sources into a ranked list of Opportunities.

Sources (in priority order):
  1. Remotive  — free API, structured JSON, remote-first roles
  2. DDG scrape — web search + real page scraping, broader coverage
  3. Adzuna    — (wired in automatically once credentials are set in settings)

Flow:
  1. Generate personalised search terms from candidate profile + intake
  2. Query all enabled sources in parallel
  3. Deduplicate by URL
  4. Gemini-normalise each raw posting → Opportunity
  5. Return list[Opportunity] ready for MatchingEngine
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Optional

from reskillio.radar.models import Opportunity
from reskillio.radar.job_normalizer import JobNormalizer
from reskillio.radar.sources import ddg_scrape
from reskillio.radar.sources.remotive import fetch_by_keyword, fetch_ops_categories

logger = logging.getLogger(__name__)

_DDG_LIMIT       = 4     # results per DDG query — scraping is slow
_DDG_MAX_QUERIES = 3     # cap parallel DDG queries
_FETCH_TIMEOUT   = 45    # seconds — per-source thread timeout


def fetch_opportunities(
    top_skills:   list[str],
    top_roles:    list[str],
    identity:     str,
    target_role:  Optional[str],
    industry:     str,
    project_id:   str,
    region:       str = "us-central1",
) -> list[Opportunity]:
    """
    Fetch, deduplicate, and normalise job opportunities for a candidate.

    Parameters
    ----------
    top_skills:  Candidate's top skills (from BQ profile).
    top_roles:   Roles inferred by MarketPulseAgent (highest-demand for this profile).
    identity:    Candidate's work identity (builder / operator / fixer / connector / expert).
    target_role: Explicit target role from intake, if set.
    industry:    Primary industry from intake or analysis.
    project_id:  GCP project for Gemini normalisation calls.
    region:      Vertex AI region.
    """
    search_terms = _build_search_terms(top_skills, top_roles, identity, target_role, industry)
    logger.info(f"[job_fetcher] search terms: {search_terms}")

    raw_jobs = _fetch_all_sources(search_terms)
    logger.info(f"[job_fetcher] {len(raw_jobs)} raw jobs before dedup")

    raw_jobs = _dedup_by_url(raw_jobs)
    logger.info(f"[job_fetcher] {len(raw_jobs)} raw jobs after dedup")

    if not raw_jobs:
        return []

    # Cap at 25 before normalisation — each job needs a Gemini call
    raw_jobs = raw_jobs[:25]

    normalizer   = JobNormalizer(project_id=project_id, region=region)
    opportunities = normalizer.normalize_batch(raw_jobs)
    logger.info(f"[job_fetcher] {len(opportunities)} opportunities normalised")

    return opportunities


# ---------------------------------------------------------------------------
# Search term generation
# ---------------------------------------------------------------------------

def _build_search_terms(
    top_skills:  list[str],
    top_roles:   list[str],
    identity:    str,
    target_role: Optional[str],
    industry:    str,
) -> list[str]:
    """
    Build a prioritised, deduplicated list of search terms.

    Priority:
      1. Explicit target role (highest signal)
      2. Market-inferred top roles
      3. Skill-based searches
      4. Identity-specific patterns
    """
    terms: list[str] = []

    # 1. Target role (from intake)
    if target_role:
        terms.append(target_role)
        terms.append(f"{target_role} remote")

    # 2. Market-inferred roles (from MarketPulseAgent)
    for role in top_roles[:3]:
        terms.append(role)

    # 3. Skill-cluster searches
    if top_skills:
        skill_str = " ".join(top_skills[:3])
        terms.append(f"{skill_str} remote")
        if industry and industry.lower() not in ("operations", "unknown", ""):
            terms.append(f"{top_skills[0]} {industry}")

    # 4. Identity-specific fractional / consulting patterns
    _IDENTITY_TERMS: dict[str, list[str]] = {
        "builder": [
            "fractional VP operations startup",
            "interim COO Series A remote",
        ],
        "operator": [
            "fractional operations director remote",
            "operations consultant scale-up",
        ],
        "fixer": [
            "interim operations manager turnaround",
            "operations consultant post-acquisition",
        ],
        "connector": [
            "fractional Chief of Staff remote",
            "partnerships consultant remote",
        ],
        "expert": [
            "advisory board operations",
            "strategic advisor consulting remote",
        ],
    }
    terms.extend(_IDENTITY_TERMS.get(identity, ["operations consultant remote"]))

    # Deduplicate while preserving order; cap total
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(t.strip())

    return unique[:8]


# ---------------------------------------------------------------------------
# Parallel source fetching
# ---------------------------------------------------------------------------

def _fetch_all_sources(search_terms: list[str]) -> list[dict]:
    """
    Run Remotive + DDG in parallel threads.
    Adzuna is tried automatically if credentials are present.
    """
    futures_map: dict = {}
    raw_jobs: list[dict] = []

    with ThreadPoolExecutor(max_workers=8) as pool:

        # Remotive — category sweep (most reliable for ops/consulting roles)
        f = pool.submit(fetch_ops_categories)
        futures_map[f] = "remotive:categories"

        # Remotive — keyword searches using short 1-2 word terms
        remotive_keywords = _extract_short_keywords(search_terms)
        for kw in remotive_keywords[:4]:
            f = pool.submit(fetch_by_keyword, kw)
            futures_map[f] = f"remotive:{kw}"

        # DDG — site-targeted scrape against scrapable job boards
        for term in search_terms[:_DDG_MAX_QUERIES]:
            f = pool.submit(ddg_scrape.search_and_scrape, term, _DDG_LIMIT)
            futures_map[f] = f"ddg:{term}"

        # Adzuna — activates automatically once credentials are in settings
        try:
            from reskillio.radar.sources.adzuna import fetch as adzuna_fetch, NotConfiguredError
            for term in search_terms[:3]:
                f = pool.submit(adzuna_fetch, term)
                futures_map[f] = f"adzuna:{term}"
        except Exception:
            pass

        for future in as_completed(futures_map, timeout=_FETCH_TIMEOUT):
            label = futures_map[future]
            try:
                result = future.result(timeout=5)
                logger.info(f"[job_fetcher] {label} → {len(result)} jobs")
                raw_jobs.extend(result)
            except TimeoutError:
                logger.warning(f"[job_fetcher] {label} timed out")
            except Exception as exc:
                logger.warning(f"[job_fetcher] {label} failed: {exc}")

    return raw_jobs


def _extract_short_keywords(search_terms: list[str]) -> list[str]:
    """
    Derive short 1-2 word keywords from full search phrases for Remotive,
    which only returns results for simple terms.
    E.g. 'fractional VP operations startup' → 'operations'
         'Head of Operations' → 'operations'
    """
    priority = ["operations", "management", "consulting", "finance",
                "product", "sales", "project", "strategy", "analyst"]
    found: list[str] = []
    seen:  set[str]  = set()

    for term in search_terms:
        words = term.lower().split()
        for w in words:
            if w in priority and w not in seen:
                seen.add(w)
                found.append(w)

    # Always include "operations" and "management" as base terms
    for base in ("operations", "management"):
        if base not in seen:
            found.append(base)

    return found


def _dedup_by_url(raw_jobs: list[dict]) -> list[dict]:
    """Keep the first occurrence of each URL; also dedup by title+company."""
    seen_urls:   set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict]    = []

    for job in raw_jobs:
        url   = (job.get("url") or "").strip()
        title = (job.get("title") or "").lower().strip()
        co    = (job.get("company_name") or "").lower().strip()
        tc_key = f"{title}|{co}"

        if url and url in seen_urls:
            continue
        if title and tc_key in seen_titles:
            continue

        if url:
            seen_urls.add(url)
        if title and co:
            seen_titles.add(tc_key)

        unique.append(job)

    return unique
