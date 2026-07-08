"""
JobNormalizer — converts a raw job dict (any source) into an Opportunity model.

Rule-based extraction from structured source fields (title, tags, job_type,
published_at, salary).  No LLM calls — Remotive already provides the data
in structured form, so LLM normalization was burning tokens needlessly.
"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from reskillio.radar.models import (
    Opportunity, EngagementType, HiringSignal, CompanyStage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup maps
# ---------------------------------------------------------------------------

_ENGAGEMENT_MAP = {
    "full_time":  EngagementType.CONSULTING,
    "contract":   EngagementType.CONSULTING,
    "freelance":  EngagementType.FRACTIONAL,
    "part_time":  EngagementType.FRACTIONAL,
    "consultant": EngagementType.CONSULTING,
    "interim":    EngagementType.INTERIM,
    "advisory":   EngagementType.ADVISORY,
    "fractional": EngagementType.FRACTIONAL,
}

_SIGNAL_MAP: dict[str, HiringSignal] = {}  # computed at call time from date

_STAGE_KEYWORDS = {
    CompanyStage.SEED:     ["seed", "pre-seed", "pre-series"],
    CompanyStage.SERIES_A: ["series a", "series-a"],
    CompanyStage.SERIES_B: ["series b", "series-b"],
    CompanyStage.SERIES_C: ["series c", "series-c", "series d", "late stage"],
    CompanyStage.PE_BACKED: ["pe-backed", "private equity", "portfolio company"],
    CompanyStage.ENTERPRISE: [
        "fortune 500", "enterprise", "global company", "publicly traded",
        "nasdaq", "nyse", "10,000", "100,000 employees",
    ],
}

_CULTURE_PATTERNS = [
    "remote-first", "fully remote", "async", "asynchronous",
    "fast-paced", "high growth", "hypergrowth", "move fast",
    "equity", "stock options", "early stage", "building from scratch",
    "post-acquisition", "turnaround", "series", "vc-backed",
    "mission-driven", "impact", "work-life balance", "flexible",
    "international", "global team",
]


class JobNormalizer:

    def __init__(self, project_id: str, region: str = "us-central1") -> None:
        self.project_id = project_id
        self.region     = region

    def normalize(self, raw: dict) -> Optional[Opportunity]:
        """Convert a raw source dict into an Opportunity using rule-based extraction."""
        title = (raw.get("title") or "").strip()
        company = (raw.get("company_name") or raw.get("company") or "").strip()
        if not title:
            return None

        return _dict_to_opportunity(raw)

    def normalize_batch(self, raws: list[dict], max_jobs: int = 25) -> list[Opportunity]:
        """Normalize up to max_jobs raw jobs — purely in-process, no LLM calls."""
        results: list[Opportunity] = []
        for raw in raws[:max_jobs]:
            try:
                opp = self.normalize(raw)
                if opp is not None:
                    results.append(opp)
            except Exception as exc:
                logger.warning(f"[normalizer] item failed: {exc}")
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hiring_signal_from_date(published_at: str) -> HiringSignal:
    if not published_at:
        return HiringSignal.INFERRED
    try:
        # Remotive format: "2025-06-15T10:00:00"
        pub = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - pub).days
        if age_days <= 30:
            return HiringSignal.ACTIVELY_HIRING
        if age_days <= 90:
            return HiringSignal.RECENTLY_POSTED
    except Exception:
        pass
    return HiringSignal.INFERRED


def _company_stage_from_text(text: str) -> CompanyStage:
    low = text.lower()
    for stage, keywords in _STAGE_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            return stage
    return CompanyStage.SMB


def _culture_signals_from_text(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _CULTURE_PATTERNS if p in low][:8]


def _dict_to_opportunity(raw: dict) -> Opportunity:
    title       = (raw.get("title") or "Role").strip()
    company     = (raw.get("company_name") or raw.get("company") or "Unknown").strip()
    description = raw.get("description") or ""
    tags        = raw.get("tags") or []
    job_type    = (raw.get("job_type") or "full_time").lower().replace(" ", "_")
    published   = raw.get("published_at") or raw.get("publication_date") or ""
    salary_str  = raw.get("salary") or ""

    # Skills: prefer tags (already structured), fall back to description extraction
    skills = [s for s in tags if isinstance(s, str)][:15]

    # Salary
    rate_floor, rate_ceiling = _parse_salary_string(salary_str)
    if not rate_floor and raw.get("salary_min"):
        rate_floor   = _safe_float(raw["salary_min"]) / 250
        rate_ceiling = _safe_float(raw.get("salary_max")) / 250 if raw.get("salary_max") else None

    combined_text = f"{title} {description}"

    return Opportunity(
        opportunity_id   = str(uuid.uuid4()),
        company_name     = company,
        company_stage    = _company_stage_from_text(combined_text),
        company_industry = raw.get("industry", ""),
        company_location = raw.get("location", ""),
        role_title       = title,
        engagement_type  = _ENGAGEMENT_MAP.get(job_type, EngagementType.CONSULTING),
        commitment_days_per_week = None,
        rate_floor    = rate_floor,
        rate_ceiling  = rate_ceiling,
        rate_unit     = "year" if (rate_floor and rate_floor > 500) else "day",
        required_skills  = skills,
        culture_signals  = _culture_signals_from_text(combined_text),
        hiring_signal    = _hiring_signal_from_date(published),
        source_url       = raw.get("url"),
        remote_ok        = True,   # Remotive is a remote-jobs board
        location_required= None,
        discovered_at    = datetime.now(timezone.utc),
    )


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_salary_string(s: str) -> tuple[Optional[float], Optional[float]]:
    if not s:
        return None, None
    s = s.lower().replace(",", "").replace(" ", "")

    nums = re.findall(r"\d+(?:\.\d+)?k?", s)
    values = []
    for n in nums:
        try:
            v = float(n.rstrip("k")) * (1000 if n.endswith("k") else 1)
            values.append(v)
        except ValueError:
            continue

    if not values:
        return None, None

    floor   = min(values)
    ceiling = max(values) if len(values) > 1 else None

    if "day" in s or "/d" in s:
        return floor, ceiling
    if "hour" in s or "/h" in s:
        return floor * 8, (ceiling * 8 if ceiling else None)
    if "month" in s or "/m" in s:
        return floor / 21, (ceiling / 21 if ceiling else None)
    if floor > 500:
        return floor / 250, (ceiling / 250 if ceiling else None)
    return floor, ceiling
