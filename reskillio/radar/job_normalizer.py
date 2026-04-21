"""
JobNormalizer — converts a raw job dict (any source) into an Opportunity model.

Uses Gemini to extract structured fields from free-text posting content.
Falls back gracefully if Gemini is unavailable or the JSON is malformed.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
import warnings
from datetime import datetime, timezone
from typing import Optional

from reskillio.radar.models import (
    Opportunity, EngagementType, HiringSignal, CompanyStage,
)

logger = logging.getLogger(__name__)

_MODEL_VERTEX = "gemini-2.5-flash"
_MODEL_STUDIO = "gemini-2.0-flash"

_SYSTEM = (
    "You extract structured data from job postings. "
    "You respond ONLY with valid JSON — no markdown fences, no explanation. "
    "If a field cannot be determined from the text, use null."
)

_PROMPT = """\
Extract structured fields from this job posting and return ONLY a JSON object.

Job posting:
\"\"\"
{text}
\"\"\"

Return exactly this JSON shape (no extra keys, no markdown):
{{
  "company_name":            "<string or empty>",
  "role_title":              "<job title>",
  "engagement_type":         "<one of: fractional, consulting, advisory, interim, full_time>",
  "required_skills":         ["skill1", "skill2"],
  "culture_signals":         ["phrase1", "phrase2"],
  "company_stage":           "<one of: seed, series_a, series_b, series_c, pe_backed, smb, enterprise>",
  "rate_floor":              null,
  "rate_ceiling":            null,
  "rate_unit":               "<day|month|hour|year>",
  "commitment_days_per_week": null,
  "remote_ok":               true,
  "hiring_signal":           "<one of: actively_hiring, recently_posted, inferred>",
  "location_required":       null
}}

Rules:
- engagement_type: "fractional"=part-time leadership, "consulting"=project work, \
"advisory"=board/advisor, "interim"=temporary full-time, "full_time"=permanent.
- required_skills: concrete technical/domain skills only (Python, Salesforce, SQL, \
supply chain, etc.) — no soft skills ("communication", "teamwork").
- culture_signals: environment descriptors ("fast-paced", "building from scratch", \
"post-acquisition", "remote-first", "equity", etc.).
- company_stage: infer from headcount/funding/language. Use "smb" when unclear for \
an established small business, "enterprise" for large corps.
- rate_floor/rate_ceiling: convert to daily USD. Annual ÷ 250, monthly ÷ 21. \
Null if not mentioned.
- hiring_signal: "actively_hiring" if the posting is live/dated <30 days, \
"recently_posted" if dated 30–90 days, else "inferred".
"""


class JobNormalizer:

    def __init__(self, project_id: str, region: str = "us-central1") -> None:
        self.project_id = project_id
        self.region     = region

    def normalize(self, raw: dict) -> Optional[Opportunity]:
        """
        Convert a raw source dict into an Opportunity.
        Returns None if the posting cannot be parsed into a usable opportunity.
        """
        text = _build_input_text(raw)
        if not text or len(text) < 80:
            return None

        try:
            extracted = self._call_gemini(text)
        except Exception as exc:
            logger.warning(f"[normalizer] Gemini failed for {raw.get('url','?')}: {exc}")
            extracted = {}

        # If Gemini gave us nothing usable, try a light heuristic fallback
        if not extracted.get("role_title"):
            extracted["role_title"] = raw.get("title", "")
        if not extracted.get("company_name"):
            extracted["company_name"] = raw.get("company_name", "")

        # Still no role title → skip
        if not extracted.get("role_title"):
            return None

        # Remotive gives us tags that map well to skills
        if raw.get("tags") and not extracted.get("required_skills"):
            extracted["required_skills"] = raw["tags"][:10]

        return _dict_to_opportunity(extracted, raw)

    def normalize_batch(self, raws: list[dict], max_jobs: int = 25) -> list[Opportunity]:
        """Normalize up to max_jobs raw jobs in parallel; silently drops failures."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        subset = raws[:max_jobs]
        results: list[Opportunity] = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self.normalize, raw): raw for raw in subset}
            for future in as_completed(futures, timeout=50):
                try:
                    opp = future.result(timeout=5)
                    if opp is not None:
                        results.append(opp)
                except Exception as exc:
                    logger.warning(f"[normalizer] batch item failed: {exc}")
        return results

    # ------------------------------------------------------------------

    def _call_gemini(self, text: str) -> dict:
        warnings.filterwarnings("ignore", category=UserWarning, module="vertexai")

        prompt = _PROMPT.format(text=text[:3000])

        try:
            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.region,
            )
            response = client.models.generate_content(
                model=_MODEL_VERTEX,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=800,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return _parse_json(response.text)

        except Exception:
            pass

        # Studio fallback
        try:
            import os
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError("No API key")

            from google import genai
            from google.genai import types as genai_types

            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=_MODEL_STUDIO,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    temperature=0.1,
                    max_output_tokens=800,
                ),
            )
            return _parse_json(response.text)

        except Exception as exc:
            raise RuntimeError(f"Gemini unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_input_text(raw: dict) -> str:
    """Assemble the text fed to Gemini from source fields."""
    parts = []
    if raw.get("title"):
        parts.append(f"Title: {raw['title']}")
    if raw.get("company_name"):
        parts.append(f"Company: {raw['company_name']}")
    if raw.get("salary"):
        parts.append(f"Salary: {raw['salary']}")
    if raw.get("location"):
        parts.append(f"Location: {raw['location']}")
    if raw.get("description"):
        parts.append(raw["description"])
    return " ".join(parts)


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```(?:json)?\s*|\s*```", "", (text or "")).strip()
    try:
        result = json.loads(clean)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        # Try to recover the first JSON object from the text
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


_ENGAGEMENT_MAP = {
    "fractional": EngagementType.FRACTIONAL,
    "consulting":  EngagementType.CONSULTING,
    "advisory":    EngagementType.ADVISORY,
    "interim":     EngagementType.INTERIM,
    "full_time":   EngagementType.CONSULTING,   # treat full-time as consulting for radar
    "contract":    EngagementType.CONSULTING,
    "part_time":   EngagementType.FRACTIONAL,
}

_STAGE_MAP = {
    "seed":       CompanyStage.SEED,
    "series_a":   CompanyStage.SERIES_A,
    "series_b":   CompanyStage.SERIES_B,
    "series_c":   CompanyStage.SERIES_C,
    "pe_backed":  CompanyStage.PE_BACKED,
    "smb":        CompanyStage.SMB,
    "enterprise": CompanyStage.ENTERPRISE,
}

_SIGNAL_MAP = {
    "actively_hiring": HiringSignal.ACTIVELY_HIRING,
    "recently_posted": HiringSignal.RECENTLY_POSTED,
    "inferred":        HiringSignal.INFERRED,
}


def _dict_to_opportunity(d: dict, raw: dict) -> Opportunity:
    rate_floor   = _safe_float(d.get("rate_floor"))
    rate_ceiling = _safe_float(d.get("rate_ceiling"))
    days_pw      = _safe_float(d.get("commitment_days_per_week"))

    # Remotive salary field often contains human-readable strings like "$80k-$120k"
    if not rate_floor and raw.get("salary"):
        rate_floor, rate_ceiling = _parse_salary_string(raw["salary"])

    # Adzuna provides salary_min/salary_max in annual USD
    if not rate_floor and raw.get("salary_min"):
        rate_floor   = _safe_float(raw["salary_min"])  / 250
        rate_ceiling = _safe_float(raw.get("salary_max")) / 250 if raw.get("salary_max") else None

    return Opportunity(
        opportunity_id   = str(uuid.uuid4()),
        company_name     = (d.get("company_name") or raw.get("company_name") or "Unknown").strip(),
        company_stage    = _STAGE_MAP.get(d.get("company_stage", ""), CompanyStage.SMB),
        company_industry = raw.get("industry", ""),
        company_location = d.get("location_required") or raw.get("location", ""),
        role_title       = (d.get("role_title") or raw.get("title", "Role")).strip(),
        engagement_type  = _ENGAGEMENT_MAP.get(
            (d.get("engagement_type") or "").lower(), EngagementType.CONSULTING
        ),
        commitment_days_per_week = days_pw,
        rate_floor    = rate_floor,
        rate_ceiling  = rate_ceiling,
        rate_unit     = d.get("rate_unit") or "day",
        required_skills  = [s for s in (d.get("required_skills") or []) if isinstance(s, str)][:15],
        culture_signals  = [s for s in (d.get("culture_signals")  or []) if isinstance(s, str)][:8],
        hiring_signal    = _SIGNAL_MAP.get(d.get("hiring_signal", ""), HiringSignal.INFERRED),
        source_url       = raw.get("url"),
        remote_ok        = bool(d.get("remote_ok", True)),
        location_required= d.get("location_required"),
        discovered_at    = datetime.now(timezone.utc),
    )


def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _parse_salary_string(s: str) -> tuple[Optional[float], Optional[float]]:
    """
    Parse human-readable salary strings like '$80k-$120k/yr', '£400/day'.
    Returns (floor_daily_usd, ceiling_daily_usd) or (None, None).
    """
    if not s:
        return None, None
    s = s.lower().replace(",", "").replace(" ", "")

    # Extract all numbers (handles "80k", "80000", "80")
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

    # Detect unit and convert to daily
    if "day" in s or "/d" in s:
        return floor, ceiling
    if "hour" in s or "/h" in s:
        return floor * 8, (ceiling * 8 if ceiling else None)
    if "month" in s or "/m" in s:
        return floor / 21, (ceiling / 21 if ceiling else None)
    # Default: annual → daily
    if floor > 500:   # almost certainly annual
        return floor / 250, (ceiling / 250 if ceiling else None)
    return floor, ceiling
