"""
Job description parser.

Responsibilities:
  1. Detect seniority level from title + body text.
  2. Detect industry from keywords.
  3. Split body into required / preferred sections.
  4. Feed each section to the skill extractor with requirement label.
"""

from __future__ import annotations

import re

from reskillio.models.jd import Industry, RequirementLevel, SeniorityLevel

# ---------------------------------------------------------------------------
# Seniority detection
# ---------------------------------------------------------------------------

_SENIORITY_PATTERNS: list[tuple[re.Pattern[str], SeniorityLevel]] = [
    (re.compile(r"\b(staff|principal|distinguished|fellow)\b", re.I), SeniorityLevel.STAFF),
    (re.compile(r"\b(engineering\s+manager|em|head\s+of|director)\b", re.I), SeniorityLevel.MANAGER),
    (re.compile(r"\b(tech\s+lead|technical\s+lead|team\s+lead|lead\s+engineer)\b", re.I), SeniorityLevel.LEAD),
    (re.compile(r"\b(senior|sr\.?)\b", re.I), SeniorityLevel.SENIOR),
    (re.compile(r"\b(junior|jr\.?|entry.level|associate|graduate|new\s+grad)\b", re.I), SeniorityLevel.JUNIOR),
    (re.compile(r"\b(mid.level|mid\s+level|intermediate|ii|2)\b", re.I), SeniorityLevel.MID),
]

_EXPERIENCE_YEARS = re.compile(r"(\d+)\+?\s*(?:to|-)\s*(\d+)?\s*years?", re.I)


def detect_seniority(title: str, text: str) -> SeniorityLevel:
    """Detect seniority level from job title first, then body text."""
    for pattern, level in _SENIORITY_PATTERNS:
        if pattern.search(title or ""):
            return level

    # Fall back to body text
    for pattern, level in _SENIORITY_PATTERNS:
        if pattern.search(text):
            return level

    # Year-range heuristic
    match = _EXPERIENCE_YEARS.search(text)
    if match:
        years = int(match.group(1))
        if years <= 2:
            return SeniorityLevel.JUNIOR
        elif years <= 4:
            return SeniorityLevel.MID
        else:
            return SeniorityLevel.SENIOR

    return SeniorityLevel.UNKNOWN


# ---------------------------------------------------------------------------
# Industry detection
# ---------------------------------------------------------------------------

_INDUSTRY_KEYWORDS: list[tuple[re.Pattern[str], Industry]] = [
    (re.compile(
        r"\b(machine\s+learning|data\s+science|artificial\s+intelligence|llm|"
        r"nlp|deep\s+learning|mlops|feature\s+engineering|vertex\s+ai|sagemaker)\b", re.I
    ), Industry.DATA_AI),

    (re.compile(
        r"\b(fintech|banking|payments|trading|risk\s+model|compliance|"
        r"financial\s+services|hedge\s+fund|investment|ledger|fraud\s+detection)\b", re.I
    ), Industry.FINTECH),

    (re.compile(
        r"\b(healthcare|healthtech|clinical|ehr|hipaa|hl7|fhir|"
        r"medical\s+device|patient|hospital|pharma|biotech)\b", re.I
    ), Industry.HEALTHTECH),

    (re.compile(
        r"\b(e.commerce|ecommerce|retail|marketplace|cart|fulfilment|"
        r"merchandising|supply\s+chain|logistics|shopify|amazon)\b", re.I
    ), Industry.ECOMMERCE),

    (re.compile(
        r"\b(security|cybersecurity|penetration\s+test|soc|siem|"
        r"vulnerability|firewall|zero\s+trust|threat|cissp|ceh)\b", re.I
    ), Industry.CYBERSECURITY),

    (re.compile(
        r"\b(devops|sre|site\s+reliability|platform\s+engineer|"
        r"infrastructure|terraform|ansible|ci.?cd|kubernetes|helm|gitops)\b", re.I
    ), Industry.CLOUD_DEVOPS),

    (re.compile(
        r"\b(product\s+manager|product\s+management|roadmap|product\s+strategy|"
        r"product\s+owner|go.to.market|okr|user\s+research|product\s+led)\b", re.I
    ), Industry.PRODUCT_MANAGEMENT),
]


def detect_industry(title: str, text: str) -> Industry:
    """Detect industry from job title and description text."""
    combined = f"{title or ''} {text}"
    scores: dict[Industry, int] = {}
    for pattern, industry in _INDUSTRY_KEYWORDS:
        count = len(pattern.findall(combined))
        if count:
            scores[industry] = scores.get(industry, 0) + count

    if not scores:
        return Industry.SOFTWARE_ENGINEERING  # sensible default

    return max(scores, key=lambda k: scores[k])


# ---------------------------------------------------------------------------
# Required / preferred section splitting
# ---------------------------------------------------------------------------

_REQUIRED_HEADINGS = re.compile(
    r"^(requirements?|required\s+skills?|what\s+you.ll\s+need|"
    r"must.have|qualifications?|essential\s+skills?|you\s+have|"
    r"minimum\s+qualifications?|basic\s+qualifications?)[\s:]*$",
    re.I | re.MULTILINE,
)

_PREFERRED_HEADINGS = re.compile(
    r"^(preferred\s+qualifications?|nice.to.have|bonus|desirable|"
    r"preferred\s+skills?|good\s+to\s+have|plus|what\s+would\s+be\s+great|"
    r"preferred\s+experience)[\s:]*$",
    re.I | re.MULTILINE,
)

_ANY_HEADING = re.compile(
    r"^[A-Z][^\n]{0,60}[:\s]*$",
    re.MULTILINE,
)


def split_required_preferred(text: str) -> tuple[str, str]:
    """
    Split JD text into (required_text, preferred_text).

    Falls back to (full_text, "") if no preferred section is found.
    """
    lines = text.splitlines()
    required_parts: list[str] = []
    preferred_parts: list[str] = []
    current: list[str] = required_parts  # default bucket

    found_preferred = False

    for line in lines:
        stripped = line.strip()
        if _PREFERRED_HEADINGS.match(stripped):
            current = preferred_parts
            found_preferred = True
            continue
        if _REQUIRED_HEADINGS.match(stripped):
            current = required_parts
            continue
        current.append(line)

    required_text = "\n".join(required_parts).strip()
    preferred_text = "\n".join(preferred_parts).strip() if found_preferred else ""

    return required_text, preferred_text
