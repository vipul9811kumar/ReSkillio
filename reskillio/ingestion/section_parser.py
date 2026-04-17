"""
Resume section parser.

Splits raw resume text into labelled sections by detecting common
headings (case-insensitive).  Unrecognised text before the first
heading is captured as "other".
"""

from __future__ import annotations

import re

from reskillio.models.resume import SectionType

# ---------------------------------------------------------------------------
# Heading → SectionType mapping
# Ordered: more specific phrases before generic ones
# ---------------------------------------------------------------------------

HEADING_PATTERNS: list[tuple[re.Pattern[str], SectionType]] = [
    # Summary / Profile
    (re.compile(
        r"^(professional\s+summary|career\s+summary|executive\s+summary|"
        r"summary\s+of\s+qualifications|profile|objective|about\s+me|summary)$",
        re.IGNORECASE,
    ), SectionType.SUMMARY),

    # Experience
    (re.compile(
        r"^(professional\s+experience|work\s+experience|employment\s+history|"
        r"career\s+history|work\s+history|experience)$",
        re.IGNORECASE,
    ), SectionType.EXPERIENCE),

    # Skills
    (re.compile(
        r"^(technical\s+skills|core\s+competencies|competencies|key\s+skills|"
        r"skills\s+&\s+expertise|skills\s+and\s+expertise|expertise|technologies|"
        r"tools\s+&\s+technologies|skills)$",
        re.IGNORECASE,
    ), SectionType.SKILLS),

    # Education
    (re.compile(
        r"^(academic\s+background|academic\s+qualifications|qualifications|education)$",
        re.IGNORECASE,
    ), SectionType.EDUCATION),

    # Certifications
    (re.compile(
        r"^(professional\s+certifications|licenses\s+&\s+certifications|"
        r"certificates|certifications)$",
        re.IGNORECASE,
    ), SectionType.CERTIFICATIONS),

    # Projects
    (re.compile(
        r"^(personal\s+projects|key\s+projects|notable\s+projects|projects)$",
        re.IGNORECASE,
    ), SectionType.PROJECTS),
]

# A heading line: short (≤ 60 chars), no sentence-ending punctuation mid-line
_HEADING_LINE = re.compile(r"^[A-Z][^\n]{0,58}$")


def parse_sections(text: str) -> list[tuple[SectionType, str, str]]:
    """
    Split resume text into sections.

    Returns
    -------
    list of (SectionType, heading, body_text)
    """
    lines = text.splitlines()
    # Find all heading positions
    heading_indices: list[tuple[int, SectionType, str]] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or not _HEADING_LINE.match(stripped):
            continue
        section_type = _classify_heading(stripped)
        if section_type is not None:
            heading_indices.append((i, section_type, stripped))

    if not heading_indices:
        # No recognisable headings — return whole text as OTHER
        return [(SectionType.OTHER, "resume", text)]

    sections: list[tuple[SectionType, str, str]] = []

    # Text before the first heading
    first_idx = heading_indices[0][0]
    if first_idx > 0:
        pre_text = "\n".join(lines[:first_idx]).strip()
        if pre_text:
            sections.append((SectionType.OTHER, "header", pre_text))

    for pos, (line_no, section_type, heading) in enumerate(heading_indices):
        next_line_no = (
            heading_indices[pos + 1][0]
            if pos + 1 < len(heading_indices)
            else len(lines)
        )
        body = "\n".join(lines[line_no + 1 : next_line_no]).strip()
        sections.append((section_type, heading, body))

    return sections


def _classify_heading(text: str) -> SectionType | None:
    """Return the SectionType for a heading line, or None if unrecognised."""
    cleaned = text.strip().rstrip(":").strip()
    for pattern, section_type in HEADING_PATTERNS:
        if pattern.match(cleaned):
            return section_type
    return None