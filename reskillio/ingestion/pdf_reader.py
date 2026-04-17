"""
PDF → raw text extraction using pdfplumber.

Handles multi-page PDFs, strips artefacts, and returns a single
cleaned string ready for section parsing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from loguru import logger


def extract_text_from_pdf(source: str | Path | bytes) -> str:
    """
    Extract and clean text from a PDF.

    Parameters
    ----------
    source:
        File path (str or Path) or raw PDF bytes (e.g. from an upload).

    Returns
    -------
    str
        Cleaned, concatenated text from all pages.

    Raises
    ------
    ValueError
        If the PDF contains no extractable text (scanned image PDF).
    """
    if isinstance(source, bytes):
        import io
        pdf_file = io.BytesIO(source)
    else:
        pdf_file = Path(source)

    pages: list[str] = []

    with pdfplumber.open(pdf_file) as pdf:
        logger.debug(f"PDF has {len(pdf.pages)} page(s)")
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if text:
                pages.append(text)
            else:
                logger.debug(f"Page {i + 1} yielded no text (image-only?)")

    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned image — OCR support is not yet available."
        )

    raw = "\n\n".join(pages)
    return _clean(raw)


def _clean(text: str) -> str:
    """Remove ligatures, normalise whitespace, drop junk characters."""
    # Common PDF ligature substitutions
    ligatures = {
        "\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff",
        "\ufb03": "ffi", "\ufb04": "ffl", "\u2019": "'",
        "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-",
    }
    for char, replacement in ligatures.items():
        text = text.replace(char, replacement)

    # Collapse runs of blank lines to a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()