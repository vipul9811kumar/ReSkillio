"""
F16 — POST /analyze — full career-rebound orchestration endpoint.

Accepts a resume (PDF upload or raw text) plus a target role and optional
job description, then runs all five pipeline stages in sequence:

  1. extract   — spaCy skill extraction → BigQuery → candidate profile
  2. gap        — gap analysis vs the JD (skipped if jd_text omitted)
  3. industry   — BQML industry match scores
  4. narrative  — Gemini RAG career narrative
  5. pathway    — CrewAI 90-day reskilling plan (opt-in, slow)

This is the single interview-demo endpoint — one call, full output.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from reskillio.models.analyze import AnalysisResult

router = APIRouter(tags=["analyze"])


def _require_gcp():
    from config.settings import settings
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP is not configured — set GCP_PROJECT_ID in .env",
        )
    return settings


def _pdf_to_text(pdf_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pdfplumber."""
    import io
    import pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(p for p in pages if p.strip())


# ---------------------------------------------------------------------------
# POST /analyze  (multipart/form-data)
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalysisResult,
    status_code=status.HTTP_200_OK,
    summary="Full career-rebound analysis",
    description=(
        "Single orchestration endpoint for the interview demo. "
        "Upload a resume PDF **or** paste resume_text, provide a target role, "
        "and optionally a JD. Returns skills, gap score, industry rankings, "
        "Gemini narrative, and (opt-in) a 90-day reskilling roadmap."
    ),
)
async def analyze(
    # ── Resume input (one of the two is required) ────────────────────────
    resume:      Annotated[Optional[UploadFile], File(description="Resume PDF file")] = None,
    resume_text: Annotated[Optional[str],        Form(description="Resume plain text (alternative to PDF upload)")] = None,

    # ── Core params ───────────────────────────────────────────────────────
    candidate_id: Annotated[Optional[str], Form(description="Candidate ID (auto-generated if omitted)")] = None,
    target_role:  Annotated[Optional[str], Form(description="Target job title (optional — ReSkillio will infer best fit if omitted)")] = None,

    # ── Free-text context (optional — surfaces traits the resume suppresses) ─
    context_text: Annotated[Optional[str], Form(description="Candidate's own words: what problems they loved, what their manager would say they're great at")] = None,

    # ── JD (optional — enables gap analysis) ─────────────────────────────
    jd_text:  Annotated[Optional[str], Form(description="Job description text for gap analysis")] = None,
    jd_title: Annotated[Optional[str], Form(description="JD title (used in narrative)")] = None,
    industry: Annotated[Optional[str], Form(description="Target industry (auto-detected if omitted)")] = None,

    # ── Pipeline flags ────────────────────────────────────────────────────
    include_pathway: Annotated[bool, Form(
        description="Include 90-day reskilling pathway (CrewAI, ~45 s). Default false."
    )] = False,
) -> AnalysisResult:
    """
    Full career-rebound analysis in one API call.

    **Resume input**: provide `resume` (PDF file) **or** `resume_text` (plain text).
    If both are provided, the PDF takes precedence.

    **Gap analysis**: only runs when `jd_text` is provided.

    **Pathway**: disabled by default — set `include_pathway=true` to enable the
    CrewAI reskilling roadmap (~45 seconds extra).

    **Stages in response**: each entry in `stages` reports `success`, `duration_ms`,
    and `error` (if any) so callers can see exactly which steps completed.
    """
    settings = _require_gcp()

    # ── Resolve resume text ───────────────────────────────────────────────
    text: str = ""

    if resume is not None:
        try:
            pdf_bytes = await resume.read()
            if not pdf_bytes:
                raise HTTPException(400, "Uploaded file is empty.")
            text = _pdf_to_text(pdf_bytes)
            if not text.strip():
                raise HTTPException(422, "Could not extract text from the uploaded PDF.")
            logger.info(
                f"[analyze] PDF '{resume.filename}' — {len(pdf_bytes):,} bytes "
                f"→ {len(text):,} chars"
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(422, f"PDF parsing failed: {exc}") from exc

    elif resume_text:
        text = resume_text.strip()
        if not text:
            raise HTTPException(400, "resume_text is empty.")
    else:
        raise HTTPException(
            400,
            "Provide either a resume PDF file (resume=) or plain text (resume_text=).",
        )

    # ── Resolve candidate_id and target_role ─────────────────────────────
    cid         = (candidate_id or "").strip() or f"demo-{uuid.uuid4().hex[:8]}"
    role_str    = (target_role  or "").strip() or "Career Transition"

    logger.info(
        f"[analyze] Starting full analysis — "
        f"candidate={cid} role='{role_str}' "
        f"context={'yes' if context_text else 'no'} "
        f"jd={'yes' if jd_text else 'no'} pathway={include_pathway}"
    )

    # ── Run orchestration pipeline ────────────────────────────────────────
    from reskillio.pipelines.analyze_pipeline import run_full_analysis

    try:
        result = run_full_analysis(
            resume_text=text,
            candidate_id=cid,
            target_role=role_str,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            jd_text=jd_text or None,
            jd_title=jd_title or None,
            industry=industry or None,
            include_pathway=include_pathway,
            spacy_model=settings.spacy_model,
            context_text=context_text or None,
        )
    except Exception as exc:
        logger.exception(f"[analyze] Orchestration error: {exc}")
        raise HTTPException(500, f"Analysis failed: {exc}") from exc

    return result
