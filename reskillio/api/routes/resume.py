"""POST /resume/upload — PDF resume ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from loguru import logger

from reskillio.api.routes.extract import _get_store, _get_profile_store
from reskillio.models.resume import ResumeExtractionResult
from reskillio.pipelines.resume_pipeline import run_resume_pipeline
from reskillio.storage.bigquery_store import BigQuerySkillStore
from reskillio.storage.profile_store import CandidateProfileStore
from config.settings import settings

router = APIRouter(prefix="/resume", tags=["resume"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/upload",
    response_model=ResumeExtractionResult,
    status_code=status.HTTP_200_OK,
)
async def upload_resume(
    file: UploadFile = File(..., description="PDF resume file"),
    candidate_id: str = Form(..., description="Unique identifier for the candidate"),
    store: BigQuerySkillStore | None = Depends(_get_store),
    profile_store: CandidateProfileStore | None = Depends(_get_profile_store),
) -> ResumeExtractionResult:
    """
    Upload a PDF resume and extract skills by section.

    - Parses sections: Summary, Experience, Skills, Education, etc.
    - Extracts and categorises skills per section.
    - Returns section-level breakdown plus deduplicated skill list.
    - Persists to BigQuery when GCP is configured.
    """
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Only PDF files are accepted. Got: {file.content_type}",
        )

    pdf_bytes = await file.read()

    if len(pdf_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds the 10 MB limit.",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info(
        f"Resume upload: candidate='{candidate_id}' "
        f"file='{file.filename}' size={len(pdf_bytes)} bytes"
    )

    try:
        result = run_resume_pipeline(
            source=pdf_bytes,
            candidate_id=candidate_id,
            filename=file.filename or "resume.pdf",
            model_name=settings.spacy_model,
            store=store,
            profile_store=profile_store,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(f"Resume pipeline failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resume processing failed.",
        ) from exc

    return result
