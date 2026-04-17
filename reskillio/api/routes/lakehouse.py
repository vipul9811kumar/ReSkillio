"""
F14 — Medallion lakehouse REST endpoints.

/lakehouse/status                    — row counts for all layers
/lakehouse/ingest/resume             — write raw text to bronze.resume_raw
/lakehouse/ingest/jd/{jd_id}         — write raw JD text to bronze.jd_raw
/lakehouse/promote/candidate         — bronze → silver for one candidate
/lakehouse/promote/jd/{jd_id}        — bronze → silver for one JD
/lakehouse/gold/refresh              — compute all gold tables for a candidate
/lakehouse/gold/match-scores/{id}    — fetch gold match scores
/lakehouse/gold/rankings/{id}        — fetch gold industry rankings
/lakehouse/gold/readiness/{id}       — fetch gold readiness index
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel

from reskillio.models.lakehouse import (
    CandidateReadiness,
    GoldRefreshRequest,
    GoldRefreshResponse,
    IndustryRanking,
    LakehouseStatus,
    MatchScore,
    PromoteResponse,
)

router = APIRouter(prefix="/lakehouse", tags=["lakehouse"])


def _manager():
    try:
        from config.settings import settings
        if not settings.gcp_project_id:
            raise HTTPException(503, "GCP_PROJECT_ID not configured")
        from reskillio.storage.lakehouse import LakehouseManager
        return LakehouseManager(project_id=settings.gcp_project_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Lakehouse unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status", response_model=LakehouseStatus)
def get_status():
    """Row counts for every table across Bronze / Silver / Gold."""
    return _manager().status()


# ---------------------------------------------------------------------------
# Bronze ingestion
# ---------------------------------------------------------------------------

class IngestResumeRequest(BaseModel):
    candidate_id:    str
    raw_text:        str
    filename:        Optional[str] = None
    file_size_bytes: Optional[int] = None
    page_count:      Optional[int] = None


class IngestResumeResponse(BaseModel):
    resume_id:    str
    candidate_id: str
    char_count:   int


@router.post("/ingest/resume", response_model=IngestResumeResponse)
def ingest_resume(req: IngestResumeRequest):
    """Write raw resume text to bronze.resume_raw."""
    mgr = _manager()
    resume_id = mgr.ingest_resume_raw(
        candidate_id=req.candidate_id,
        raw_text=req.raw_text,
        filename=req.filename,
        file_size_bytes=req.file_size_bytes,
        page_count=req.page_count,
    )
    return IngestResumeResponse(
        resume_id=resume_id,
        candidate_id=req.candidate_id,
        char_count=len(req.raw_text),
    )


class IngestJDRequest(BaseModel):
    industry:   str
    raw_text:   str
    title:      Optional[str] = None
    company:    Optional[str] = None
    source_url: Optional[str] = None


@router.post("/ingest/jd/{jd_id}")
def ingest_jd(jd_id: str, req: IngestJDRequest):
    """Write raw JD text to bronze.jd_raw."""
    _manager().ingest_jd_raw(
        jd_id=jd_id,
        industry=req.industry,
        raw_text=req.raw_text,
        title=req.title,
        company=req.company,
        source_url=req.source_url,
    )
    return {"jd_id": jd_id, "char_count": len(req.raw_text), "status": "ingested"}


# ---------------------------------------------------------------------------
# Silver promotion
# ---------------------------------------------------------------------------

class PromoteRequest(BaseModel):
    candidate_id: str


@router.post("/promote/candidate", response_model=PromoteResponse)
def promote_candidate(req: PromoteRequest):
    """
    Validate and promote skill extractions → silver.candidate_skills.

    Applies: confidence < 0.30 → LOW_CONFIDENCE; category UNKNOWN → UNKNOWN.
    """
    mgr  = _manager()
    rows = mgr.promote_candidate(req.candidate_id)
    return PromoteResponse(
        candidate_id=req.candidate_id,
        silver_rows=rows,
        message=f"Promoted {rows} skill rows to Silver for candidate {req.candidate_id}",
    )


@router.post("/promote/jd/{jd_id}")
def promote_jd(jd_id: str):
    """Promote jd_profiles + jd_skills → silver layer."""
    mgr  = _manager()
    rows = mgr.promote_jd(jd_id)
    return {"jd_id": jd_id, "silver_rows": rows, "status": "promoted"}


# ---------------------------------------------------------------------------
# Gold computation
# ---------------------------------------------------------------------------

@router.post("/gold/refresh", response_model=GoldRefreshResponse)
def refresh_gold(req: GoldRefreshRequest):
    """
    Compute all Gold tables for one candidate.

    Steps:
      1. compute_match_scores(candidate_id, jd_ids)
      2. compute_industry_rankings(candidate_id)
      3. compute_readiness(candidate_id)

    Requires the candidate to already be in Silver (call /promote/candidate first).
    """
    mgr = _manager()

    match_scores: list[MatchScore] = []
    if req.jd_ids:
        try:
            match_scores = mgr.compute_match_scores(req.candidate_id, req.jd_ids)
        except Exception as exc:
            logger.warning(f"[lakehouse] match_scores failed: {exc}")

    rankings: list[IndustryRanking] = []
    try:
        rankings = mgr.compute_industry_rankings(req.candidate_id)
    except Exception as exc:
        logger.warning(f"[lakehouse] industry_rankings failed: {exc}")

    readiness: Optional[CandidateReadiness] = None
    try:
        readiness = mgr.compute_readiness(req.candidate_id)
    except Exception as exc:
        logger.warning(f"[lakehouse] readiness failed: {exc}")

    tier = readiness.readiness_tier if readiness else "N/A"
    idx  = f"{readiness.readiness_index:.1f}" if readiness else "N/A"
    return GoldRefreshResponse(
        candidate_id=req.candidate_id,
        match_scores=match_scores,
        industry_rankings=rankings,
        readiness=readiness,
        message=(
            f"Gold refreshed — {len(match_scores)} match scores, "
            f"{len(rankings)} industry rankings, readiness={idx} ({tier})"
        ),
    )


@router.get("/gold/match-scores/{candidate_id}", response_model=list[MatchScore])
def get_match_scores(candidate_id: str):
    """Fetch all stored Gold match scores for a candidate."""
    return _manager().get_match_scores(candidate_id)


@router.get("/gold/rankings/{candidate_id}", response_model=list[IndustryRanking])
def get_industry_rankings(candidate_id: str):
    """Fetch Gold industry rankings for a candidate, ordered by skill coverage."""
    return _manager().get_industry_rankings(candidate_id)


@router.get("/gold/readiness/{candidate_id}", response_model=Optional[CandidateReadiness])
def get_readiness(candidate_id: str):
    """Fetch the latest Gold readiness index for a candidate."""
    result = _manager().get_readiness(candidate_id)
    if result is None:
        raise HTTPException(404, f"No readiness index found for candidate {candidate_id}")
    return result
