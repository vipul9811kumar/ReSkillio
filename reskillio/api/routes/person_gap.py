"""POST /person-gap — personalised gap analysis."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Form, HTTPException, status
from loguru import logger

from reskillio.models.person_gap import PersonGapResult

router = APIRouter(tags=["person_gap"])


def _require_gcp():
    from config.settings import settings
    if not settings.gcp_project_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCP not configured — set GCP_PROJECT_ID",
        )
    return settings


@router.post(
    "/person-gap",
    response_model=PersonGapResult,
    status_code=status.HTTP_200_OK,
    summary="Generate a personalised gap analysis tied to intake context",
    description=(
        "Reads the candidate's skill profile and intake answers (loved_aspects, want_next) "
        "to produce a gap analysis grounded in their actual stated goals. "
        "Falls back to trait profile if intake is not completed."
    ),
)
async def person_gap(
    candidate_id:       Annotated[str,           Form()] = ...,
    archetype:          Annotated[Optional[str], Form()] = None,
    identity_statement: Annotated[Optional[str], Form()] = None,
    industry:           Annotated[Optional[str], Form()] = None,
    ideal_stage:        Annotated[Optional[str], Form()] = None,
    work_values:        Annotated[Optional[str], Form(description="Comma-separated")] = None,
    skills_csv:         Annotated[Optional[str], Form(description="Comma-separated skill names — fallback if BQ profile is empty")] = None,
    # intake fields — pulled from BigQuery if not passed directly
    loved_aspects:      Annotated[Optional[str], Form()] = None,
    want_next:          Annotated[Optional[str], Form()] = None,
    open_to_fractional: Annotated[bool,          Form()] = False,
    engagement_format:  Annotated[Optional[str], Form()] = None,
) -> PersonGapResult:
    settings = _require_gcp()

    # If intake fields not passed, try to read from BigQuery
    _loved = (loved_aspects or "").strip()
    _want  = (want_next or "").strip()

    if not _loved and not _want:
        try:
            from reskillio.intake.intake_store import IntakeStore
            store = IntakeStore(project_id=settings.gcp_project_id)
            intake_profile = store.get_profile(candidate_id)
            if intake_profile:
                _loved            = intake_profile.loved_aspects or ""
                _want             = intake_profile.want_next or ""
                open_to_fractional = intake_profile.open_to_fractional
                engagement_format  = intake_profile.engagement_format or ""
                logger.info(f"[person_gap] Loaded intake profile from BQ for {candidate_id}")
        except Exception as exc:
            logger.warning(f"[person_gap] Intake BQ read failed: {exc}")

    wv = [v.strip() for v in (work_values or "").split(",") if v.strip()]

    from reskillio.pipelines.person_gap_pipeline import run_person_gap

    try:
        fallback_skills = [s.strip() for s in (skills_csv or "").split(",") if s.strip()]

        result = run_person_gap(
            candidate_id=candidate_id,
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
            archetype=archetype or "Operator",
            identity_statement=identity_statement or "",
            industry=industry or "your field",
            ideal_stage=ideal_stage or "Growth-stage",
            work_values=wv or None,
            loved_aspects=_loved,
            want_next=_want,
            open_to_fractional=open_to_fractional,
            engagement_format=engagement_format or "",
            fallback_skills=fallback_skills,
        )
    except Exception as exc:
        logger.exception(f"[person_gap] Pipeline error: {exc}")
        raise HTTPException(500, f"Person gap failed: {exc}") from exc

    return result
