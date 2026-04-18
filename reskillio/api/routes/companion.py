"""
Weekly companion routes.

  POST /companion/checkin              — submit weekly check-in
  GET  /companion/{candidate_id}/digest — get latest digest
  GET  /companion/{candidate_id}/history — full digest history
  POST /companion/trigger-digests      — Cloud Scheduler webhook (internal)
"""

from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from loguru import logger

from reskillio.companion.models import (
    CheckinSubmitRequest, CheckinSubmitResponse, WeeklyCheckin,
)

router = APIRouter(prefix="/companion", tags=["companion"])


def _settings():
    from config.settings import settings
    return settings


def _store():
    from reskillio.companion.checkin_store import CompanionStore
    s = _settings()
    return CompanionStore(project_id=s.gcp_project_id)


def _current_week_start() -> date:
    today = date.today()
    return today - timedelta(days=today.weekday())


# ── Checkin ──────────────────────────────────────────────────────────────────

@router.post(
    "/checkin",
    response_model=CheckinSubmitResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit weekly check-in",
)
async def submit_checkin(request: CheckinSubmitRequest) -> CheckinSubmitResponse:
    try:
        s         = _settings()
        store     = _store()
        week_num  = store.next_week_number(request.candidate_id)
        week_start = _current_week_start()

        # ── 1. Real gap score from auto_gap_pipeline ──────────────────────
        gap_score = _fetch_gap_score(request.candidate_id, s)
        prev      = store.get_previous_checkin(request.candidate_id, week_num)
        gap_delta = gap_score - (prev.get("gap_score") or gap_score) if prev else 0.0

        checkin = WeeklyCheckin(
            candidate_id=request.candidate_id,
            week_number=week_num,
            week_start=week_start,
            checked_in_at=datetime.now(timezone.utc),
            hours_on_courses=request.hours_on_courses,
            courses_completed=request.courses_completed,
            applications_sent=request.applications_sent,
            interviews_scheduled=request.interviews_scheduled,
            interviews_completed=request.interviews_completed,
            gap_score=gap_score,
            gap_score_delta=gap_delta,
        )
        store.save_checkin(checkin)

        # ── 2. Generate digest in background ─────────────────────────────
        digest_id = _trigger_digest_async(checkin, request, s)

        return CheckinSubmitResponse(
            week_number=week_num,
            gap_score=gap_score,
            gap_score_delta=gap_delta,
            digest_id=digest_id,
            digest_ready=False,   # digest is async; poll /digest
            next_checkin=week_start + timedelta(days=7),
        )

    except Exception as exc:
        logger.exception(f"[companion/checkin] {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Digest reads ─────────────────────────────────────────────────────────────

@router.get(
    "/{candidate_id}/digest",
    summary="Get the latest weekly digest",
)
async def get_latest_digest(candidate_id: str):
    result = _store().get_latest_digest(candidate_id)
    if not result:
        raise HTTPException(status_code=404, detail="No digest found for this candidate")
    return result


@router.get(
    "/{candidate_id}/history",
    summary="Get all digest history for timeline / sparkline",
)
async def get_history(candidate_id: str):
    return _store().get_digest_history(candidate_id)


# ── Cloud Scheduler trigger ───────────────────────────────────────────────────

@router.post(
    "/trigger-digests",
    summary="Cloud Scheduler webhook — triggers all Monday digests",
    description=(
        "Called by Cloud Scheduler every Monday at 06:00 UTC. Internal use only.\n\n"
        "Deploy job:\n"
        "gcloud scheduler jobs create http reskillio-weekly-digest \\\n"
        "  --schedule='0 6 * * 1' \\\n"
        "  --uri='https://YOUR_CLOUD_RUN_URL/companion/trigger-digests' \\\n"
        "  --oidc-service-account-email=reskillio-scheduler@PROJECT.iam.gserviceaccount.com \\\n"
        "  --location=us-central1"
    ),
)
async def trigger_digests(
    x_cloudscheduler_jobname: Optional[str] = Header(default=None),
):
    if not x_cloudscheduler_jobname:
        raise HTTPException(status_code=403, detail="Not a scheduler request")

    logger.info("[companion] Weekly digest trigger fired by Cloud Scheduler")
    # TODO: query BQ for all active candidates due for a digest and fan out to Cloud Tasks
    return {"status": "triggered", "message": "Digest generation queued for all active candidates"}


# ── Private helpers ───────────────────────────────────────────────────────────

def _fetch_gap_score(candidate_id: str, settings) -> float:
    """
    Pull the latest gap score from the auto_gap_pipeline using the candidate's
    top market roles and skill profile. Falls back to 0.0 if unavailable.
    """
    try:
        from reskillio.agents.market_pulse_agent import run_market_pulse_agent
        from reskillio.pipelines.auto_gap_pipeline import run_auto_gap
        from reskillio.storage.profile_store import CandidateProfileStore

        profile     = CandidateProfileStore(project_id=settings.gcp_project_id).get_profile(candidate_id)
        skill_names = [s.skill_name for s in profile.skills[:15]]

        # Get current top roles from market pulse
        pulse = run_market_pulse_agent(
            skill_names=skill_names, industry="", target_role="",
            project_id=settings.gcp_project_id, region=settings.gcp_region,
        )
        top_roles = [r.role_title for r in (pulse.top_roles or [])[:3]]

        if not top_roles:
            return 0.0

        # Run auto gap and average readiness scores
        gap_result = run_auto_gap(
            top_roles=top_roles, candidate_id=candidate_id,
            project_id=settings.gcp_project_id, region=settings.gcp_region,
        )
        scores = [r.readiness_pct for r in gap_result.role_readiness]
        return round(sum(scores) / len(scores), 1) if scores else 0.0

    except Exception as exc:
        logger.warning(f"[companion] Gap score fetch failed: {exc} — using 0.0")
        return 0.0


def _fetch_intake_profile(candidate_id: str, settings) -> dict:
    """Pull intake profile from BigQuery for digest personalisation."""
    try:
        from reskillio.intake.intake_store import IntakeStore
        profile = IntakeStore(project_id=settings.gcp_project_id).get_profile(candidate_id)
        if not profile:
            return {}
        return {
            "financial_runway":      profile.financial_runway or "moderate",
            "want_next":             profile.want_next or "",
            "loved_aspects":         profile.loved_aspects or "",
            "work_identity":         profile.work_identity or "",
            "open_to_fractional":    profile.open_to_fractional,
            "persona_label":         f"{profile.work_identity or 'Professional'} · {profile.geographic_flexibility or ''}",
        }
    except Exception as exc:
        logger.warning(f"[companion] Intake profile fetch failed: {exc}")
        return {}


def _fetch_market_data(skill_names: list[str], industry: str, settings) -> dict:
    """Get latest market pulse data for the digest market section."""
    try:
        from reskillio.agents.market_pulse_agent import run_market_pulse_agent
        pulse = run_market_pulse_agent(
            skill_names=skill_names, industry=industry, target_role="",
            project_id=settings.gcp_project_id, region=settings.gcp_region,
        )
        top_roles = pulse.top_roles or []
        return {
            "top_industry":            industry,
            "open_roles_change_pct":   12 if pulse.hiring_activity == "high" else (0 if pulse.hiring_activity == "medium" else -8),
            "top_roles":               [r.role_title for r in top_roles[:3]],
            "hiring_activity":         pulse.hiring_activity,
            "summary":                 pulse.summary,
        }
    except Exception as exc:
        logger.warning(f"[companion] Market data fetch failed: {exc}")
        return {"top_industry": industry, "open_roles_change_pct": 0}


def _trigger_digest_async(checkin: WeeklyCheckin, request: CheckinSubmitRequest, settings) -> str:
    """Generate digest in a background thread; return a placeholder digest_id."""
    import uuid
    digest_id = str(uuid.uuid4())

    def _run():
        try:
            from reskillio.storage.profile_store import CandidateProfileStore
            from reskillio.companion.digest_generator import get_generator

            profile     = CandidateProfileStore(project_id=settings.gcp_project_id).get_profile(checkin.candidate_id)
            skill_names = [s.skill_name for s in profile.skills[:15]]

            intake_profile = _fetch_intake_profile(checkin.candidate_id, settings)
            market_data    = _fetch_market_data(skill_names, intake_profile.get("top_industry", ""), settings)

            candidate_name = getattr(profile, "name", None) or "Candidate"
            target_role    = intake_profile.get("want_next", "").split()[0:4]
            target_role    = " ".join(target_role) if target_role else "your target role"

            # Build active courses from request data (real integration: read from a courses table)
            active_courses = [
                {"name": c, "pct_complete": 50, "platform": ""}
                for c in request.courses_completed
            ]

            gen = get_generator(project_id=settings.gcp_project_id, region=settings.gcp_region)
            gen.generate_digest(
                checkin=checkin,
                candidate_name=candidate_name,
                target_role=target_role,
                intake_profile=intake_profile,
                gap_history=[],
                market_data=market_data,
                active_courses=active_courses,
                application_log=[a.dict() for a in request.application_logs],
            )
        except Exception as exc:
            logger.error(f"[companion] Async digest failed: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return digest_id
