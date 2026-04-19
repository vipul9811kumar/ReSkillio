"""
reskillio/api/routes/radar.py

  POST /radar/search           — run full radar for a candidate
  POST /radar/pitch            — generate outreach pitch for a match
  POST /radar/match/{id}/save  — save / update match status
  GET  /radar/{candidate_id}   — get saved matches
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from reskillio.radar.models import (
    RadarRequest, RadarResponse, OpportunityMatch,
    PitchRequest, PitchResponse, SaveMatchRequest,
)
from reskillio.radar.matching_engine import MatchingEngine

logger    = logging.getLogger(__name__)
router    = APIRouter(prefix="/radar", tags=["radar"])
_engine   = MatchingEngine()


def _settings():
    from config.settings import settings
    return settings


@router.post("/search", response_model=RadarResponse)
async def search_opportunities(request: RadarRequest) -> RadarResponse:
    """
    Full radar search for a candidate.
    1. Loads candidate profile + intake from BigQuery
    2. Runs RadarAgentCrew to find live opportunities
    3. Scores each via MatchingEngine
    4. Returns ranked matches (score >= 60)
    """
    try:
        s = _settings()

        # ── Load real candidate profile ───────────────────────────────────
        from reskillio.storage.profile_store import CandidateProfileStore
        profile     = CandidateProfileStore(project_id=s.gcp_project_id).get_profile(request.candidate_id)
        skill_names = [sk.skill_name for sk in (profile.skills or [])]
        candidate_skills = [
            {"name": sk.skill_name, "category": getattr(sk, "category", "general"),
             "confidence": getattr(sk, "confidence_score", 0.9)}
            for sk in (profile.skills or [])[:20]
        ]

        # ── Load intake for identity / prefs ─────────────────────────────
        candidate_identity = "builder"
        candidate_prefs    = {
            "geographic_flexibility": "local_remote",
            "open_to_remote": True,
            "engagement_format": "fractional",
            "financial_runway": "moderate",
        }
        candidate_seniority = {"team_size_managed": 0, "budget_managed_millions": 0}
        top_industry = "operations"

        try:
            from reskillio.intake.intake_store import IntakeStore
            intake = IntakeStore(project_id=s.gcp_project_id).get_profile(request.candidate_id)
            if intake:
                candidate_identity = getattr(intake, "work_identity", "builder") or "builder"
                top_industry       = getattr(intake, "industry", "operations") or "operations"
                candidate_prefs    = {
                    "geographic_flexibility": getattr(intake, "geographic_flexibility", "local_remote") or "local_remote",
                    "open_to_remote":         True,
                    "engagement_format":      getattr(intake, "engagement_format", "fractional") or "fractional",
                    "financial_runway":       getattr(intake, "financial_runway", "moderate") or "moderate",
                }
        except Exception as exc:
            logger.warning(f"[radar] Intake load failed, using defaults: {exc}")

        top_skills = skill_names[:10] or ["operations", "team leadership", "process optimization"]

        # Pull top roles from enrichment data if available (MarketPulseAgent output)
        top_roles: list[str] = []
        target_role: str | None = None
        try:
            from reskillio.intake.intake_store import IntakeStore
            intake_full = IntakeStore(project_id=s.gcp_project_id).get_profile(request.candidate_id)
            if intake_full:
                target_role = getattr(intake_full, "target_role", None)
        except Exception:
            pass

        # ── Multi-source job fetch ────────────────────────────────────────
        from reskillio.radar.job_fetcher import fetch_opportunities
        opportunities = fetch_opportunities(
            top_skills=top_skills,
            top_roles=top_roles,
            identity=candidate_identity,
            target_role=target_role,
            industry=top_industry,
            project_id=s.gcp_project_id,
            region=s.gcp_region,
        )

        # ── Score and filter ──────────────────────────────────────────────
        matches = []
        for opp in opportunities:
            if request.types and opp.engagement_type not in request.types:
                continue
            breakdown, skill_detail = _engine.score_match(
                candidate_skills=candidate_skills,
                candidate_identity=candidate_identity,
                candidate_seniority=candidate_seniority,
                candidate_prefs=candidate_prefs,
                opportunity=opp,
            )
            if breakdown.overall_score < 45:
                continue

            matches.append(OpportunityMatch(
                match_id=str(uuid.uuid4()),
                candidate_id=request.candidate_id,
                opportunity_id=opp.opportunity_id,
                opportunity=opp,
                score_breakdown=breakdown,
                skill_detail=skill_detail,
                match_reasons=_build_reasons(opp, breakdown, skill_detail),
                fit_narrative=f"{breakdown.overall_score:.0f}/100 match — {opp.engagement_type.value} opportunity",
                generated_at=datetime.now(timezone.utc),
            ))

        matches.sort(key=lambda m: m.score_breakdown.overall_score, reverse=True)
        matches = matches[:request.max_results]

        rates    = [m.opportunity.rate_floor for m in matches if m.opportunity.rate_floor]
        avg_rate = round(sum(rates) / len(rates), 0) if rates else None

        return RadarResponse(
            candidate_id=request.candidate_id,
            matches=matches,
            total_found=len(matches),
            avg_daily_rate=avg_rate,
            generated_at=datetime.now(timezone.utc),
            persona_used=f"{candidate_identity} · {top_industry}",
        )

    except Exception as e:
        logger.error(f"Radar search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pitch", response_model=PitchResponse)
async def generate_pitch(request: PitchRequest) -> PitchResponse:
    """Generate a personalised outreach pitch for a specific opportunity match."""
    try:
        s = _settings()
        from reskillio.radar.pitch_generator import get_pitch_generator
        from reskillio.storage.profile_store import CandidateProfileStore

        profile = CandidateProfileStore(project_id=s.gcp_project_id).get_profile(request.candidate_id)
        candidate_profile = {
            "top_skills":       [sk.skill_name for sk in (profile.skills or [])[:5]],
            "work_identity":    getattr(profile, "work_identity", "operator"),
            "years_experience": getattr(profile, "years_experience", "10+"),
            "scale_signal":     "large-scale operations",
            "key_achievement":  "complex operational environments",
        }

        # Pitch needs a match — for now reconstruct a minimal one from request
        # In production: load from radar_matches BQ table
        raise HTTPException(
            status_code=501,
            detail="Load match from BigQuery radar_matches table first (setup_radar_tables.py)"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/match/{match_id}/save")
async def save_match(match_id: str, request: SaveMatchRequest):
    """Save or update a match status."""
    # TODO: upsert to BigQuery radar_matches table
    return {"match_id": match_id, "status": request.status, "saved": True}


@router.get("/{candidate_id}")
async def get_saved_matches(candidate_id: str):
    """Get saved matches for a candidate from BigQuery."""
    # TODO: load from radar_matches BQ table
    return {"candidate_id": candidate_id, "matches": [], "note": "Wire radar_matches BQ table"}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_reasons(opp, breakdown, skill_detail) -> list[str]:
    reasons = []
    if breakdown.skill_overlap_score >= 80:
        skills = ", ".join(skill_detail.matched_skills[:3])
        reasons.append(f"Strong skill overlap: {skills} directly match what they need")
    if skill_detail.transferable_skills:
        t = skill_detail.transferable_skills[0]
        reasons.append(
            f"Transferable match: your {t['candidate_skill']} maps to their "
            f"{t['opportunity_skill']} (similarity {t['similarity']:.2f})"
        )
    if breakdown.trait_fit_score >= 80:
        reasons.append(
            f"Identity fit: your {opp.ideal_identity or 'profile'} matches their culture signals"
        )
    return reasons[:3]
