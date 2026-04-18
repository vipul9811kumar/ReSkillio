"""Intake conversation routes — POST /intake/start, POST /intake/turn, GET /intake/*."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger

from reskillio.models.intake import (
    IntakeProfile,
    IntakeStartRequest,
    IntakeStartResponse,
    IntakeTurnRequest,
    IntakeTurnResponse,
)

router = APIRouter(prefix="/intake", tags=["intake"])

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from config.settings import settings
        if not settings.gcp_project_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="GCP not configured — set GCP_PROJECT_ID",
            )
        from reskillio.intake.conversation_engine import IntakeConversationEngine
        _engine = IntakeConversationEngine(
            project_id=settings.gcp_project_id,
            region=settings.gcp_region,
        )
    return _engine


@router.post(
    "/start",
    response_model=IntakeStartResponse,
    status_code=status.HTTP_200_OK,
    summary="Start a new intake conversation session",
)
async def start_intake(req: IntakeStartRequest) -> IntakeStartResponse:
    engine = _get_engine()
    try:
        session_id, message, suggestions = engine.start_session(req.candidate_id)
    except Exception as exc:
        logger.exception(f"[intake/start] {exc}")
        raise HTTPException(500, f"Intake start failed: {exc}") from exc

    return IntakeStartResponse(
        session_id=session_id,
        question_n=1,
        message=message,
        suggestions=suggestions,
    )


@router.post(
    "/turn",
    response_model=IntakeTurnResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a user message and get the next intake reply",
)
async def intake_turn(req: IntakeTurnRequest) -> IntakeTurnResponse:
    engine = _get_engine()
    try:
        reply, question_n, suggestions, completed, profile = engine.process_turn(
            session_id=req.session_id,
            user_message=req.message,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"[intake/turn] {exc}")
        raise HTTPException(500, f"Intake turn failed: {exc}") from exc

    # Persist to BigQuery on completion
    if completed and profile:
        profile.completed_at = datetime.utcnow()
        _persist_async(profile)

    return IntakeTurnResponse(
        reply=reply,
        question_n=question_n,
        suggestions=suggestions,
        completed=completed,
        profile=profile,
    )


@router.get(
    "/{candidate_id}/profile",
    response_model=Optional[IntakeProfile],
    status_code=status.HTTP_200_OK,
    summary="Get the completed intake profile for a candidate",
)
async def get_intake_profile(candidate_id: str) -> Optional[IntakeProfile]:
    from config.settings import settings
    from reskillio.intake.intake_store import IntakeStore
    store = IntakeStore(project_id=settings.gcp_project_id)
    try:
        return store.get_profile(candidate_id)
    except Exception as exc:
        logger.warning(f"[intake/profile] {exc}")
        return None


@router.get(
    "/session/{session_id}/progress",
    status_code=status.HTTP_200_OK,
    summary="Get progress for an in-flight intake session",
)
async def get_session_progress(session_id: str) -> dict:
    engine = _get_engine()
    session = engine.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    from reskillio.intake.conversation_engine import _QUESTIONS
    theme = _QUESTIONS.get(session.current_question, {}).get("theme", "")

    return {
        "session_id":          session_id,
        "candidate_id":        session.candidate_id,
        "completed_questions": session.current_question - 1,
        "total_questions":     5,
        "current_theme":       theme,
        "completed":           session.completed,
    }


def _persist_async(profile: IntakeProfile) -> None:
    import threading
    from config.settings import settings
    from reskillio.intake.intake_store import IntakeStore

    def _run():
        try:
            store = IntakeStore(project_id=settings.gcp_project_id)
            store.ensure_table()
            store.upsert_profile(profile)
        except Exception as exc:
            logger.error(f"[intake] BQ persist failed: {exc}")

    threading.Thread(target=_run, daemon=True).start()
