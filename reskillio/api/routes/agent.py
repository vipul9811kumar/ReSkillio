"""POST /agent/extract — LangGraph SkillExtractorAgent endpoint."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from reskillio.agents.skill_extractor_agent import run_skill_extractor_agent
from config.settings import settings

router = APIRouter(prefix="/agent", tags=["agent"])


class AgentExtractRequest(BaseModel):
    candidate_id: str  = Field(..., description="Candidate identifier")
    text:         str  = Field(..., min_length=10, description="Resume or section text")
    model_name:   str  = Field(default="en_core_web_lg")
    max_retries:  int  = Field(default=2, ge=0, le=5)
    store:        bool = Field(default=True, description="Persist results to BigQuery")


class AgentExtractResponse(BaseModel):
    candidate_id:  str
    skill_count:   int
    skills:        list[dict]
    stored:        bool
    retry_count:   int           # number of extra passes taken (0 = succeeded first try)
    extraction_id: Optional[str]
    error:         Optional[str]
    graph_path:    str           # human-readable trace of nodes executed


@router.post(
    "/extract",
    response_model=AgentExtractResponse,
    status_code=status.HTTP_200_OK,
)
def agent_extract(body: AgentExtractRequest) -> AgentExtractResponse:
    """
    Extract skills via the LangGraph SkillExtractorAgent.

    Graph: input_node → extract_node → validate_node → store_node
    Conditional routing: retries extraction (with broader strategy) if
    skill_count < 3, up to max_retries times.

    Returns full agent state including retry trace.
    """
    logger.info(
        f"Agent extract request: candidate='{body.candidate_id}' "
        f"text_len={len(body.text)} max_retries={body.max_retries}"
    )

    try:
        project_id = settings.gcp_project_id if body.store else ""
        final = run_skill_extractor_agent(
            candidate_id=body.candidate_id,
            input_text=body.text,
            project_id=project_id,
            model_name=body.model_name,
            max_retries=body.max_retries,
        )
    except Exception as exc:
        logger.error(f"Agent extract failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent failed: {exc}",
        ) from exc

    passes = final["retry_count"]
    path_nodes = ["input_node"]
    for i in range(passes):
        path_nodes.append(f"extract_node(pass={i + 1})")
        path_nodes.append("validate_node")
    path_nodes.append("store_node")
    graph_path = " → ".join(path_nodes)

    return AgentExtractResponse(
        candidate_id=final["candidate_id"],
        skill_count=final["skill_count"],
        skills=final["skills"],
        stored=final["stored"],
        retry_count=max(0, passes - 1),  # passes = total extract calls; retries = passes - 1
        extraction_id=final.get("extraction_id"),
        error=final.get("error"),
        graph_path=graph_path,
    )