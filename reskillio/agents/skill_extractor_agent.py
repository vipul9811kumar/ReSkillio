"""
F9 — SkillExtractorAgent (LangGraph).

Stateful graph: input_node → extract_node → validate_node → store_node
Conditional routing: retry extract if skill_count < 3.

Retry strategy
--------------
Pass 0  (first attempt):  per-section extraction, min_confidence=0.7
Pass 1+ (retry):          full concatenated text, min_confidence=0.4

Lowering the confidence threshold and removing section boundaries catches
skills that span sections or were below the high-precision first-pass bar.
"""

from __future__ import annotations

import uuid
from typing import Literal, Optional

from loguru import logger
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from reskillio.ingestion.section_parser import parse_sections
from reskillio.models.skill import ExtractionResult, Skill
from reskillio.nlp.skill_extractor import SkillExtractor, _deduplicate

MAX_RETRIES = 2
_FIRST_PASS_CONFIDENCE  = 0.7
_RETRY_PASS_CONFIDENCE  = 0.4


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Inputs ───────────────────────────────────────────────────────────
    candidate_id: str
    input_text:   str
    model_name:   str
    project_id:   str

    # ── Derived from input ────────────────────────────────────────────────
    sections: list[dict]          # [{section_type, heading, text}, ...]

    # ── Extraction outputs ────────────────────────────────────────────────
    skills:      list[dict]       # serialised Skill dicts
    skill_count: int

    # ── Control flow ─────────────────────────────────────────────────────
    retry_count: int
    max_retries: int

    # ── Final results ─────────────────────────────────────────────────────
    stored:        bool
    extraction_id: Optional[str]
    error:         Optional[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def input_node(state: AgentState) -> dict:
    """
    Validate inputs and split text into labelled sections.
    Sections become the unit of extraction in the first pass.
    """
    logger.info(f"[agent] input_node: candidate='{state['candidate_id']}'")

    text = state.get("input_text", "").strip()
    if not text:
        logger.warning("[agent] input_node: empty input_text")
        return {"error": "input_text is empty", "sections": []}

    parsed = parse_sections(text)
    sections = [
        {"section_type": st.value, "heading": heading, "text": body}
        for st, heading, body in parsed
        if body.strip()
    ]

    logger.info(f"[agent] input_node: {len(sections)} section(s) detected")
    return {"sections": sections, "error": None}


def extract_node(state: AgentState) -> dict:
    """
    Run spaCy skill extraction.

    First pass  — per-section, min_confidence=0.7 (high precision)
    Retry pass  — full text, min_confidence=0.4  (broader recall)
    """
    retry = state.get("retry_count", 0)
    logger.info(
        f"[agent] extract_node: pass={retry + 1}  "
        f"candidate='{state['candidate_id']}'"
    )

    if state.get("error"):
        return {"skills": [], "skill_count": 0, "retry_count": retry + 1}

    model_name  = state.get("model_name", "en_core_web_lg")
    sections    = state.get("sections", [])

    if retry == 0:
        # ── Pass 0: per-section, high-precision ──────────────────────────
        extractor = SkillExtractor(
            model_name=model_name,
            min_confidence=_FIRST_PASS_CONFIDENCE,
        )
        all_skills: list[Skill] = []
        for sec in sections:
            text = sec.get("text", "").strip()
            if text:
                result = extractor.extract(text)
                all_skills.extend(result.skills)
        deduped = _deduplicate(all_skills)
    else:
        # ── Pass 1+: full text, broader recall ───────────────────────────
        extractor = SkillExtractor(
            model_name=model_name,
            min_confidence=_RETRY_PASS_CONFIDENCE,
        )
        full_text = "\n\n".join(s.get("text", "") for s in sections)
        result    = extractor.extract(full_text)
        deduped   = result.skills

    skill_dicts = [s.model_dump() for s in deduped]
    logger.info(
        f"[agent] extract_node: pass={retry + 1}  "
        f"skills={len(deduped)}  "
        f"confidence_threshold="
        f"{'%.1f' % (_FIRST_PASS_CONFIDENCE if retry == 0 else _RETRY_PASS_CONFIDENCE)}"
    )

    return {
        "skills":      skill_dicts,
        "skill_count": len(deduped),
        "retry_count": retry + 1,
    }


def validate_node(state: AgentState) -> dict:
    """
    Quality gate — logs decision but makes no state changes.
    Routing is handled by route_after_validate.
    """
    skill_count = state.get("skill_count", 0)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)

    if skill_count < 3 and retry_count < max_retries:
        logger.warning(
            f"[agent] validate_node: skill_count={skill_count} < 3  "
            f"→ retry (attempt {retry_count + 1}/{max_retries})"
        )
    elif skill_count < 3:
        logger.warning(
            f"[agent] validate_node: skill_count={skill_count} < 3 "
            f"but max retries reached — proceeding to store"
        )
    else:
        logger.info(
            f"[agent] validate_node: skill_count={skill_count} ✓ — proceeding to store"
        )

    return {}


def store_node(state: AgentState) -> dict:
    """
    Persist extraction to BigQuery and refresh the candidate profile.
    Skipped (gracefully) if there was an upstream error or no project_id.
    """
    logger.info(f"[agent] store_node: candidate='{state['candidate_id']}'")

    if state.get("error"):
        logger.warning(f"[agent] store_node: skipping — upstream error: {state['error']}")
        return {"stored": False, "extraction_id": None}

    project_id = state.get("project_id", "")
    skills      = state.get("skills", [])

    if not project_id:
        logger.warning("[agent] store_node: no project_id — skipping BQ write")
        return {"stored": False, "extraction_id": None}

    try:
        from reskillio.models.skill import Skill, SkillCategory
        from reskillio.models.skill import ExtractionResult
        from reskillio.storage.bigquery_store import BigQuerySkillStore
        from reskillio.storage.profile_store import CandidateProfileStore

        skill_objects = [
            Skill(
                name=s["name"],
                category=SkillCategory(s["category"]),
                confidence=s["confidence"],
                source_text=s.get("source_text", s["name"]),
            )
            for s in skills
        ]

        full_text = "\n\n".join(
            sec.get("text", "") for sec in state.get("sections", [])
        )
        extraction = ExtractionResult(
            input_text=full_text,
            skills=skill_objects,
            model_used=state.get("model_name", "en_core_web_lg"),
        )

        bq_store = BigQuerySkillStore(project_id=project_id)
        bq_store.store_extraction(extraction, candidate_id=state["candidate_id"])

        profile_store = CandidateProfileStore(project_id=project_id)
        profile_store.upsert_profile(state["candidate_id"])

        extraction_id = str(uuid.uuid4())
        logger.info(
            f"[agent] store_node: stored {len(skills)} skills  "
            f"extraction_id={extraction_id}"
        )
        return {"stored": True, "extraction_id": extraction_id}

    except Exception as exc:
        logger.error(f"[agent] store_node: BQ write failed: {exc}")
        return {
            "stored": False,
            "extraction_id": None,
            "error": f"storage failed: {exc}",
        }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_validate(
    state: AgentState,
) -> Literal["extract_node", "store_node"]:
    """Retry if skill_count < 3 and retries remain; otherwise store."""
    if (
        state.get("skill_count", 0) < 3
        and state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES)
    ):
        return "extract_node"
    return "store_node"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("input_node",    input_node)
    workflow.add_node("extract_node",  extract_node)
    workflow.add_node("validate_node", validate_node)
    workflow.add_node("store_node",    store_node)

    workflow.add_edge(START,           "input_node")
    workflow.add_edge("input_node",    "extract_node")
    workflow.add_edge("extract_node",  "validate_node")
    workflow.add_conditional_edges(
        "validate_node",
        route_after_validate,
        {"extract_node": "extract_node", "store_node": "store_node"},
    )
    workflow.add_edge("store_node", END)

    return workflow


# Compiled singleton — reused across requests
_graph = build_graph().compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_skill_extractor_agent(
    candidate_id: str,
    input_text: str,
    project_id: str = "",
    model_name: str = "en_core_web_lg",
    max_retries: int = MAX_RETRIES,
) -> AgentState:
    """
    Run the SkillExtractorAgent graph and return the final state.

    Parameters
    ----------
    candidate_id:   Unique candidate identifier.
    input_text:     Raw text to extract skills from (resume text, section text, etc.).
    project_id:     GCP project for BigQuery persistence. Pass "" to skip storage.
    model_name:     spaCy model name.
    max_retries:    Maximum number of retry attempts if skill_count < 3.

    Returns
    -------
    AgentState — final graph state including skills, retry_count, stored flag.
    """
    initial: AgentState = {
        "candidate_id": candidate_id,
        "input_text":   input_text,
        "model_name":   model_name,
        "project_id":   project_id,
        "sections":     [],
        "skills":       [],
        "skill_count":  0,
        "retry_count":  0,
        "max_retries":  max_retries,
        "stored":       False,
        "extraction_id": None,
        "error":        None,
    }

    logger.info(
        f"[agent] Starting SkillExtractorAgent: candidate='{candidate_id}' "
        f"text_len={len(input_text)} max_retries={max_retries}"
    )

    final_state: AgentState = _graph.invoke(initial)

    logger.info(
        f"[agent] Done: skills={final_state['skill_count']}  "
        f"retries={final_state['retry_count'] - 1}  "  # retry_count starts at 0, incremented per pass
        f"stored={final_state['stored']}"
    )
    return final_state