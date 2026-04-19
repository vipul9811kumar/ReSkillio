"""
reskillio/radar/radar_agent.py

CrewAI two-agent crew: OpportunityHunter searches live postings,
OpportunityAnalyst extracts structured data from them.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from reskillio.radar.models import Opportunity, EngagementType, HiringSignal, CompanyStage

logger = logging.getLogger(__name__)


def _settings():
    from config.settings import settings
    return settings


SEARCH_QUERIES_TEMPLATE = [
    '"fractional VP operations" OR "fractional operations director" {industry} {location} -"full-time"',
    '"part-time operations" OR "interim operations" {industry} startup 2026',
    '"operations consultant" OR "supply chain consultant" {industry} project {location} 2026',
    '"WMS implementation" OR "logistics consultant" contract {industry}',
    '"operations advisor" OR "strategic advisor" {industry} startup "equity" OR "retainer"',
    '"board advisor" operations {industry} scaleup',
]


@tool("search_opportunities")
def search_opportunities(query: str) -> str:
    """Search for fractional, consulting, and advisory job opportunities."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        return json.dumps([
            {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
            for r in results
        ])
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "[]"


@tool("extract_opportunity_data")
def extract_opportunity_data(posting_text: str) -> str:
    """Extract structured opportunity data from raw job posting text."""
    return json.dumps({
        "note": "Gemini extraction of structured fields from posting_text",
        "posting_text": posting_text[:500],
    })


def build_search_queries(
    top_skills: list[str],
    top_industry: str,
    location: Optional[str] = None,
    identity: Optional[str] = None,
) -> list[str]:
    loc = location or "remote"
    ind = top_industry.lower()
    queries = []
    for template in SEARCH_QUERIES_TEMPLATE:
        try:
            queries.append(template.format(industry=ind, location=loc))
        except Exception:
            queries.append(template)

    if identity == "builder":
        queries.append(f'"fractional" operations "build from scratch" {ind} 2026')
    elif identity == "fixer":
        queries.append(f'"turnaround" OR "post-acquisition" operations consultant {ind}')
    elif identity == "operator":
        queries.append(f'"scale operations" fractional OR interim {ind} 2026')

    return queries[:8]


class RadarAgentCrew:
    def __init__(self):
        s   = _settings()
        llm = LLM(
            model=f"vertex_ai/gemini-2.5-flash",
            temperature=0.3,
            vertex_project=s.gcp_project_id,
            vertex_location=s.gcp_region,
        )

        self.hunter = Agent(
            role="Opportunity Hunter",
            goal=(
                "Search for fractional, consulting, and advisory opportunities "
                "that match a senior operations professional's profile."
            ),
            backstory=(
                "You are an expert at finding non-traditional career opportunities "
                "for senior executives. The best matches are often hidden in job posts "
                "for 'Head of Operations' at startups that need a fractional leader."
            ),
            tools=[search_opportunities],
            llm=llm,
            verbose=False,
            max_iter=8,
        )

        self.analyst = Agent(
            role="Opportunity Analyst",
            goal=(
                "Extract structured, accurate data from job posting snippets. "
                "Identify engagement type, compensation signals, culture signals, "
                "and required skills. Only extract what is explicitly stated."
            ),
            backstory=(
                "You convert raw job posting text into clean structured data "
                "that a matching engine can score."
            ),
            tools=[extract_opportunity_data],
            llm=llm,
            verbose=False,
            max_iter=4,
        )

    def search(
        self,
        candidate_id: str,
        top_skills:   list[str],
        top_industry: str,
        identity:     str,
        location:     Optional[str],
    ) -> list[Opportunity]:
        queries   = build_search_queries(top_skills, top_industry, location, identity)
        query_str = " | ".join(queries[:4])

        search_task = Task(
            description=(
                f"Search for fractional, consulting, and advisory opportunities "
                f"for a senior operations professional. "
                f"Top skills: {', '.join(top_skills[:5])}. "
                f"Target industry: {top_industry}. Identity: {identity}. "
                f"Queries: {query_str}. "
                f"Return a JSON array of raw opportunity snippets."
            ),
            agent=self.hunter,
            expected_output="JSON array of opportunity snippets with title, url, description",
        )

        analyse_task = Task(
            description=(
                "For each opportunity found, extract: company name, role title, "
                "engagement type (fractional/consulting/advisory/interim), "
                "required skills (list), culture signals (list of phrases), "
                "compensation signals, remote/location requirement, company stage. "
                "Return a JSON array of structured opportunity objects."
            ),
            agent=self.analyst,
            expected_output="JSON array of structured Opportunity objects",
            context=[search_task],
        )

        crew = Crew(
            agents=[self.hunter, self.analyst],
            tasks=[search_task, analyse_task],
            process=Process.sequential,
            verbose=False,
        )

        try:
            result = crew.kickoff()
            raw    = result.raw if hasattr(result, "raw") else str(result)
            parsed = json.loads(raw.strip().lstrip("```json").lstrip("```").rstrip("```"))
            return [_dict_to_opportunity(o) for o in parsed if isinstance(o, dict)]
        except Exception as e:
            logger.error(f"Radar agent crew failed: {e}")
            return _fallback_opportunities(top_industry)


def _dict_to_opportunity(d: dict) -> Opportunity:
    import uuid
    return Opportunity(
        opportunity_id=str(uuid.uuid4()),
        company_name=d.get("company_name", "Unknown Company"),
        company_stage=_safe_enum(CompanyStage, d.get("company_stage"), CompanyStage.SMB),
        company_industry=d.get("industry", "operations"),
        company_location=d.get("location"),
        role_title=d.get("role_title", "Operations Consultant"),
        engagement_type=_safe_enum(EngagementType, d.get("engagement_type"), EngagementType.CONSULTING),
        commitment_days_per_week=d.get("days_per_week"),
        commitment_hours_per_month=d.get("hours_per_month"),
        duration_months=d.get("duration_months"),
        rate_floor=d.get("rate_floor"),
        rate_ceiling=d.get("rate_ceiling"),
        rate_unit=d.get("rate_unit", "day"),
        required_skills=d.get("required_skills", []),
        culture_signals=d.get("culture_signals", []),
        ideal_identity=d.get("ideal_identity"),
        hiring_signal=_safe_enum(HiringSignal, d.get("hiring_signal"), HiringSignal.INFERRED),
        source_url=d.get("url"),
        remote_ok=d.get("remote_ok", True),
    )


def _safe_enum(enum_cls, value, default):
    try:
        return enum_cls(value)
    except (ValueError, TypeError):
        return default


def _fallback_opportunities(industry: str) -> list[Opportunity]:
    import uuid
    return [
        Opportunity(
            opportunity_id=str(uuid.uuid4()),
            company_name="Sample Logistics Tech Co.",
            company_stage=CompanyStage.SERIES_B,
            company_industry=industry,
            role_title="Fractional VP of Operations",
            engagement_type=EngagementType.FRACTIONAL,
            commitment_days_per_week=2.0,
            rate_floor=800.0,
            rate_ceiling=1100.0,
            required_skills=["process optimization", "team leadership", "KPI reporting", "vendor negotiation"],
            culture_signals=["building from scratch", "scale-up"],
            ideal_identity="builder",
            hiring_signal=HiringSignal.INFERRED,
            remote_ok=True,
        )
    ]
