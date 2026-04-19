"""tests/unit/test_matching_engine.py — Run with: pytest tests/unit/test_matching_engine.py -v"""
import pytest
from reskillio.radar.matching_engine import MatchingEngine
from reskillio.radar.models import Opportunity, EngagementType, CompanyStage, HiringSignal

engine = MatchingEngine()

def make_opp(**kw):
    d = dict(
        opportunity_id="t1", company_name="TestCo",
        company_stage=CompanyStage.SERIES_B, company_industry="logistics",
        role_title="Fractional VP Ops", engagement_type=EngagementType.FRACTIONAL,
        required_skills=["process optimization", "team leadership", "KPI reporting"],
        culture_signals=["building from scratch", "scale-up"], ideal_identity="builder",
        hiring_signal=HiringSignal.ACTIVELY_HIRING, remote_ok=True,
    )
    d.update(kw)
    return Opportunity(**d)

SKILLS = [
    {"name": "Process Optimization", "category": "technical", "confidence": 1.0},
    {"name": "Team Leadership",       "category": "soft",      "confidence": 0.9},
    {"name": "KPI Reporting",         "category": "technical", "confidence": 0.9},
]
PREFS = {
    "geographic_flexibility": "local_remote",
    "open_to_remote": True,
    "engagement_format": "fractional",
    "financial_runway": "moderate",
}
SEN = {"team_size_managed": 40, "budget_managed_millions": 200, "years_experience": 11}


class TestSkillOverlap:
    def test_perfect(self):
        o = make_opp(required_skills=["process optimization", "team leadership"])
        b, d = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert d.overlap_pct == 100.0

    def test_partial(self):
        o = make_opp(required_skills=["process optimization", "python", "tableau"])
        b, d = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert "Process Optimization" in d.matched_skills
        assert "python" in d.missing_skills

    def test_no_match(self):
        o = make_opp(required_skills=["kubernetes", "pytorch", "rust"])
        b, d = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert d.overlap_pct == 0.0


class TestTraitFit:
    def test_builder_builder(self):
        o = make_opp(culture_signals=["building from scratch", "greenfield"], ideal_identity="builder")
        b, _ = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert b.trait_fit_score >= 70

    def test_builder_enterprise_low(self):
        o = make_opp(
            company_stage=CompanyStage.ENTERPRISE,
            culture_signals=["maintain existing"],
            ideal_identity="operator",
        )
        b, _ = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert b.trait_fit_score < 60


class TestWeights:
    def test_weights_sum(self):
        o = make_opp()
        b, _ = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        manual = round(b.skill_overlap_score * 0.4 + b.trait_fit_score * 0.3 + b.context_score * 0.3, 1)
        assert abs(b.overall_score - manual) < 0.2

    def test_bounds(self):
        o = make_opp()
        b, _ = engine.score_match(SKILLS, "builder", SEN, PREFS, o)
        assert 0 <= b.overall_score <= 100
