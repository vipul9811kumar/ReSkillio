"""
reskillio/radar/matching_engine.py

Score = 40% skill_overlap + 30% trait_fit + 30% context_score
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from reskillio.radar.models import (
    Opportunity, MatchScoreBreakdown, SkillMatchDetail, EngagementType,
)

logger = logging.getLogger(__name__)

W_SKILL   = 0.40
W_TRAIT   = 0.30
W_CONTEXT = 0.30

IDENTITY_FIT = {
    "builder": {
        "building from scratch": 1.0,
        "create the playbook":   1.0,
        "greenfield":            0.9,
        "startup":               0.85,
        "scale-up":              0.8,
        "established process":   0.4,
        "maintain existing":     0.3,
    },
    "operator": {
        "optimise existing":     1.0,
        "scale operations":      1.0,
        "established process":   0.9,
        "building from scratch": 0.5,
        "scale-up":              0.85,
        "enterprise":            0.8,
    },
    "fixer": {
        "turnaround":            1.0,
        "cleanup":               1.0,
        "post-acquisition":      0.95,
        "broken process":        0.9,
        "building from scratch": 0.4,
    },
    "connector": {
        "partnerships":            1.0,
        "vendor relationships":    0.95,
        "stakeholder management":  0.9,
        "cross-functional":        0.85,
    },
    "expert": {
        "deep domain expertise": 1.0,
        "thought leadership":    0.9,
        "advisory":              0.95,
        "specialist":            0.9,
    },
}

STAGE_FIT = {
    ("builder",  "series_a"):   1.0,
    ("builder",  "series_b"):   0.9,
    ("builder",  "seed"):       0.85,
    ("builder",  "smb"):        0.75,
    ("builder",  "pe_backed"):  0.5,
    ("builder",  "enterprise"): 0.3,

    ("operator", "series_b"):   1.0,
    ("operator", "series_c"):   0.95,
    ("operator", "pe_backed"):  0.9,
    ("operator", "enterprise"): 0.85,
    ("operator", "smb"):        0.8,
    ("operator", "series_a"):   0.7,

    ("fixer",    "pe_backed"):  1.0,
    ("fixer",    "series_c"):   0.9,
    ("fixer",    "enterprise"): 0.85,
    ("fixer",    "smb"):        0.75,
    ("fixer",    "series_b"):   0.7,

    ("connector","series_b"):   0.9,
    ("connector","series_c"):   0.85,
    ("connector","enterprise"): 0.9,

    ("expert",   "advisory"):   1.0,
    ("expert",   "series_a"):   0.8,
    ("expert",   "enterprise"): 0.85,
}


class MatchingEngine:

    def score_match(
        self,
        candidate_skills:    list[dict],
        candidate_identity:  str,
        candidate_seniority: dict,
        candidate_prefs:     dict,
        opportunity:         Opportunity,
        skill_embeddings:    Optional[dict] = None,
    ) -> tuple[MatchScoreBreakdown, SkillMatchDetail]:

        skill_score, skill_detail = self._score_skill_overlap(
            candidate_skills, opportunity, skill_embeddings
        )
        trait_score   = self._score_trait_fit(candidate_identity, opportunity)
        context_score = self._score_context(candidate_seniority, candidate_prefs, opportunity)

        overall = round(
            skill_score   * W_SKILL +
            trait_score   * W_TRAIT +
            context_score * W_CONTEXT,
            1,
        )

        breakdown = MatchScoreBreakdown(
            skill_overlap_score=round(skill_score, 1),
            trait_fit_score=round(trait_score, 1),
            context_score=round(context_score, 1),
            overall_score=overall,
        )
        return breakdown, skill_detail

    # ── Skill overlap (40%) ──────────────────────────────────────────────────

    def _score_skill_overlap(
        self,
        candidate_skills: list[dict],
        opportunity:      Opportunity,
        embeddings:       Optional[dict],
    ) -> tuple[float, SkillMatchDetail]:
        candidate_names = {s["name"].lower(): s for s in candidate_skills}
        required = [r.lower() for r in opportunity.required_skills]

        matched, missing, transferable = [], [], []

        for req in required:
            if req in candidate_names:
                matched.append(candidate_names[req]["name"])
                continue

            norm_req   = self._normalise(req)
            norm_match = next(
                (c for c in candidate_names if self._normalise(c) == norm_req), None
            )
            if norm_match:
                matched.append(candidate_names[norm_match]["name"])
                continue

            if embeddings:
                best_sim, best_candidate = 0.0, None
                req_vec = embeddings.get(req)
                if req_vec:
                    for cname, cskill in candidate_names.items():
                        c_vec = embeddings.get(cname)
                        if c_vec:
                            sim = self._cosine(req_vec, c_vec)
                            if sim > best_sim:
                                best_sim, best_candidate = sim, cskill["name"]
                if best_sim >= 0.75:
                    transferable.append({
                        "opportunity_skill": req,
                        "candidate_skill":   best_candidate,
                        "similarity":        round(best_sim, 2),
                    })
                    matched.append(best_candidate)
                    continue

            missing.append(req)

        total       = len(required)
        overlap_pct = round((len(matched) / total * 100) if total > 0 else 0, 1)

        transfer_names  = [t["candidate_skill"] for t in transferable]
        direct_score    = len([m for m in matched if m not in transfer_names]) / max(total, 1) * 100
        transfer_score  = len(transferable) / max(total, 1) * 100 * 0.75
        skill_score     = min(direct_score + transfer_score, 100)

        return skill_score, SkillMatchDetail(
            matched_skills=matched,
            missing_skills=missing,
            transferable_skills=transferable,
            overlap_pct=overlap_pct,
        )

    @staticmethod
    def _normalise(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]", "", s.lower())

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x**2 for x in a))
        mag_b = math.sqrt(sum(x**2 for x in b))
        return dot / (mag_a * mag_b + 1e-9)

    # ── Trait fit (30%) ──────────────────────────────────────────────────────

    def _score_trait_fit(self, identity: str, opp: Opportunity) -> float:
        if not identity:
            return 50.0

        signal_scores = []
        identity_map  = IDENTITY_FIT.get(identity, {})
        for signal in opp.culture_signals:
            signal_l = signal.lower()
            best = max(
                (v for k, v in identity_map.items() if k in signal_l),
                default=0.5,
            )
            signal_scores.append(best)

        if opp.ideal_identity and opp.ideal_identity == identity:
            signal_scores.append(1.0)
        elif opp.ideal_identity and opp.ideal_identity != identity:
            signal_scores.append(0.3)

        culture_fit     = (sum(signal_scores) / len(signal_scores) * 100) if signal_scores else 60.0
        stage_key       = (identity, opp.company_stage.value)
        stage_fit_score = STAGE_FIT.get(stage_key, 0.65) * 100

        return round(culture_fit * 0.6 + stage_fit_score * 0.4, 1)

    # ── Context score (30%) ──────────────────────────────────────────────────

    def _score_context(self, seniority: dict, prefs: dict, opp: Opportunity) -> float:
        scores = []

        team_managed = seniority.get("team_size_managed", 0)
        scores.append(1.0 if team_managed >= 20 else 0.7 if team_managed >= 5 else 0.4)

        budget = seniority.get("budget_managed_millions", 0)
        scores.append(1.0 if budget >= 50 else 0.75 if budget >= 10 else 0.5 if budget >= 1 else 0.3)

        geo_flex = prefs.get("geographic_flexibility", "local_only")
        if opp.remote_ok and prefs.get("open_to_remote", True):
            scores.append(1.0)
        elif geo_flex in ("national", "global"):
            scores.append(0.9)
        elif geo_flex == "regional" and opp.location_required:
            scores.append(0.7)
        elif geo_flex == "local_only" and not opp.remote_ok:
            scores.append(0.5)
        else:
            scores.append(0.6)

        pref_format = prefs.get("engagement_format", "any")
        if pref_format == "any" or pref_format == opp.engagement_type.value:
            scores.append(1.0)
        elif pref_format in ("fractional", "consulting") and opp.engagement_type.value in ("fractional", "consulting"):
            scores.append(0.8)
        else:
            scores.append(0.5)

        runway = prefs.get("financial_runway", "moderate")
        if runway in ("critical", "tight") and opp.engagement_type == EngagementType.CONSULTING:
            scores.append(1.0)
        elif runway in ("comfortable", "secure") and opp.engagement_type == EngagementType.ADVISORY:
            scores.append(0.9)
        else:
            scores.append(0.65)

        return round(sum(scores) / len(scores) * 100, 1)
