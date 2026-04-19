"""
F5 — Gap Analysis Engine.

Compare a candidate's skill profile against a JD's required skills:

  1. Exact match      — skill name appears in both.
                        Credit = candidate confidence_avg (0–1).
  2. Transferable     — unmatched JD skill has cosine similarity >= SOFT_LOWER
                        with a candidate skill.
                        Credit = similarity (hard zone ≥ threshold) or
                                 similarity × 0.5 (soft zone SOFT_LOWER–threshold).
  3. Preferred bonus  — JD preferred skills the candidate has exactly.
                        Credit = PREFERRED_WEIGHT per skill.
  4. Missing          — no exact or semantic match found. Credit = 0.

gap_score = min(100,
    (exact_credit + transfer_credit + preferred_credit) /
    (total_required + PREFERRED_WEIGHT × total_preferred)
    × 100
)

Two BQ round-trips for embeddings are collapsed into one.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from reskillio.embeddings.vertex_embedder import VertexEmbedder, skill_text, EMBEDDING_MODEL
from reskillio.models.gap import GapAnalysisResult, TransferableSkill
from reskillio.models.jd import Industry, JDExtractionResult
from reskillio.pipelines.jd_pipeline import run_jd_pipeline
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.jd_store import JDStore
from reskillio.storage.profile_store import CandidateProfileStore

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.75   # hard transferable zone: full similarity credit
_SOFT_LOWER           = 0.65   # soft transferable zone: interpolated × 0.5 weight
_PREFERRED_WEIGHT     = 0.30   # preferred-skill bonus relative to required


# ---------------------------------------------------------------------------
# Cosine similarity (pure numpy, no external dep)
# ---------------------------------------------------------------------------

def _cosine_sim(a: list[float], b: list[float]) -> float:
    av = np.array(a, dtype=np.float32)
    bv = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(av) * np.linalg.norm(bv)
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Recommendation text
# ---------------------------------------------------------------------------

def _recommendation(score: float, missing: list[str]) -> str:
    top_missing = ", ".join(missing[:3])
    suffix = f" Focus on: {top_missing}." if top_missing else ""

    if score >= 85:
        return f"Strong match — you meet most requirements for this role.{suffix}"
    if score >= 65:
        return f"Good foundation — a few targeted skill gaps to address.{suffix}"
    if score >= 45:
        return f"Moderate gap — upskilling in key areas is recommended.{suffix}"
    if score >= 25:
        return f"Significant gap — consider adjacent roles while building missing skills.{suffix}"
    return f"Large gap — this may be a stretch target; focus on foundational skills first.{suffix}"


# ---------------------------------------------------------------------------
# Embedding helpers — single BQ fetch for all needed skills
# ---------------------------------------------------------------------------

def _fetch_all_embeddings(
    jd_skills: list[str],
    candidate_skills_obj,       # list[ProfiledSkill]
    embedding_store: EmbeddingStore,
    embedder: VertexEmbedder,
    category_map_jd: dict[str, str],
    category_map_cand: dict[str, str],
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Fetch embeddings for JD skills and candidate skills in a single BQ call.
    Embeds any missing vectors on-the-fly and upserts to the catalog.
    Returns (jd_vecs, cand_vecs) dicts keyed by lowercase skill name.
    """
    cand_names = [s.skill_name for s in candidate_skills_obj]
    all_names  = list({*jd_skills, *cand_names})                 # deduplicated

    cached = embedding_store.get_embeddings_batch(all_names)     # single BQ round-trip

    missing_jd   = [n for n in jd_skills  if n.lower() not in cached]
    missing_cand = [n for n in cand_names if n.lower() not in cached]
    to_embed     = list({*missing_jd, *missing_cand})

    if to_embed:
        logger.debug(f"Embedding {len(to_embed)} uncached skills on the fly")
        combined_cat = {**category_map_cand, **category_map_jd}
        pairs    = [(n, combined_cat.get(n.lower(), "unknown")) for n in to_embed]
        embedded = embedder.embed_skills(pairs)
        embedding_store.upsert_embeddings(
            skills=embedded,
            embed_text_fn=skill_text,
            model_name=EMBEDDING_MODEL,
        )
        for name, _cat, vec in embedded:
            cached[name.lower()] = vec

    jd_vecs   = {n.lower(): cached[n.lower()] for n in jd_skills  if n.lower() in cached}
    cand_vecs = {n.lower(): cached[n.lower()] for n in cand_names if n.lower() in cached}
    return jd_vecs, cand_vecs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_gap_analysis(
    candidate_id: str,
    project_id: str,
    jd_id: str | None = None,
    jd_text: str | None = None,
    jd_title: str | None = None,
    jd_company: str | None = None,
    industry: Industry | None = None,
    similarity_threshold: float = _SIMILARITY_THRESHOLD,
    region: str = "us-central1",
) -> GapAnalysisResult:
    """
    Compare candidate profile against a JD and return a structured gap report.

    Provide either `jd_id` (stored JD) or `jd_text` (inline, not stored).
    """
    if not jd_id and not jd_text:
        raise ValueError("Provide either jd_id or jd_text.")

    profile_store   = CandidateProfileStore(project_id=project_id)
    embedding_store = EmbeddingStore(project_id=project_id)
    embedder        = VertexEmbedder(project_id=project_id, region=region)

    # ------------------------------------------------------------------
    # 1. Candidate profile
    # ------------------------------------------------------------------
    profile = profile_store.get_profile(candidate_id)
    if not profile.skills:
        raise ValueError(f"No profile found for candidate '{candidate_id}'.")

    candidate_skills = {s.skill_name.lower(): s for s in profile.skills}
    logger.info(
        f"Gap analysis: candidate='{candidate_id}' ({len(candidate_skills)} skills) "
        f"jd_id={jd_id or 'inline'}"
    )

    # ------------------------------------------------------------------
    # 2. JD skills — required + preferred
    # ------------------------------------------------------------------
    result_meta: dict = {}
    required_jd_skills: list[str] = []
    preferred_jd_skills: list[str] = []

    if jd_id:
        jd_store = JDStore(project_id=project_id)
        rows = jd_store.get_jd(jd_id)
        if not rows:
            raise ValueError(f"JD '{jd_id}' not found in BigQuery.")
        for r in rows:
            if r["requirement"] == "required":
                required_jd_skills.append(r["skill_name"])
            elif r["requirement"] == "preferred":
                preferred_jd_skills.append(r["skill_name"])
        result_meta = {
            "jd_id":      jd_id,
            "jd_title":   rows[0].get("title"),
            "jd_company": rows[0].get("company"),
            "industry":   rows[0].get("industry"),
            "seniority":  rows[0].get("seniority"),
        }
    else:
        jd_result: JDExtractionResult = run_jd_pipeline(
            text=jd_text,
            title=jd_title,
            company=jd_company,
            industry=industry,
            store=None,
        )
        required_jd_skills  = [s.name for s in jd_result.required_skills]
        preferred_jd_skills = [s.name for s in jd_result.preferred_skills]
        result_meta = {
            "jd_id":      None,
            "jd_title":   jd_title,
            "jd_company": jd_company,
            "industry":   jd_result.industry.value,
            "seniority":  jd_result.seniority.value,
        }

    if not required_jd_skills:
        logger.warning("JD has no required skills — returning zero gap score.")
        return GapAnalysisResult(
            candidate_id=candidate_id,
            gap_score=100.0,
            total_required=0,
            recommendation="No required skills detected in this JD.",
            **result_meta,
        )

    logger.info(
        f"  JD required: {len(required_jd_skills)}  preferred: {len(preferred_jd_skills)}"
    )

    # ------------------------------------------------------------------
    # 3. Exact matching (case-insensitive) with confidence weighting
    # ------------------------------------------------------------------
    matched: list[str] = []
    unmatched_jd: list[str] = []
    exact_credit = 0.0

    for jd_skill in required_jd_skills:
        cand = candidate_skills.get(jd_skill.lower())
        if cand is not None:
            matched.append(jd_skill)
            # confidence_avg is 0–1; clamp to [0.5, 1.0] so a low-confidence
            # extraction still gives meaningful credit rather than near-zero.
            exact_credit += max(0.5, min(1.0, cand.confidence_avg))
        else:
            unmatched_jd.append(jd_skill)

    logger.info(f"  Exact matches: {len(matched)} (credit={exact_credit:.2f})  Unmatched: {len(unmatched_jd)}")

    # ------------------------------------------------------------------
    # 4. Preferred skill bonus (exact-match only, lightweight)
    # ------------------------------------------------------------------
    preferred_credit = 0.0
    preferred_matched: list[str] = []

    for pref_skill in preferred_jd_skills:
        if pref_skill.lower() in candidate_skills:
            preferred_matched.append(pref_skill)
            preferred_credit += _PREFERRED_WEIGHT

    if preferred_matched:
        logger.info(f"  Preferred matches: {preferred_matched} (bonus={preferred_credit:.2f})")

    # ------------------------------------------------------------------
    # 5. Semantic matching for unmatched required JD skills
    # ------------------------------------------------------------------
    transferable: list[TransferableSkill] = []
    missing: list[str] = []
    transfer_credit = 0.0

    if unmatched_jd:
        jd_category_map   = {s.lower(): "unknown" for s in unmatched_jd}
        cand_category_map = {s.skill_name.lower(): s.category.value for s in profile.skills}

        jd_vecs, cand_vecs = _fetch_all_embeddings(
            jd_skills=unmatched_jd,
            candidate_skills_obj=profile.skills,
            embedding_store=embedding_store,
            embedder=embedder,
            category_map_jd=jd_category_map,
            category_map_cand=cand_category_map,
        )

        soft_lower = min(_SOFT_LOWER, similarity_threshold - 0.01)

        for jd_skill in unmatched_jd:
            jd_vec = jd_vecs.get(jd_skill.lower())
            if jd_vec is None:
                missing.append(jd_skill)
                continue

            best_sim   = 0.0
            best_match = ""

            for cand_skill_obj in profile.skills:
                cand_vec = cand_vecs.get(cand_skill_obj.skill_name.lower())
                if cand_vec is None:
                    continue
                sim = _cosine_sim(jd_vec, cand_vec)
                if sim > best_sim:
                    best_sim   = sim
                    best_match = cand_skill_obj.skill_name

            if best_sim >= similarity_threshold:
                # Hard transferable zone: credit = actual similarity (0.75–1.0)
                transferable.append(TransferableSkill(
                    jd_skill=jd_skill,
                    candidate_skill=best_match,
                    similarity=round(best_sim, 4),
                ))
                transfer_credit += best_sim
                logger.debug(
                    f"  Transferable (hard): '{jd_skill}' ← '{best_match}' "
                    f"(sim={best_sim:.3f}, credit={best_sim:.3f})"
                )
            elif best_sim >= soft_lower:
                # Soft zone: partial credit, shown as a low-confidence transferable
                soft_credit = best_sim * 0.5
                transferable.append(TransferableSkill(
                    jd_skill=jd_skill,
                    candidate_skill=best_match,
                    similarity=round(best_sim, 4),
                ))
                transfer_credit += soft_credit
                logger.debug(
                    f"  Transferable (soft): '{jd_skill}' ← '{best_match}' "
                    f"(sim={best_sim:.3f}, credit={soft_credit:.3f})"
                )
            else:
                missing.append(jd_skill)

    logger.info(
        f"  Transferable: {len(transferable)} (credit={transfer_credit:.2f})  "
        f"Missing: {len(missing)}"
    )

    # ------------------------------------------------------------------
    # 6. Gap score
    #
    # Denominator includes preferred skills at their fractional weight so
    # having them can push the score above what required-only allows,
    # but a candidate with zero preferred skills isn't penalised.
    # ------------------------------------------------------------------
    total_required  = len(required_jd_skills)
    total_preferred = len(preferred_jd_skills)
    denominator     = total_required + _PREFERRED_WEIGHT * total_preferred

    numerator = exact_credit + transfer_credit + preferred_credit
    gap_score = round(min(100.0, numerator / denominator * 100), 1)

    logger.info(
        f"  Score: {numerator:.2f} / {denominator:.2f} → gap_score={gap_score}"
    )

    return GapAnalysisResult(
        candidate_id=candidate_id,
        gap_score=gap_score,
        total_required=total_required,
        matched_skills=matched,
        transferable_skills=sorted(transferable, key=lambda t: t.similarity, reverse=True),
        missing_skills=missing,
        similarity_threshold=similarity_threshold,
        recommendation=_recommendation(gap_score, missing),
        **result_meta,
    )
