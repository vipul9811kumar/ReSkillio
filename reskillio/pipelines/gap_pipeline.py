"""
F5 — Gap Analysis Engine.

Compare a candidate's skill profile against a JD's required skills:

  1. Exact match    — skill name appears in both (full credit).
  2. Transferable   — unmatched JD skill has cosine similarity >= threshold
                      with a candidate skill (partial credit, weight 0.7).
  3. Missing        — no exact or semantic match found (no credit).

gap_score = min(100, (matched + 0.7 * transferable) / total_required * 100)
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

_TRANSFERABLE_WEIGHT = 0.7


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
# Embedding helpers
# ---------------------------------------------------------------------------

def _ensure_embeddings(
    skill_names: list[str],
    category_map: dict[str, str],
    embedding_store: EmbeddingStore,
    embedder: VertexEmbedder,
) -> dict[str, list[float]]:
    """
    Return embeddings for all skill_names.
    Fetches from BQ first; embeds on the fly for any missing ones
    and upserts them to the catalog.
    """
    cached = embedding_store.get_embeddings_batch(skill_names)
    missing_names = [n for n in skill_names if n.lower() not in cached]

    if missing_names:
        logger.debug(f"Embedding {len(missing_names)} uncached skills on the fly")
        pairs = [(n, category_map.get(n.lower(), "unknown")) for n in missing_names]
        embedded = embedder.embed_skills(pairs)

        embedding_store.upsert_embeddings(
            skills=embedded,
            embed_text_fn=skill_text,
            model_name=EMBEDDING_MODEL,
        )
        for name, _cat, vec in embedded:
            cached[name.lower()] = vec

    return cached


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
    similarity_threshold: float = 0.75,
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
    # 2. JD required skills
    # ------------------------------------------------------------------
    result_meta: dict = {}

    if jd_id:
        jd_store = JDStore(project_id=project_id)
        rows = jd_store.get_jd(jd_id)
        if not rows:
            raise ValueError(f"JD '{jd_id}' not found in BigQuery.")
        required_jd_skills = [
            r["skill_name"] for r in rows if r["requirement"] == "required"
        ]
        result_meta = {
            "jd_id":    jd_id,
            "jd_title": rows[0].get("title"),
            "jd_company": rows[0].get("company"),
            "industry": rows[0].get("industry"),
            "seniority": rows[0].get("seniority"),
        }
    else:
        jd_result: JDExtractionResult = run_jd_pipeline(
            text=jd_text,
            title=jd_title,
            company=jd_company,
            industry=industry,
            store=None,  # don't persist inline JDs
        )
        required_jd_skills = [s.name for s in jd_result.required_skills]
        result_meta = {
            "jd_id":    None,
            "jd_title": jd_title,
            "jd_company": jd_company,
            "industry": jd_result.industry.value,
            "seniority": jd_result.seniority.value,
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

    logger.info(f"  JD required skills ({len(required_jd_skills)}): {required_jd_skills}")

    # ------------------------------------------------------------------
    # 3. Exact matching (case-insensitive name)
    # ------------------------------------------------------------------
    matched: list[str] = []
    unmatched_jd: list[str] = []

    for jd_skill in required_jd_skills:
        if jd_skill.lower() in candidate_skills:
            matched.append(jd_skill)
        else:
            unmatched_jd.append(jd_skill)

    logger.info(f"  Exact matches: {len(matched)}  Unmatched JD skills: {len(unmatched_jd)}")

    # ------------------------------------------------------------------
    # 4. Semantic matching for unmatched JD skills
    # ------------------------------------------------------------------
    transferable: list[TransferableSkill] = []
    missing: list[str] = []

    if unmatched_jd:
        # Build category maps (used for on-the-fly embedding)
        jd_category_map = {s.lower(): "unknown" for s in unmatched_jd}
        cand_category_map = {
            s.skill_name.lower(): s.category.value for s in profile.skills
        }

        # Fetch / embed all needed vectors in two batch calls
        jd_vecs   = _ensure_embeddings(unmatched_jd, jd_category_map, embedding_store, embedder)
        cand_vecs = _ensure_embeddings(
            [s.skill_name for s in profile.skills], cand_category_map, embedding_store, embedder
        )

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
                transferable.append(
                    TransferableSkill(
                        jd_skill=jd_skill,
                        candidate_skill=best_match,
                        similarity=round(best_sim, 4),
                    )
                )
                logger.debug(
                    f"  Transferable: '{jd_skill}' ← '{best_match}' (sim={best_sim:.3f})"
                )
            else:
                missing.append(jd_skill)

    logger.info(
        f"  Transferable: {len(transferable)}  Missing: {len(missing)}"
    )

    # ------------------------------------------------------------------
    # 5. Gap score
    # ------------------------------------------------------------------
    total    = len(required_jd_skills)
    raw      = (len(matched) + _TRANSFERABLE_WEIGHT * len(transferable)) / total * 100
    gap_score = round(min(100.0, raw), 1)

    logger.info(f"  gap_score={gap_score}")

    return GapAnalysisResult(
        candidate_id=candidate_id,
        gap_score=gap_score,
        total_required=total,
        matched_skills=matched,
        transferable_skills=sorted(transferable, key=lambda t: t.similarity, reverse=True),
        missing_skills=missing,
        similarity_threshold=similarity_threshold,
        recommendation=_recommendation(gap_score, missing),
        **result_meta,
    )