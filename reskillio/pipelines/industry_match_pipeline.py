"""
F6 — Industry match pipeline.

Builds a frequency-weighted candidate centroid embedding, then calls
ML.DISTANCE against all 8 pre-built industry vectors in BigQuery.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from reskillio.embeddings.ai_studio_embedder import AIStudioEmbedder as VertexEmbedder, skill_text, EMBEDDING_MODEL
from reskillio.models.industry import IndustryMatchResult
from reskillio.storage.embedding_store import EmbeddingStore
from reskillio.storage.industry_vector_store import IndustryVectorStore
from reskillio.storage.profile_store import CandidateProfileStore


def _build_candidate_vector(
    candidate_id: str,
    profile_store: CandidateProfileStore,
    embedding_store: EmbeddingStore,
    embedder: VertexEmbedder,
) -> list[float]:
    """
    Compute a frequency-weighted centroid embedding for a candidate.

    centroid = Σ(frequency_i × embedding_i) / Σ(frequency_i)

    Embeds any skills not yet in the catalog on the fly.
    """
    profile = profile_store.get_profile(candidate_id)
    if not profile.skills:
        raise ValueError(f"No profile found for candidate '{candidate_id}'.")

    skill_names = [s.skill_name for s in profile.skills]
    freq_map    = {s.skill_name.lower(): s.frequency for s in profile.skills}
    cat_map     = {s.skill_name.lower(): s.category.value for s in profile.skills}

    # Fetch cached embeddings
    vecs = embedding_store.get_embeddings_batch(skill_names)

    # Embed missing ones on the fly
    missing = [n for n in skill_names if n.lower() not in vecs]
    if missing:
        logger.debug(f"Embedding {len(missing)} uncached candidate skills")
        pairs    = [(n, cat_map.get(n.lower(), "unknown")) for n in missing]
        embedded = embedder.embed_skills(pairs)
        embedding_store.upsert_embeddings(
            skills=embedded,
            embed_text_fn=skill_text,
            model_name=EMBEDDING_MODEL,
        )
        for name, _cat, vec in embedded:
            vecs[name.lower()] = vec

    # Weighted centroid
    total_freq = sum(freq_map.values())
    dims = len(next(iter(vecs.values())))
    centroid = np.zeros(dims, dtype=np.float64)

    covered = 0
    for name_lower, vec in vecs.items():
        w = freq_map.get(name_lower, 1) / total_freq
        centroid += w * np.array(vec, dtype=np.float64)
        covered += 1

    logger.info(
        f"Candidate vector built from {covered}/{len(skill_names)} skills "
        f"(freq-weighted centroid)"
    )
    return centroid.tolist()


def _groq_industry_match(
    candidate_id: str,
    skill_names: list[str],
    project_id: str,
    region: str,
) -> IndustryMatchResult:
    """
    Groq-based fallback when BQ industry vectors are unavailable.
    Asks the LLM to score the candidate's skills against the 8 known industries.
    """
    import json, re
    from datetime import datetime
    from reskillio.models.industry import _INDUSTRY_LABELS, IndustryScore, IndustryMatchResult
    from reskillio.utils.gemini import call_gemini

    industries_list = "\n".join(f'  "{k}": "{v}"' for k, v in _INDUSTRY_LABELS.items())
    skills_str = ", ".join(skill_names[:20]) or "no skills provided"

    prompt = f"""Given this candidate skill set, rate how well they match each industry on a scale of 0-100.

Candidate skills: {skills_str}

Industries to score:
{industries_list}

Return ONLY valid JSON (no markdown):
{{
  "scores": [
    {{"industry": "<key>", "score": <0-100>}},
    ...
  ]
}}

Score all 8 industries. Higher = stronger match. Base scores only on the skill set."""

    system = "You are a recruiting analyst. Respond only with valid JSON."
    raw = call_gemini(prompt, system, project_id, region, temperature=0.1, max_tokens=400)
    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    data = json.loads(raw)

    scored = sorted(data["scores"], key=lambda x: x["score"], reverse=True)
    scores = [
        IndustryScore(
            rank=i + 1,
            industry=row["industry"],
            industry_label=_INDUSTRY_LABELS.get(row["industry"], row["industry"]),
            match_score=float(row["score"]),
            cosine_distance=round(1.0 - row["score"] / 100.0, 6),
        )
        for i, row in enumerate(scored)
    ]
    top = scores[0] if scores else None
    logger.info(
        f"[industry-match] Groq fallback — top='{top.industry_label if top else '?'}' "
        f"score={top.match_score if top else 0:.1f}"
    )
    return IndustryMatchResult(
        candidate_id=candidate_id,
        top_industry=top.industry if top else "",
        top_industry_label=top.industry_label if top else "",
        scores=scores,
        computed_at=datetime.utcnow(),
        method="groq_llm_scoring",
    )


def run_industry_match(
    candidate_id: str,
    project_id: str,
    region: str = "us-central1",
) -> IndustryMatchResult:
    """
    Score a candidate against all 8 industries.
    Primary path: BQ embedding centroid + ML.DISTANCE against industry vectors.
    Fallback: Groq LLM scoring from skill list (no BQ dependencies).
    """
    profile_store   = CandidateProfileStore(project_id=project_id)
    embedding_store = EmbeddingStore(project_id=project_id)
    industry_store  = IndustryVectorStore(project_id=project_id)
    embedder        = VertexEmbedder(project_id=project_id, region=region)

    logger.info(f"Industry match started for candidate='{candidate_id}'")

    # Read candidate skills — session cache first, then candidate_profiles, then skill_extractions
    skill_names_for_fallback: list[str] = []
    from reskillio.storage.session_cache import get as _cache_get
    skill_names_for_fallback = _cache_get(candidate_id)

    if not skill_names_for_fallback:
        try:
            profile = profile_store.get_profile(candidate_id)
            skill_names_for_fallback = [s.skill_name for s in profile.skills]
        except Exception:
            pass

    if not skill_names_for_fallback:
        try:
            from reskillio.storage.bigquery_store import BigQuerySkillStore
            rows = BigQuerySkillStore(project_id=project_id).get_skills_for_candidate(candidate_id)
            skill_names_for_fallback = [r["skill_name"] for r in rows]
        except Exception:
            pass

    try:
        # 1. Build candidate centroid vector (needs candidate_profiles)
        candidate_vec = _build_candidate_vector(
            candidate_id, profile_store, embedding_store, embedder
        )

        # 2. BQML cosine scoring against pre-built industry vectors
        logger.info("Running ML.DISTANCE against industry_vectors...")
        scored_rows = industry_store.score_candidate(candidate_vec)

        if not scored_rows:
            raise RuntimeError("No industry vectors found — falling back to Groq scoring")

        logger.info("Industry match scores (BQ):")
        for row in scored_rows:
            logger.info(
                f"  {row['industry']:25} "
                f"match_score={row['match_score']:5.1f}  "
                f"cosine_dist={row['cosine_distance']:.4f}"
            )
        return IndustryMatchResult.from_bq_rows(candidate_id, scored_rows)

    except Exception as exc:
        logger.warning(f"[industry-match] BQ path failed ({exc}) — using Groq fallback")
        if not skill_names_for_fallback:
            raise RuntimeError("No skills available for industry match") from exc
        return _groq_industry_match(candidate_id, skill_names_for_fallback, project_id, region)