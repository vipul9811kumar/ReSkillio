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

    prompt = f"""Rate how well this skill set matches each industry. Return JSON only.

Skills: {skills_str}

Industries:
{industries_list}

JSON format (no preamble, no markdown):
{{"scores":[{{"industry":"<key>","score":<0-100>}}]}}

Include all 8 industries. Higher score = stronger match."""

    system = "Respond with valid JSON only. No markdown, no explanation, no preamble."
    raw = call_gemini(prompt, system, project_id, region, temperature=0.0, max_tokens=500)

    # Try to extract a JSON object even if the model adds preamble text
    data = None
    # Strip markdown fences first
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    # Try direct parse
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Extract the first {...} block if direct parse failed
    if data is None:
        m = re.search(r'\{[\s\S]*"scores"[\s\S]*\}', cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    # Hard fallback: assign rough scores from keyword matching if Groq returned garbage
    if data is None or not data.get("scores"):
        logger.warning("[industry-match] Groq response unparseable — using keyword fallback")
        skill_str_lower = skills_str.lower()
        keyword_map = {
            "data_ai":        ["python", "machine learning", "tensorflow", "pytorch", "nlp", "data science", "ai", "ml", "scikit"],
            "cloud_devops":   ["aws", "azure", "gcp", "kubernetes", "docker", "terraform", "devops", "ci/cd", "jenkins"],
            "fintech":        ["financial", "banking", "payments", "trading", "risk", "compliance", "fintech"],
            "healthcare":     ["clinical", "ehr", "healthcare", "medical", "hospital", "pharma"],
            "ecommerce":      ["e-commerce", "retail", "shopify", "marketplace", "logistics"],
            "operations":     ["operations", "supply chain", "procurement", "logistics", "lean", "six sigma"],
            "cyber_security": ["security", "soc", "penetration", "firewall", "siem", "vulnerability"],
            "media_content":  ["content", "marketing", "seo", "social media", "creative", "design"],
        }
        fallback_scores = []
        for ind_key, keywords in keyword_map.items():
            hits = sum(1 for kw in keywords if kw in skill_str_lower)
            score = min(85, hits * 15 + 10)
            fallback_scores.append({"industry": ind_key, "score": score})
        data = {"scores": fallback_scores}

    scored = sorted(data["scores"], key=lambda x: x.get("score", 0), reverse=True)
    scores = [
        IndustryScore(
            rank=i + 1,
            industry=row["industry"],
            industry_label=_INDUSTRY_LABELS.get(row["industry"], row["industry"]),
            match_score=float(row.get("score", 0)),
            cosine_distance=round(1.0 - float(row.get("score", 0)) / 100.0, 6),
        )
        for i, row in enumerate(scored)
        if row.get("industry") in _INDUSTRY_LABELS
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

    # Read candidate skills — session cache first (fast path), then BQ fallbacks
    skill_names_for_fallback: list[str] = []
    from reskillio.storage.session_cache import get as _cache_get
    skill_names_for_fallback = _cache_get(candidate_id)

    # If session cache has skills, skip BQ path entirely — BQ Sandbox DML is blocked
    # and the candidate vector would fail anyway. Go straight to Groq scoring.
    if skill_names_for_fallback:
        logger.info(
            f"[industry-match] Session cache hit ({len(skill_names_for_fallback)} skills) "
            "— using Groq scoring directly"
        )
        return _groq_industry_match(candidate_id, skill_names_for_fallback, project_id, region)

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