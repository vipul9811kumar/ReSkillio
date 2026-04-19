# ReSkillio

**AI-powered career rebound platform** — upload a resume, get a complete career intelligence report: extracted skills, gap analysis, industry fit, a Gemini-written narrative, a 90-day roadmap, a personalised weekly digest, and a live opportunity radar for fractional / consulting / advisory roles.

Built on **Google Cloud** (BigQuery, Vertex AI, Gemini 2.5 Flash, Cloud Run, Cloud Tasks, Cloud Scheduler) with **FastAPI**, **spaCy** NLP, **LangGraph**, **CrewAI**, and a BigQuery medallion lakehouse.

---

## Live URLs

| Surface | URL |
|---------|-----|
| **Try it (demo app)** | https://vipul9811kumar.github.io/ReSkillio/app.html |
| **Intake conversation** | https://vipul9811kumar.github.io/ReSkillio/intake.html |
| **Weekly Companion digest** | https://vipul9811kumar.github.io/ReSkillio/companion.html |
| **Opportunity Radar** | https://vipul9811kumar.github.io/ReSkillio/radar.html |
| **Cloud Run API** | https://reskillio-10933517215.us-central1.run.app |
| **Swagger docs** | https://reskillio-10933517215.us-central1.run.app/docs |

---

## What It Does

### Core Analysis Pipeline (`POST /analyze`)

| Stage | What happens | GCP service |
|-------|-------------|-------------|
| **1. Extract** | spaCy PhraseMatcher + NER pulls 200+ skills from resume text | BigQuery |
| **1.5 Trait inference** | Gemini infers archetype (Builder/Operator/Fixer/Connector/Expert), identity statement, work values | Vertex AI Gemini |
| **2. Gap analysis** | Exact + semantic similarity vs JD; gap score 0–100 | Vertex AI Embeddings |
| **3. Industry match** | BQML cosine distance against 8 industry centroid vectors | BigQuery ML |
| **4. Narrative** | Gemini RAG-grounded 3-sentence career story | Vertex AI Gemini |
| **5. Pathway** | CrewAI 2-agent crew researches courses + 90-day roadmap (opt-in, ~45s) | Gemini + DuckDuckGo |

All stages run in one `POST /analyze` call. Each stage is fail-safe — a downstream failure never blocks earlier results.

### Depth Phases (beyond the core pipeline)

| Phase | What it adds |
|-------|-------------|
| **Intake** | 5-question Gemini conversational intake — goals, identity, financial runway, preferences. Stored in BigQuery. Feeds richer person gap and radar matching. |
| **Person Gap** | Single Gemini call maps candidate skills against stated goals (not just a JD). Returns aligned skills, growth skills, surprise transfers, readiness score, recommended actions. |
| **Weekly Companion** | Monday check-in → Gemini digest (narrative + 4 action items). Gap score pulled from auto-gap pipeline. Cloud Scheduler fans out via Cloud Tasks every Monday 06:00 UTC. |
| **Opportunity Radar** | CrewAI 2-agent crew searches live postings. Scores each opportunity: 40% skill overlap (3-tier matching) + 30% trait fit (identity vs culture signals) + 30% context (seniority, geo, format). |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    GitHub Pages (static frontend)                     │
│   app.html · intake.html · companion.html · radar.html               │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTPS
┌──────────────────────────────▼───────────────────────────────────────┐
│                     FastAPI  (Cloud Run)                              │
│                                                                       │
│  POST /analyze          POST /intake/start, /intake/turn             │
│  POST /companion/checkin  GET /companion/{id}/digest                 │
│  POST /companion/trigger-digests  (Cloud Scheduler webhook)          │
│  POST /companion/{id}/generate-digest  (Cloud Tasks target)          │
│  POST /radar/search     POST /radar/pitch                            │
│  POST /person-gap       GET  /intake/{id}/profile                    │
│  + 20 core pipeline endpoints                                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
  ┌───────────────┐   ┌─────────────────┐  ┌──────────────────┐
  │   BigQuery    │   │   Vertex AI     │  │  Cloud           │
  │               │   │                 │  │  Infrastructure  │
  │  reskillio.*  │   │  gemini-2.5-    │  │                  │
  │  ├ skill_ext  │   │  flash          │  │  Cloud Scheduler │
  │  ├ profiles   │   │                 │  │  (Monday 06UTC)  │
  │  ├ embeddings │   │  text-embed-004 │  │                  │
  │  ├ jd_*       │   │                 │  │  Cloud Tasks     │
  │  ├ industry_v │   │  Model Registry │  │  (digest fan-out)│
  │  ├ intake_*   │   └─────────────────┘  │                  │
  │  ├ weekly_    │                        │  Cloud Monitoring│
  │  │  checkins  │                        │  (drift metrics) │
  │  ├ companion_ │                        └──────────────────┘
  │  │  digests   │
  │  ├ radar_opp  │
  │  └ radar_     │
  │    matches    │
  │               │
  │  Medallion    │
  │  ├ BRONZE     │
  │  ├ SILVER     │
  │  └ GOLD       │
  └───────────────┘
```

---

## GCP Service Map

| GCP Service | Used for |
|-------------|----------|
| **Cloud Run** | API hosting (`reskillio-api`) |
| **BigQuery** | Skill storage, profiles, embeddings, JD catalog, industry vectors, intake profiles, companion check-ins/digests, radar matches, medallion lakehouse |
| **Vertex AI Embeddings** | Skill vector embedding (`text-embedding-004`, 768-dim) |
| **Vertex AI Gemini** | Narrative, trait inference, person gap, digest generation, pitch generation (`gemini-2.5-flash`) |
| **Vertex AI Model Registry** | Versioned spaCy skill extractor |
| **Cloud Scheduler** | Monday 06:00 UTC digest trigger |
| **Cloud Tasks** | Per-candidate digest fan-out queue (`reskillio-digest-queue`) |
| **Cloud Storage** | Model artifacts, taxonomy JSON |
| **Cloud Build** | CI/CD — deploy and retraining pipelines |
| **Cloud Monitoring** | Drift metrics + alert policy |

---

## API Reference

Base URL: `https://reskillio-10933517215.us-central1.run.app`  
Interactive docs: `/docs` (Swagger) · `/redoc` (ReDoc)

### Core Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Full 5-stage analysis (skills → gap → industry → narrative → pathway) |
| `POST` | `/enrich` | Background enrichment: trait inference + auto-gap + market pulse |
| `POST` | `/extract` | Extract skills from raw text |
| `POST` | `/resume/upload` | Upload PDF resume |
| `GET`  | `/candidate/{id}/profile` | Aggregated skill profile |
| `POST` | `/gap` | Gap analysis vs a stored JD |
| `GET`  | `/industry/match/{id}` | Industry fit scores (8 industries) |
| `POST` | `/narrative` | Gemini RAG career narrative |
| `POST` | `/market/analyze` | CrewAI real-time skill demand analysis |
| `POST` | `/pathway/plan` | CrewAI 90-day reskilling roadmap |

### Intake

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/intake/start` | Start a 5-question intake conversation |
| `POST` | `/intake/turn` | Send a message, get next question + suggestions |
| `GET`  | `/intake/{id}/profile` | Retrieve completed intake profile |
| `GET`  | `/intake/session/{sid}/progress` | Current question number |

### Person Gap

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/person-gap` | Gemini-powered gap analysis grounded in stated goals and identity |

### Weekly Companion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/companion/checkin` | Submit weekly check-in; triggers async digest generation |
| `GET`  | `/companion/{id}/digest` | Get latest digest |
| `GET`  | `/companion/{id}/history` | All digests (for sparkline) |
| `POST` | `/companion/trigger-digests` | Cloud Scheduler webhook — fans out to Cloud Tasks |
| `POST` | `/companion/{id}/generate-digest` | Cloud Tasks target — generate one candidate's digest |

### Opportunity Radar

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/radar/search` | Run full radar: agent search + scoring for a candidate |
| `POST` | `/radar/pitch` | Generate personalised outreach pitch for a match |
| `POST` | `/radar/match/{id}/save` | Save / update match status |

### Infrastructure

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/monitoring/drift/recent` | Recent drift records |
| `GET`  | `/lakehouse/status` | Row counts across Bronze/Silver/Gold |
| `POST` | `/lakehouse/gold/refresh` | Recompute Gold tables |
| `GET`  | `/lakehouse/gold/readiness/{id}` | Candidate readiness index |

---

## Agents

| Agent | Type | Description |
|-------|------|-------------|
| **SkillExtractorAgent** | LangGraph | Stateful retry graph: section-by-section extraction, retries on full text if < 3 skills |
| **MarketPulseAgent** | CrewAI | Real-time skill demand via DuckDuckGo; outputs demand score + trend per skill |
| **PathwayPlannerAgent** | CrewAI 2-agent | Researcher finds courses; Planner builds 3-phase 90-day roadmap |
| **IntakeConversationEngine** | Gemini | 5-question conversation (temp 0.75 chat + temp 0.05 structured extraction per turn) |
| **DigestGenerator** | Gemini | 2 calls per digest: narrative (temp 0.6) + action items JSON (temp 0.2) |
| **RadarAgentCrew** | CrewAI 2-agent | Hunter searches live postings via ddgs; Analyst extracts structured fields |
| **PitchGenerator** | Gemini | 3 calls: outreach pitch + engagement tips + warm intro angle |

---

## Scoring

### Gap Score (0–100)
Weighted average of readiness across top 3 target roles from MarketPulseAgent + auto-gap pipeline.

### Opportunity Radar Score (0–100)
```
40% × skill_overlap   — 3-tier: exact name → normalised → semantic (cosine ≥ 0.75)
30% × trait_fit       — identity (Builder/Operator/Fixer) vs culture signals + stage fit matrix
30% × context_score   — seniority, budget managed, geo flexibility, engagement format, financial runway
```
Only matches scoring ≥ 60 are returned.

### Gold Readiness Index
```
MIN(100,
  avg_match_score    × 0.40   -- JD alignment
  + industry_coverage × 0.30  -- domain fit
  + avg_confidence   × 0.20   -- extraction quality
  + breadth_score    × 0.10   -- skill breadth
)
```

---

## BigQuery Tables

| Dataset | Table | Purpose |
|---------|-------|---------|
| `reskillio` | `skill_extractions` | Raw spaCy extraction results |
| `reskillio` | `candidate_profiles` | Aggregated skill profiles |
| `reskillio` | `skill_embeddings` | 768-dim Vertex AI embeddings |
| `reskillio` | `jd_catalog` | Stored job descriptions |
| `reskillio` | `industry_vectors` | 8 industry centroid vectors (BQML) |
| `reskillio` | `candidate_intake_profiles` | Completed intake profiles |
| `reskillio` | `weekly_checkins` | Monday check-in records |
| `reskillio` | `companion_digests` | Generated weekly digests |
| `reskillio` | `radar_opportunities` | Curated opportunity pool |
| `reskillio` | `radar_matches` | Per-candidate scored matches |
| `reskillio_bronze` | `raw_resume_ingestion` | Append-only raw ingestion |
| `reskillio_silver` | `candidate_skills` | Validated, deduplicated |
| `reskillio_gold` | `candidate_readiness` | Computed readiness index |

---

## Getting Started

### Prerequisites

- Python 3.12+
- GCP project with billing enabled
- Service account with: `BigQuery Admin`, `Vertex AI User`, `Storage Admin`, `Cloud Tasks Admin`, `Monitoring Editor`

### Local Setup

```bash
# 1. Clone and install
git clone https://github.com/vipul9811kumar/ReSkillio.git
cd ReSkillio
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 2. Configure environment
cp .env.example .env
# Set: GCP_PROJECT_ID, GCP_REGION, GOOGLE_APPLICATION_CREDENTIALS

# 3. Authenticate locally
gcloud auth application-default login

# 4. Bootstrap all GCP resources
python scripts/setup_gcp.py
python scripts/build_industry_vectors.py
python scripts/setup_companion_tables.py   # BigQuery companion tables
python scripts/setup_radar_tables.py       # BigQuery radar tables

# 5. Start the API
uvicorn reskillio.api.main:app --reload --port 8000
```

### One-Time Cloud Infrastructure

```bash
# Cloud Tasks queue for digest fan-out
gcloud tasks queues create reskillio-digest-queue \
  --project=reskillio-dev-2026 \
  --location=us-central1 \
  --max-concurrent-dispatches=10 \
  --max-dispatches-per-second=2

# Cloud Scheduler job — Monday digests
gcloud scheduler jobs create http reskillio-weekly-digest \
  --project=reskillio-dev-2026 \
  --schedule="0 6 * * 1" \
  --uri="https://reskillio-10933517215.us-central1.run.app/companion/trigger-digests" \
  --message-body="{}" \
  --headers="Content-Type=application/json" \
  --oidc-service-account-email=reskillio-sa@reskillio-dev-2026.iam.gserviceaccount.com \
  --location=us-central1
```

### Deploy to Cloud Run

```bash
gcloud builds submit --config=cloudbuild.deploy.yaml
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GCP_PROJECT_ID` | yes | — | GCP project ID |
| `GCP_REGION` | no | `us-central1` | Vertex AI region |
| `GOOGLE_APPLICATION_CREDENTIALS` | yes | — | Path to service account JSON |
| `SPACY_MODEL` | no | `en_core_web_lg` | spaCy NER model |
| `LOG_LEVEL` | no | `INFO` | Logging verbosity |
| `ENVIRONMENT` | no | `development` | `development` or `production` |

---

## Project Structure

```
ReSkillio/
├── reskillio/
│   ├── api/
│   │   ├── main.py                    # FastAPI app + all router registration
│   │   └── routes/
│   │       ├── analyze.py             # POST /analyze
│   │       ├── enrich.py              # POST /enrich (background enrichment)
│   │       ├── prompt.py              # POST /prompt (UI-facing orchestration)
│   │       ├── intake.py              # POST /intake/start, /turn, GET /profile
│   │       ├── person_gap.py          # POST /person-gap
│   │       ├── companion.py           # POST /companion/checkin, GET /digest, trigger
│   │       ├── radar.py               # POST /radar/search, /pitch
│   │       ├── extract.py             # POST /extract
│   │       ├── gap.py                 # POST /gap
│   │       ├── narrative.py           # POST /narrative
│   │       ├── market.py              # POST /market/analyze
│   │       ├── pathway.py             # POST /pathway/plan
│   │       └── ...                    # + 8 more route modules
│   ├── pipelines/
│   │   ├── analyze_pipeline.py        # 5-stage orchestrator
│   │   ├── auto_gap_pipeline.py       # Auto gap vs top market roles
│   │   ├── trait_inference_pipeline.py
│   │   ├── person_gap_pipeline.py     # Goals-grounded gap analysis
│   │   ├── skill_pipeline.py
│   │   ├── gap_pipeline.py
│   │   ├── industry_match_pipeline.py
│   │   └── narrative_pipeline.py
│   ├── agents/
│   │   ├── skill_extractor_agent.py   # LangGraph retry graph
│   │   ├── market_pulse_agent.py      # CrewAI real-time demand
│   │   └── pathway_planner_agent.py   # CrewAI 2-agent crew
│   ├── intake/
│   │   ├── conversation_engine.py     # 5-question Gemini intake
│   │   └── intake_store.py            # BigQuery intake profiles
│   ├── companion/
│   │   ├── models.py                  # WeeklyCheckin, WeeklyDigest, ActionItem
│   │   ├── checkin_store.py           # BigQuery companion tables
│   │   └── digest_generator.py        # Gemini narrative + action items
│   ├── radar/
│   │   ├── models.py                  # Opportunity, OpportunityMatch, scoring models
│   │   ├── matching_engine.py         # 3-dim scoring algorithm
│   │   ├── radar_agent.py             # CrewAI 2-agent opportunity hunter
│   │   └── pitch_generator.py         # Gemini pitch + tips + intro angle
│   ├── models/                        # Pydantic v2 response models
│   │   ├── analyze.py
│   │   ├── intake.py
│   │   ├── person_gap.py
│   │   └── ...
│   └── storage/                       # BigQuery store classes
│       ├── bigquery_store.py
│       ├── profile_store.py
│       ├── embedding_store.py
│       └── lakehouse.py
├── docs/                              # GitHub Pages frontend
│   ├── app.html                       # Main demo / analysis flow
│   ├── intake.html                    # 5-question intake chat UI
│   ├── companion.html                 # Weekly digest UI
│   ├── radar.html                     # Opportunity radar UI
│   ├── index.html
│   └── architecture.html
├── scripts/
│   ├── setup_gcp.py
│   ├── setup_companion_tables.py      # Create companion BQ tables
│   ├── setup_radar_tables.py          # Create radar BQ tables
│   ├── build_industry_vectors.py
│   └── ...
├── tests/
│   └── unit/
│       ├── test_companion.py
│       ├── test_matching_engine.py
│       └── ...
├── config/
│   └── settings.py
├── cloudbuild.deploy.yaml             # Cloud Run deploy
├── cloudbuild.yaml                    # CI/CD retraining
├── Dockerfile
└── requirements.txt
```

---

## Key Design Decisions

**Per-stage fail-safety** — each pipeline stage is wrapped in `try/except`. A downstream failure never loses already-extracted results. Every stage reports `success`, `duration_ms`, and `error`.

**Option B async pattern** — `/analyze` returns immediately with skills + trait profile. Heavy enrichment (auto-gap, market pulse, person gap) runs in background via `/enrich`, polled by the frontend.

**Three-dimensional opportunity scoring** — skill overlap alone would match anyone with supply chain knowledge to every logistics posting. Trait fit penalises a Builder identity scoring against an enterprise "maintain existing" culture signal by ~30 points even at 90% skill overlap. Context score further adjusts for financial runway vs engagement type.

**Identity × Stage fit matrix** — `IDENTITY_FIT` and `STAGE_FIT` in `matching_engine.py` are the primary tuning surface. As real match conversion data comes in, these weights update without touching the scoring formula.

**Cloud Tasks for digest fan-out** — Monday Cloud Scheduler fires once, queries BQ for all candidates due, enqueues one Cloud Tasks HTTP task per candidate. Each task independently calls `/companion/{id}/generate-digest` with OIDC auth. 30-minute deadline per task, 2/sec dispatch rate.

**Gemini SDK standardisation** — all pipelines use `google.genai` (`genai.Client(vertexai=True, project=..., location=...)`) with `thinking_budget=0` for speed. The old `vertexai.generative_models` SDK is not used anywhere.

**CrewAI LLM via litellm** — `crewai.LLM(model="vertex_ai/gemini-2.5-flash", ...)` uses litellm (already a dependency) rather than `langchain-google-vertexai`, avoiding an extra package.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Built by [vipul9811kumar](https://github.com/vipul9811kumar) · Google Cloud + spaCy + Gemini 2.5 Flash + CrewAI + LangGraph*
