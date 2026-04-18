"""FastAPI application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from reskillio.api.routes import extract, resume, candidate, embeddings, jd, gap, industry, narrative, agent, market, pathway, registry, monitoring, lakehouse, analyze, enrich, prompt, intake, person_gap, companion

app = FastAPI(
    title="ReSkillio API",
    description="AI-powered career rebound platform — skill extraction and gap analysis.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://vipul9811kumar.github.io",
        "http://localhost:3000",
        "http://localhost:8080",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(extract.router)
app.include_router(resume.router)
app.include_router(candidate.router)
app.include_router(embeddings.router)
app.include_router(jd.router)
app.include_router(gap.router)
app.include_router(industry.router)
app.include_router(narrative.router)
app.include_router(agent.router)
app.include_router(market.router)
app.include_router(pathway.router)
app.include_router(registry.router)
app.include_router(monitoring.router)
app.include_router(lakehouse.router)
app.include_router(analyze.router)
app.include_router(enrich.router)
app.include_router(prompt.router)
app.include_router(intake.router)
app.include_router(person_gap.router)
app.include_router(companion.router)


@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@app.on_event("startup")
def _startup() -> None:
    logger.info("ReSkillio API starting up")