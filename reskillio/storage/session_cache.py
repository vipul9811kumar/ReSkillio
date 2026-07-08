"""
In-memory session cache for candidate skill names.

Eliminates BQ round-trips between /analyze and downstream pipelines
(enrich, person-gap, narrative, industry-match, radar) during a single
analysis session. Railway is single-instance so a module-level dict is safe.

Key: candidate_id (UUID per upload)
Value: list of skill name strings from run_skill_extraction
"""
from __future__ import annotations
import threading

_lock: threading.Lock = threading.Lock()
_store: dict[str, list[str]] = {}


def put(candidate_id: str, skill_names: list[str]) -> None:
    with _lock:
        _store[candidate_id] = list(skill_names)


def get(candidate_id: str) -> list[str]:
    with _lock:
        return list(_store.get(candidate_id, []))
