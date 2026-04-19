"""scripts/setup_radar_tables.py — run once to create BigQuery radar tables."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from google.cloud import bigquery
from google.api_core.exceptions import NotFound

from config.settings import settings

P = settings.gcp_project_id
D = "reskillio"
c = bigquery.Client(project=P)

SCHEMAS = {
    "radar_opportunities": [
        bigquery.SchemaField("opportunity_id",   "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("company_name",      "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("company_stage",     "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("company_industry",  "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("role_title",        "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("engagement_type",   "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("rate_floor",        "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("rate_ceiling",      "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("required_skills",   "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("culture_signals",   "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("ideal_identity",    "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("remote_ok",         "BOOL",      mode="NULLABLE"),
        bigquery.SchemaField("discovered_at",     "TIMESTAMP", mode="NULLABLE"),
    ],
    "radar_matches": [
        bigquery.SchemaField("match_id",          "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("candidate_id",      "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("opportunity_id",    "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("overall_score",     "FLOAT64",   mode="REQUIRED"),
        bigquery.SchemaField("skill_score",       "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("trait_score",       "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("context_score",     "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("matched_skills",    "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("missing_skills",    "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("pitch_text",        "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("status",            "STRING",    mode="NULLABLE"),
        bigquery.SchemaField("generated_at",      "TIMESTAMP", mode="REQUIRED"),
    ],
}

print(f"=== ReSkillio Radar Setup — project: {P} ===\n")
for tid, schema in SCHEMAS.items():
    t = bigquery.Table(f"{P}.{D}.{tid}", schema=schema)
    try:
        c.get_table(t)
        print(f"  {tid} — already exists")
    except NotFound:
        c.create_table(t)
        print(f"  {tid} — created ✓")

print("\nDone.")
