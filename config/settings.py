"""Application settings loaded from environment / .env file."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # GCP
    gcp_project_id: str = ""
    gcp_region: str = "us-central1"
    google_application_credentials: str = ""

    # Vertex AI
    vertex_model_name: str = "gemini-1.5-pro-001"

    # Groq — free LLM API, no billing required (llama-3.3-70b-versatile)
    # Get a free key at https://console.groq.com/keys
    groq_api_key: str = ""

    # Gemini AI Studio (fallback if Groq key not set)
    # Get a free key at https://aistudio.google.com
    gemini_api_key: str = ""

    # Set to true to skip Vertex AI entirely and use AI Studio key for all LLM calls.
    # Use this when Vertex AI is unavailable (e.g. billing suspended, API not enabled).
    gemini_use_ai_studio: bool = False

    # spaCy
    spacy_model: str = "en_core_web_sm"

    # Adzuna job search API (https://developer.adzuna.com)
    adzuna_app_id:  str = ""
    adzuna_app_key: str = ""

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
