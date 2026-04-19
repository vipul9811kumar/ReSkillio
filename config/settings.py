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

    # Gemini AI Studio (simpler alternative to Vertex AI for generative calls)
    # Get a free key at https://aistudio.google.com
    gemini_api_key: str = ""

    # spaCy
    spacy_model: str = "en_core_web_lg"

    # Adzuna job search API (https://developer.adzuna.com)
    adzuna_app_id:  str = ""
    adzuna_app_key: str = ""

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
