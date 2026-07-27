"""Unified configuration via environment variables and .env file.

All secrets are read from environment or .env — never hardcoded.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env and environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI-compatible LLM endpoint ──
    llm_base_url: str = "https://api.example.com/v1"
    llm_api_key: str = "replace-me"
    llm_model: str = "replace-me"

    # ── OpenAI-compatible VLM (Qwen3-VL) endpoint ──
    vlm_base_url: str = "https://api.example.com/v1"
    vlm_api_key: str = "replace-me"
    vlm_model: str = "replace-me"

    # ── Milvus ──
    milvus_uri: str = "milvus.db"  # Milvus Lite local file
    milvus_token: str = ""

    # ── Local model settings ──
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-large"
    model_device: str = "auto"  # "auto", "cuda", or "cpu"

    # ── Application ──
    log_level: str = "INFO"
    data_dir: str = "data"
    storage_dir: str = "storage"

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return Path(__file__).resolve().parents[2]


# Singleton instance for the application
settings = Settings()
