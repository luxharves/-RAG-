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

    # ── V6 grounding verifier ──
    verifier_mode: str = "grounding"  # "grounding" (deterministic) | "llm" (V4 judge)
    grounding_scorer: str = "reranker"  # "reranker" (cross-encoder) | "cosine" (BGE-M3)
    grounding_scorer_floor: float = 0.1  # cross-encoder support floor
    grounding_initial_threshold: float = 0.55
    grounding_threshold_floor: float = 0.35
    grounding_threshold_decay: float = 0.9
    grounding_min_support_ratio: float = 0.7

    # ── V7 LLM gateway (retry / circuit breaker / failover / timeout) ──
    llm_timeout: float = 60.0            # per-attempt cap (s); SDK default was 600s
    llm_max_retries: int = 2             # attempts per provider = max_retries + 1
    llm_retry_base: float = 1.0
    llm_retry_multiplier: float = 2.0
    llm_retry_cap: float = 8.0
    llm_circuit_threshold: int = 3       # consecutive failures to OPEN the breaker
    llm_circuit_cooldown: float = 30.0   # s before HALF_OPEN probe
    llm_fallback_answer: str = "模型服务暂时不可用，无法回答此问题，请稍后重试。"
    # Backup providers (blank = not configured; fill all three to activate failover)
    llm_base_url_2: str = ""
    llm_api_key_2: str = ""
    llm_model_2: str = ""
    llm_base_url_3: str = ""
    llm_api_key_3: str = ""
    llm_model_3: str = ""

    # ── V9 semantic cache ──
    cache_enabled: bool = True
    cache_threshold: float = 0.9        # cosine for a semantic (paraphrase) hit
    cache_ttl_days: int | None = None   # None = never expires (fixed KB)
    cache_db_path: str = "storage/semantic_cache.db"

    @property
    def project_root(self) -> Path:
        """Absolute path to the project root directory."""
        return Path(__file__).resolve().parents[2]


# Singleton instance for the application
settings = Settings()
